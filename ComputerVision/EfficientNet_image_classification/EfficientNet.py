import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from Augmentation import PerformAtMostN, MyRandomAdjustSharpness
from collections import OrderedDict

def kernel_size_to_str(kernel_size):
    if isinstance(kernel_size, int):
        return f'{kernel_size}x{kernel_size}'
    return f'{kernel_size[0]}x{kernel_size[1]}'

class SepConvBlock(nn.Module):
    def __init__(self, kernel_size, channels, bn_moment):
        # perform depthwise conv, pointwise conv, batch norm and relu
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
        )
        self.pw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )
        self.bn = nn.BatchNorm2d(num_features=channels, momentum=bn_moment)

    def forward(self, batch):
        output = self.dw_conv(batch)
        output = self.pw_conv(output)
        return F.relu(self.bn(output))


class ConvBNReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bn_moment, activation=True, kernel_size=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = activation

        # by default kernel is 1, according to the paper
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels, bn_moment)

    def forward(self, batch):
        output = self.conv(batch)
        output = self.bn(output)
        if self.activation:
            return F.relu(output)
        return output

    def get_name(self):
        kernel_str = kernel_size_to_str(self.kernel_size)
        result = f'Conv({kernel_str}),in_ch={self.in_channels},out_ch={self.out_channels},str={self.stride},BatchNorm'
        if self.activation:
            result += ',Relu'
        return result


class MBConv(nn.Module):
    def __init__(self, exp_factor, kernel_size, in_channels, out_channels, stride, bn_moment):
        super().__init__()
        self.exp_channels = exp_factor * channels
        self.conv_bn_relu1 = ConvBNReluBlock(
            in_channels=channels,
            out_channels=self.exp_channels,
            stride=stride,
            bn_moment=bn_moment,
            activation=True
        )

        self.sep_conv_block = SepConvBlock(
            kernel_size=kernel_size,
            channels=self.exp_channels,
            bn_moment=bn_moment
        )

        # no activation here
        self.conv_bn2 = ConvBNReluBlock(
            in_channels=self.exp_channels,
            out_channels=channels,
            stride=1,
            bn_moment=bn_moment,
            activation=False,
        )
        self.stride = stride
        if stride != 1:
            # if stride is not one, can not directly perform skip connection
            # so need to apply conv with same stride before skip connection
            # no activation here
            self.skip_conv_bn = ConvBNReluBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=stride,
                bn_moment=bn_moment,
                activation=False
            )

    def forward(self, batch):
        output = self.conv1(batch)
        output = F.relu(self.bn1(output))

        output = self.sep_conv_block(output)

        output = self.conv2(output)
        output = self.bn2(output)

        if self.stride != 1:
            batch = self.skip_conv_bn(batch)
        return batch + output


class MBConvBlock(nn.Module):
    def __init__(self, layers_cnt, stride, in_channels, out_channels, kernel_size, bn_moment, exp_factor):
        super().__init__()
        self.layers_cnt = layers_cnt
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.exp_factor = exp_factor

        mb_convs = []
        if layers_cnt == 1:
            mb_convs.append([
                MBConv()
            ])
        else:
            pass

        self.mb_convs = nn.ModuleList([
            MBConv(
                exp_factor=exp_factor,
                kernel_size=kernel_size,
                channels=channels,
                stride=stride,
                bn_moment=bn_moment
            ) for _ in range(layers_cnt)
        ])

    def forward(self, batch):
        return self.mb_convs(batch)

    def get_name(self):
        kernel_str = kernel_size_to_str(self.kernel_size)
        return f'{self.layers_cnt} x MBConv{self.exp_factor}(ks={kernel_str},str={self.stride},ch={self.channels})'


class EfficientNet(nn.Module):
    def __init__(self, kernel_sizes, strides, in_channels, out_channels, block_sizes, bn_moment):
        # n_blocks = len(block_sizes)
        # block_sizes = [1, 2, 2, 3, 3, 4, 1]
        # channels = [3, 32, 16, 24, 40, 80, 112, 192, 320]
        # kernel_sizes = [(3,3), (3,3), (3,3), (5,5), (3,3), (5,5), (5,5), (3,3)]
        # resolutions = [(224,224), (112,112), (112,112), (56,56), (28,28), (14,14), (14,14), (7,7), (7,7)]
        # strides = []
        # out_channels = [32, 16, 24]
        # in_channels = []

        super().__init__()
        first_conv = ConvBNReluBlock(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            stride=strides[0],
            activation=False,
            kernel_size=kernel_sizes[0]
        )
        layers_lst = [(first_conv.get_name(), first_conv)]

        for ks, st, ch, l_cnt in zip(kernel_sizes[1:], strides[1:], channels[1:], layers_cnt):
            block = None


if __name__ == '__main__':
    image_augmentation = None
    new_resolution = 56

    to_tensor_tr = T.ToTensor()
    color_and_sharpness_tr = PerformAtMostN(
        [T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01),
         MyRandomAdjustSharpness(0.5, 3, 10, prob=1)],
        prob=0.5,
        n=2
    )

    rotation_and_perspective_tr = PerformAtMostN(
        [T.RandomRotation((-20, 20)),
         T.RandomPerspective(distortion_scale=0.2)],
        prob=0.5,
        n=2,
    )

    transforms = nn.Sequential(
        T.Resize(new_resolution),
        T.RandomApply(nn.ModuleList([
                T.RandomResizedCrop(new_resolution, scale=(0.7, 1.0)),
                color_and_sharpness_tr,
                T.RandomErasing(value='random'),
                T.RandomHorizontalFlip(),
                rotation_and_perspective_tr
            ]),
            p=0.85
        )
    )
