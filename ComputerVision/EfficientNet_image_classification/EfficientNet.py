import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from Augmentation import PerformAtMostN, MyRandomAdjustSharpness
from collections import OrderedDict


def swish(t):
    return F.sigmoid(t) * t


def kernel_size_to_str(kernel_size):
    if len(kernel_size) == 1:
        return f'{kernel_size[0]}x{kernel_size[0]}'
    return f'{kernel_size[0]}x{kernel_size[1]}'


class SeparableConv(nn.Module):
    """Separable convolution, lightweight replacement for ordinary convolution."""
    def __init__(self, kernel_size, stride, channels, bn_moment, padding):
        # perform depthwise conv, pointwise conv, batch norm and swish
        super(SeparableConv, self).__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=padding
        )
        self.pw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1,),
        )
        self.bn = nn.BatchNorm2d(num_features=channels, momentum=bn_moment)

    def forward(self, batch):
        output = self.dw_conv(batch)
        output = self.pw_conv(output)
        output = self.bn(output)
        return swish(output)


class ConvBNSwish(nn.Module):
    """Conv, BatchNorm [and Swish] grouped together for simplicity."""
    def __init__(self, in_channels, out_channels, stride, bn_moment, activation=True, kernel_size=(1,), padding=0):
        super(ConvBNSwish, self).__init__()
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
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels, bn_moment)

    def forward(self, batch):
        output = self.conv(batch)
        output = self.bn(output)
        if self.activation:
            return swish(output)
        return output

    def get_name(self):
        kernel_str = kernel_size_to_str(self.kernel_size)
        result = f'Conv({kernel_str}),in_ch={self.in_channels},out_ch={self.out_channels},str={self.stride},BatchNorm'
        if self.activation:
            result += ',swish'
        return result


class SqueezeAndExcitation(nn.Module):
    """Squeeze and Excitation module, squeezes spatial dimension using avg_pooling and then computes weights for each channel."""
    def __init__(self, channels, reduction_factor):
        super(SqueezeAndExcitation, self).__init__()
        self.fc1 = nn.Linear(in_features=channels, out_features=channels // reduction_factor)
        self.fc2 = nn.Linear(in_features=channels // reduction_factor, out_features=channels)

    def forward(self, batch):
        # squeeze part
        # global average pooling
        # output.shape = [batch, channels, H, W]
        output = torch.mean(batch, (2, 3), False)
        # output.shape = [batch, channels]
        # excitation part
        output = self.fc1(output)
        output = swish(output)
        output = self.fc2(output)
        return F.sigmoid(output)


class MBConv(nn.Module):
    """Even though in paper the block is denoted as MBConv, in the text they write that they also used squeeze-and-excitation subblock."""
    def __init__(self, exp_factor, kernel_size, in_channels, out_channels, stride, bn_moment, padding, stochastic_depth):
        super(MBConv, self).__init__()
        self.stochastic_depth_prob = stochastic_depth
        self.stride = stride

        exp_channels = exp_factor * out_channels
        self.conv_bn_swish1 = ConvBNSwish(
            in_channels=in_channels,
            out_channels=exp_channels,
            stride=1,
            bn_moment=bn_moment,
            activation=True,
            kernel_size=(1,),
            padding=0,
        )
        # here will be performed resolution reduction, if stride != 1
        self.sep_conv_block = SeparableConv(
            kernel_size=kernel_size,
            stride=stride,
            channels=self.exp_channels,
            bn_moment=bn_moment,
            padding=padding,
        )

        # no activation here
        self.conv_bn2 = ConvBNSwish(
            in_channels=self.exp_channels,
            out_channels=out_channels,
            stride=1,
            bn_moment=bn_moment,
            activation=False,
            kernel_size=(1,),
            padding=0,
        )
        self.se = None
        if out_channels == in_channels:
            self.se = SqueezeAndExcitation(out_channels, exp_factor)

    def forward(self, batch):
        output = self.conv1(batch)
        output = self.bn1(output)
        output = swish(output)

        output = self.sep_conv_block(output)

        output = self.conv2(output)
        output = self.bn2(output)

        # can perform skip connection, squeeze-excitation and stochastic depth
        # only if stride == 1 and number of input and output channels is equal (then self.se is not None)
        if self.stride == 1 and self.se is not None:
            weights = self.se(output)
            # weights.shape = [batch, channels]
            # output.shape = [batch, channels, H, W]
            output = output * weights.view((output.shape[0], output.shape[1], 1, 1))

            # stochastic depth
            if self.training:
                survived = torch.rand(output.shape[0]) < self.stochastic_depth_prob
                survived = survived.type(torch.float).view(survived.shape[0], 1, 1, 1)
                output = (survived / self.stochastic_depth_prob) * survived

            output = output + batch

        return output


class MBConvBlock(nn.Module):
    """Just a list of MBConvs, with careful handling of first MBConv, should take care of spatial resolution and channels."""
    def __init__(self, layers_cnt, stride, in_channels, out_channels, kernel_size, bn_moment, exp_factor, stochastic_depths, padding):
        super(MBConvBlock, self).__init__()
        self.layers_cnt = layers_cnt
        self.stride = stride
        self.channels = out_channels
        self.kernel_size = kernel_size
        self.exp_factor = exp_factor

        # handle first MBConv, it will reduce space resolution if needed (strided convolution)
        # also it will expand channels for the whole block
        self.mb_convs = [
            MBConv(
                exp_factor=exp_factor,
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bn_moment=bn_moment,
                padding=padding,
                stochastic_depth=stochastic_depths[0],
            )
        ]
        # the rest of the block will work with fixed space resolution and fixed number of channels
        for i in range(1, layers_cnt):
            self.mb_convs.append(
                MBConv(
                    exp_factor=exp_factor,
                    kernel_size=kernel_size,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=(1,),
                    bn_moment=bn_moment,
                    padding=(0,),
                    stochastic_depth=stochastic_depths[i],
                )
            )
        self.mb_convs = nn.ModuleList(self.mb_convs)

    def forward(self, batch):
        return self.mb_convs(batch)

    def get_name(self):
        kernel_str = kernel_size_to_str(self.kernel_size)
        return f'{self.layers_cnt} x MBConv{self.exp_factor}(ks={kernel_str},str={self.stride},ch={self.channels})'


class EfficientNet(nn.Module):
    """Custom implementation of EfficientNetB0 according to the paper."""
    def __init__(self, kernel_sizes, strides, channels, block_sizes, exp_factors, bn_moment, paddings, do, n_targets):
        super(EfficientNet, self).__init__()
        # output_resolutions = [(112,112), (112,112), (56,56), (28,28), (14,14), (14,14), (7,7), (7,7)]
        # n_blocks = len(block_sizes) (=7)

        # block_sizes = [1, 2, 2, 3, 3, 4, 1],
        #   len(block_sizes) == n_blocks

        # channels_lst = [32, 16, 24, 40, 80, 112, 192, 320],
        #   len(channels_lst) == n_blocks + 1

        # kernel_sizes = [(3,3), (3,3), (3,3), (5,5), (3,3), (5,5), (5,5), (3,3)]
        #   len(kernel_sizes) == n_blocks + 1

        # paddings = [(0,1), 0, (0,1), (1,2), (0,1), 0, (1,2), 0]
        #   len(paddings) == n_blocks + 1

        # strides = [2, 1, 2, 2, 2, 1, 2, 1]
        #   len(strides) == n_blocks + 1

        # exp_factors = [1, 6, 6, 6, 6, 6, 6]

        first_conv = ConvBNSwish(
            in_channels=channels[0],
            out_channels=channels[1],
            stride=strides[0],
            activation=False,
            kernel_size=kernel_sizes[0],
            bn_moment=bn_moment,
        )
        blocks_lst = [(first_conv.get_name(), first_conv)]

        stochastic_depth = torch.linspace(0, 0.5, sum(block_sizes) + 1)
        stochastic_depth_idx = 1

        for ks, st, ch_in, ch_out, l_cnt, e_fact, pad in zip(kernel_sizes[1:], strides[1:], channels[1:-1], channels[2:], block_sizes, exp_factors, paddings):
            block = MBConvBlock(
                layers_cnt=l_cnt,
                stride=st,
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=ks,
                bn_moment=bn_moment,
                exp_factor=e_fact,
                stochastic_depths=stochastic_depth[stochastic_depth_idx: stochastic_depth_idx + l_cnt],
                padding=pad
            )
            blocks_lst.append([block.get_name(), block])
            stochastic_depth_idx += l_cnt

        self.last_conv =
        self.drop_out = nn.Dropout(do)
        self.Linear =

    def forward(self, batch, training=True):
        for i, do in zip():
            pass

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
