# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
# from Augmentation import PerformAtMostN, MyRandomAdjustSharpness
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

import os
import math
from icecream import ic
from typing import Optional
from collections import OrderedDict, namedtuple


def swish(t):
    return torch.sigmoid(t) * t


def kernel_size_to_str(kernel_size):
    return f'{kernel_size}x{kernel_size}'


class SeparableConv(nn.Module):
    """Separable convolution, lightweight replacement for ordinary convolution."""
    def __init__(self, kernel_size, stride, channels, bn_moment, padding):
        # perform depthwise conv, pointwise conv, batch norm and swish
        super(SeparableConv, self).__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=channels,
            padding=padding
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
        output = self.bn(output)
        return swish(output)


class ConvBNSwish(nn.Module):
    """Conv, BatchNorm [and Swish] grouped together for simplicity."""
    def __init__(self, in_channels, out_channels, stride, bn_moment, activation=True, kernel_size=1, padding=0):
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
        return torch.sigmoid(output)


class MBConv(nn.Module):
    """Even though in paper the block is denoted as MBConv, in the text they write that they also used squeeze-and-excitation subblock."""
    def __init__(self, exp_factor, kernel_size, in_channels, out_channels, stride, bn_moment, padding, stochastic_depth, layer_idx):
        super(MBConv, self).__init__()
        self.stochastic_depth_prob = stochastic_depth
        self.stride = stride
        self.layer_idx = layer_idx

        exp_channels = exp_factor * out_channels
        self.conv_bn_swish1 = ConvBNSwish(
            in_channels=in_channels,
            out_channels=exp_channels,
            stride=1,
            bn_moment=bn_moment,
            activation=True,
            kernel_size=1,
            padding=0,
        )
        # here will be performed resolution reduction, if stride != 1
        self.sep_conv = SeparableConv(
            kernel_size=kernel_size,
            stride=stride,
            channels=exp_channels,
            bn_moment=bn_moment,
            padding=padding,
        )

        # no activation here
        self.conv_bn2 = ConvBNSwish(
            in_channels=exp_channels,
            out_channels=out_channels,
            stride=1,
            bn_moment=bn_moment,
            activation=False,
            kernel_size=1,
            padding=0,
        )
        self.se = None
        if out_channels == in_channels:
            self.se = SqueezeAndExcitation(out_channels, exp_factor)

    def forward(self, batch):
        output = self.conv_bn_swish1(batch)
        output = self.sep_conv(output)
        output = self.conv_bn2(output)

        # can perform skip connection, squeeze-excitation and stochastic depth
        # only if stride == 1 and number of input and output channels is equal (then self.se is not None)
        if self.stride == 1 and self.se is not None:
            print(f'{self.layer_idx} skip connection')
            weights = self.se(output)
            # weights.shape = [batch, channels]
            # output.shape = [batch, channels, H, W]
            output = output * weights.view((output.shape[0], output.shape[1], 1, 1))

            # stochastic depth
            if self.training:
                print(f'{self.layer_idx} stoch depth')
                survived = torch.rand(output.shape[0]) < self.stochastic_depth_prob
                survived = survived.type(torch.float).view(survived.shape[0], 1, 1, 1)
                output = (survived / self.stochastic_depth_prob) * survived

            output = output + batch

        return output


class MBConvBlock(nn.Module):
    """Just a list of MBConvs, with careful handling of first MBConv, should take care of spatial resolution and channels."""
    def __init__(self, layers_cnt, stride, in_channels, out_channels, kernel_size, bn_moment, exp_factor, stochastic_depths, padding, layers_offset):
        super(MBConvBlock, self).__init__()
        self.layers_cnt = layers_cnt
        self.stride = stride
        self.channels = out_channels
        self.kernel_size = kernel_size
        self.exp_factor = exp_factor
        self.padding = padding

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
                layer_idx=layers_offset,
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
                    stride=1,
                    bn_moment=bn_moment,
                    padding='same',
                    stochastic_depth=stochastic_depths[i],
                    layer_idx=layers_offset + i,
                )
            )
        self.mb_convs = nn.ModuleList(self.mb_convs)

    def forward(self, batch):
        output = batch
        for m in self.mb_convs:
            output = m(output)
        return output

    def get_name(self):
        kernel_str = kernel_size_to_str(self.kernel_size)
        return f'{self.layers_cnt} x MBConv{self.exp_factor}(ks={kernel_str},str={self.stride},ch={self.channels},pad={self.padding})'


class EfficientNet(nn.Module):
    """Custom implementation of EfficientNetB0 according to the paper."""
    def __init__(self, arch_params, bn_moment, do, n_targets, head_features, stochastic_depth):
        super(EfficientNet, self).__init__()

        first_conv = ConvBNSwish(
            in_channels=arch_params[0].channels_in,
            out_channels=arch_params[0].channels_out,
            stride=arch_params[0].stride,
            activation=False,
            kernel_size=arch_params[0].kernel_size,
            bn_moment=bn_moment,
            padding=arch_params[0].padding
        )
        blocks_lst = [(first_conv.get_name(), first_conv)]

        layers_offset = 1
        for p in arch_params[1:]:
            block = MBConvBlock(
                layers_cnt=p.block_size,
                stride=p.stride,
                in_channels=p.channels_in,
                out_channels=p.channels_out,
                kernel_size=p.kernel_size,
                bn_moment=bn_moment,
                exp_factor=p.exp_factor,
                stochastic_depths=stochastic_depth,
                padding=p.padding,
                layers_offset=layers_offset,
            )
            blocks_lst.append([block.get_name(), block])
            layers_offset += p.block_size

        self.block_dct = OrderedDict(blocks_lst)
        self.conv_head = ConvBNSwish(
            in_channels=arch_params[-1].channels_out,
            out_channels=head_features,
            stride=1,
            bn_moment=bn_moment,
            activation=False,
            kernel_size=1,
            padding=0,
        )
        self.drop_out = nn.Dropout(do)
        self.linear = nn.Linear(in_features=head_features, out_features=n_targets)

    def forward(self, batch):
        output = batch
        print(output.shape)
        for name, block in self.block_dct.items():
            output = block(output)
            # print(f'{name:50} {output.shape}')
        output = self.conv_head(output)
        output = torch.mean(output, dim=(2, 3), keepdim=False)
        output = self.drop_out(output)
        print(output.shape)
        return self.linear(output)


class EfficientNetPL(pl.LightningDataModule):
    def __init__(self, eff_net, lr, weight_decay, momentum):
        super(EfficientNetPL, self).__init__()
        self.eff_net = eff_net
        self.init_lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.optim_eps = 0.001

    def forward(self, batch):
        return self.eff_net(batch)

    def _perform_step(self, batch):
        images, targets = batch['images'], batch['targets']
        outputs = self(images)

        loss = F.cross_entropy(outputs, targets)

        predictions = outputs.argmax(dim=1)
        correct = torch.sum(predictions == targets).item()
        acc = correct / targets.shape[0]

        return loss, acc, targets[0]

    def training_step(self, batch, batch_idx):
        loss, acc, batch_size = self._perform_step(batch)
        return dict(loss=loss, acc=acc, batch_size=batch_size)

    def validation_step(self, batch, batch_idx):
        loss, acc, batch_size = self._perform_step(batch)
        return dict(loss=loss, acc=acc, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        loss, acc, batch_size = self._perform_step(batch)
        return dict(loss=loss, acc=acc, batch_size=batch_size)

    def configure_optimizers(self):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass


class Cifar10PL(pl.LightningDataModule):
    def __init__(self, path, train_perc, batch_size, seed, train_tr, test_tr):
        super(Cifar10PL, self).__init__()
        self.path = path
        self.bs = batch_size
        self.num_workers = (os.cpu_count() - 2) // torch.cuda.device_count()
        self.train_perc = train_perc
        self.seed = seed
        self.train_tr = train_tr
        self.test_tr = test_tr
        # self.train, self.val, self.test = None, None, None

    def prepare_data(self):
        """things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode, e.g. downloading data only once"""
        if not os.path.isdir(os.path.join(self.path, 'train')):
            CIFAR10(os.path.join(os.getcwd(), 'train'), download=True)
        if not os.path.isdir(os.path.join(self.path, 'test')):
            CIFAR10(os.path.join(os.getcwd(), 'train'), train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """(things to do on every accelerator in distributed mode) called on every process in DDP."""
        train_and_val = CIFAR10(os.path.join(os.getcwd(), 'train'))

        train_indices, val_indices = train_test_split(
            range(len(train_and_val)),
            train_size=self.train_perc,
            random_state=self.seed
        )

        train_data = CIFAR10(os.path.join(os.getcwd(), 'train'), transform=self.train_tr)
        self.train = Subset(train_data, train_indices)

        val_data = CIFAR10(os.path.join(os.getcwd(), 'train'), transform=self.test_tr)
        self.val = Subset(val_data, val_indices)

        self.test = CIFAR10(os.path.join(os.getcwd(), 'test'), transform=self.test_tr)

    def get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.bs,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=self.num_workers
        )

    def train_dataloader(self):
        return self.get_dataloader(self.train)

    def val_dataloader(self):
        return self.get_dataloader(self.val)

    def test_dataloader(self):
        return self.get_dataloader(self.test)


def collate_fn(lst):
    images, targets = [], []
    for i, t in lst:
        images.append(i)
        targets.append(t)
    return dict(images=torch.stack(images), targets=torch.tensor(targets))


class TrainTransform:
    def __init__(self, trainsforms):
        self.trans = trainsforms

    def __call__(self, image):
        output = image
        for t in self.trans:
            output = t(output)
        return output


# %%
if __name__ == '__main__':
    # %%

    image_augmentation = None
    new_resolution = 224

    train_transform = TrainTransform([
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Resize(new_resolution),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Resize(new_resolution)
    ])

    dataset = Cifar10PL(
        path=os.getcwd(),
        train_perc=0.7,
        batch_size=20,
        seed=0xDEAD,
        train_tr=train_transform,
        test_tr=test_transform
    )
    dataset.setup()

    train_loader = dataset.train_dataloader()
    for i, b in enumerate(train_loader):
        if i == 5:
            break
        ic(b)


    # %%
    # named tuple represents parameters for each block of Efficient net
    Block = namedtuple('Block', 'out_res block_size channels_in channels_out kernel_size stride exp_factor padding')
    # out_res is the output resolution it is for debugging
    # first dict is for the first convolution and not for the block

    params = [
        Block(out_res=112, block_size=0, channels_in=3,   channels_out=32,  kernel_size=3, stride=2, exp_factor=0, padding=1),
        Block(out_res=112, block_size=1, channels_in=32,  channels_out=16,  kernel_size=3, stride=1, exp_factor=1, padding='same'),
        Block(out_res=56,  block_size=2, channels_in=16,  channels_out=24,  kernel_size=3, stride=1, exp_factor=6, padding='same'),
        Block(out_res=28,  block_size=2, channels_in=24,  channels_out=40,  kernel_size=5, stride=2, exp_factor=6, padding=2),
        Block(out_res=14,  block_size=3, channels_in=40,  channels_out=80,  kernel_size=3, stride=2, exp_factor=6, padding=1),
        Block(out_res=14,  block_size=3, channels_in=80,  channels_out=112, kernel_size=5, stride=2, exp_factor=6, padding=2),
        Block(out_res=7,   block_size=4, channels_in=112, channels_out=192, kernel_size=5, stride=2, exp_factor=6, padding=2),
        Block(out_res=7,   block_size=1, channels_in=192, channels_out=320, kernel_size=3, stride=1, exp_factor=6, padding='same'),
    ]

    model = EfficientNet(
        params,
        bn_moment=0.99,
        do=0.2,
        n_targets=10,
        head_features=1280
    )

    resize_tr = T.Resize(new_resolution)
    # model(resize_tr(xs))


    # color_and_sharpness_tr = PerformAtMostN(
    #     [T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01),
    #      MyRandomAdjustSharpness(0.5, 3, 10, prob=1)],
    #     prob=0.5,
    #     n=2
    # )
    #
    # rotation_and_perspective_tr = PerformAtMostN(
    #     [T.RandomRotation((-20, 20)),
    #      T.RandomPerspective(distortion_scale=0.2)],
    #     prob=0.5,
    #     n=2,
    # )
    #
    # transforms = nn.Sequential(
    #     T.Resize(new_resolution),
    #     T.RandomApply(nn.ModuleList([
    #             T.RandomResizedCrop(new_resolution, scale=(0.7, 1.0)),
    #             color_and_sharpness_tr,
    #             T.RandomErasing(value='random'),
    #             T.RandomHorizontalFlip(),
    #             rotation_and_perspective_tr
    #         ]),
    #         p=0.85
    #     )
    # )
