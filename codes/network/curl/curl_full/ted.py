import torch
import torch.nn.functional as tnf
from .msca import MSCA
from engine.model.depth_wise import DepthWiseSeparableConv2d

__all__ = ['TransformedEncoderDecoder']


class TEDConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(TEDConv2d, self).__init__()
        self.conv1 = DepthWiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = DepthWiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = torch.nn.LeakyReLU()
        self.pad = torch.nn.ReflectionPad2d(1)
        return

    def forward(self, x):
        x = self.relu(self.conv1(self.pad(x)))
        return self.relu(self.conv2(self.pad(x)))


class TransformedEncoderDecoder(torch.nn.Module):
    def __init__(self):
        super(TransformedEncoderDecoder, self).__init__()

        self.down1 = TEDConv2d(in_channels=3, out_channels=16)
        self.down2 = TEDConv2d(in_channels=16, out_channels=32)
        self.down3 = TEDConv2d(in_channels=32, out_channels=64)
        self.down4 = TEDConv2d(in_channels=64, out_channels=128)
        self.down5 = TEDConv2d(in_channels=128, out_channels=128)

        # self.pool = torch.nn.MaxPool2d(kernel_size=2, padding=0)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, padding=0)
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_conv1x1_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        self.up_conv1x1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.up_conv1x1_3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        self.up_conv1x1_4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1)

        self.up4 = TEDConv2d(in_channels=128, out_channels=64)
        self.up3 = TEDConv2d(in_channels=64, out_channels=32)
        self.up2 = TEDConv2d(in_channels=32, out_channels=16)
        self.up1 = TEDConv2d(in_channels=32, out_channels=3)

        self.msca_model = MSCA(in_channels=16, mid_channels=64)

        self.final_conv = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.final_pad = torch.nn.ReflectionPad2d(1)
        return

    def adjust_wh(self, x, y):
        if x.shape[2] != y.shape[2] and x.shape[3] != y.shape[3]:
            x = tnf.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != y.shape[2]:
            x = tnf.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != y.shape[3]:
            x = tnf.pad(x, (1, 0, 0, 0))
        return x

    def forward(self, x):
        x_copy = x.clone()
        conv1 = self.down1(x)
        x = self.pool(conv1)

        conv2 = self.down2(x)
        x = self.pool(conv2)

        conv3 = self.down3(x)
        x = self.pool(conv3)

        conv4 = self.down4(x)
        x = self.pool(conv4)

        x = self.down5(x)

        x = self.up_conv1x1_1(self.upsample(x))

        x = self.adjust_wh(x, conv4)
        del conv4

        x = self.up4(x)
        x = self.up_conv1x1_2(self.upsample(x))
        x = self.adjust_wh(x, conv3)
        del conv3

        x = self.up3(x)
        x = self.up_conv1x1_3(self.upsample(x))
        x = self.adjust_wh(x, conv2)
        del conv2

        x = self.up2(x)
        x = self.up_conv1x1_4(self.upsample(x))

        conv1_fuse = self.msca_model(conv1)
        x = self.adjust_wh(x, conv1)
        x = torch.cat([x, conv1_fuse], dim=1)
        del conv1

        x = self.up1(x)

        out = x + x_copy
        return self.final_conv(self.final_pad(out))
