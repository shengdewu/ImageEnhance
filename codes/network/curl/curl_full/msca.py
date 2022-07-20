import torch
from engine.model.depth_wise import DepthWiseSeparableConv2d


class Fatten(torch.nn.Module):
    def __init__(self):
        super(Fatten, self).__init__()
        return

    def forward(self, x):
        return x.view(x.size()[0], -1)


class MidLevel(torch.nn.Module):
    def __init__(self, in_channels, out_channels=64, padding=2, dilation=2):
        super(MidLevel, self).__init__()
        self.relu = torch.nn.LeakyReLU()
        self.conv1 = DepthWiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.conv2 = DepthWiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.conv3 = DepthWiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.conv4 = DepthWiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        return

    def forward(self, x_in):
        x = self.relu(self.conv1(x_in))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.conv4(x)


class GlobalLevel(torch.nn.Module):
    def __init__(self, in_channels=16, out_channels=64):
        super(GlobalLevel, self).__init__()
        self.level = torch.nn.Sequential(
            DepthWiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            DepthWiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            DepthWiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            DepthWiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            DepthWiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            Fatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=out_channels, out_features=out_channels)
        )
        return

    def forward(self, x):
        return self.level(x)


class MSCA(torch.nn.Module):
    '''
    multi-scale contextual awareness(MSCA) connection
    '''
    def __init__(self, in_channels=16, mid_channels=64):
        super(MSCA, self).__init__()
        self.mid_level2 = MidLevel(in_channels=in_channels, out_channels=mid_channels, padding=2, dilation=2)
        self.mid_level4 = MidLevel(in_channels=in_channels, out_channels=mid_channels, padding=4, dilation=4)
        self.global_level = GlobalLevel(in_channels=in_channels, out_channels=mid_channels)
        self.conv_fuse = DepthWiseSeparableConv2d(in_channels=(mid_channels*3+in_channels), out_channels=in_channels, kernel_size=1, padding=0, stride=1)
        return

    def forward(self, x):
        mid_feature2 = self.mid_level2(x)
        mid_feature4 = self.mid_level4(x)
        global_feature = self.global_level(x)
        global_feature = global_feature.unsqueeze(2)
        global_feature = global_feature.unsqueeze(3)
        global_feature = global_feature.repeat(1, 1, mid_feature2.shape[2], mid_feature2.shape[3])
        fuse = torch.cat([x, mid_feature2, mid_feature4, global_feature], dim=1)
        return self.conv_fuse(fuse)