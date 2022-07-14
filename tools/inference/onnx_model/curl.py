import torch
from engine.model.depth_wise import DepthWiseSeparableConv2d


class CurlNet(torch.nn.Module):
    def __init__(self, kernel_number=32, knot_points=48):
        super(CurlNet, self).__init__()

        kernel_number = kernel_number #32
        knot_points = knot_points # 48
        assert knot_points % 3 == 0, 'the {} must be divisible by 3'.format(knot_points)

        self.stem_1 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(3, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_2 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_3 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_4 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_5 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_6 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_7 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(kernel_number, knot_points)
        self.dropout = torch.nn.Dropout(0.5)

        return

    def forward(self, x):
        """
        x [n, c, h, w]
        """
        s1 = self.stem_1(x)
        s2 = self.stem_2(s1)
        s3 = self.stem_3(s2)
        s4 = self.stem_4(s3)
        s5 = self.stem_5(torch.cat([s3, s4], dim=1))
        s6 = self.stem_6(torch.cat([s2, s5], dim=1))
        cure = self.stem_7(torch.cat([s1, s6], dim=1))
        cure = self.avg(cure)
        cure = cure.view(cure.size()[0], -1)
        cure = self.dropout(cure)
        cure = self.fc(cure)

        batch_size, channels = cure.shape
        per_channels = (channels / 3).to(torch.int)
        r = torch.exp(cure[:, 0:per_channels])
        g = torch.exp(cure[:, per_channels: per_channels * 2])
        b = torch.exp(cure[:, per_channels * 2:per_channels * 3])
        return r, g, b
