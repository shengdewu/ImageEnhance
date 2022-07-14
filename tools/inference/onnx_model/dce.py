import torch
from engine.model.depth_wise import DepthWiseSeparableConv2d


class DceNet(torch.nn.Module):
    def __init__(self, kernel_number=32, device='cpu'):
        super(DceNet, self).__init__()

        self.stem_1 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(3, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_2 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_3 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_4 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_5 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_6 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_7 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, 3),
            torch.nn.Tanh()
        )
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
        return cure
