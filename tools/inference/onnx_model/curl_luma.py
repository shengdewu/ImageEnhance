import torch
from engine.model.depth_wise import DepthWiseSeparableConv2d
import engine.model.attention.conv_block_attention as cbam


class CurlLumaNet(torch.nn.Module):
    def __init__(self, ratio, pre_knot_points, kernel_number=32, knot_points=48):
        super(CurlLumaNet, self).__init__()

        self.channel_attention = cbam.ChannelAttentionModule(pre_knot_points, ratio)
        self.spatial_attention = cbam.SpatialAttentionModule()

        self.stem_1 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(1, kernel_number),
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
            DepthWiseSeparableConv2d(kernel_number*2, pre_knot_points),
            torch.nn.LeakyReLU()
        )

        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(pre_knot_points, knot_points)
        self.dropout = torch.nn.Dropout(0.5)

        return

    def forward(self, gray):
        s1 = self.stem_1(gray)
        s2 = self.stem_2(s1)
        s3 = self.stem_3(s2)
        s4 = self.stem_4(s3)
        s5 = self.stem_5(torch.cat([s3, s4], dim=1))
        s6 = self.stem_6(torch.cat([s2, s5], dim=1))
        r = self.stem_7(torch.cat([s1, s6], dim=1))

        ca = self.channel_attention(r)
        f = torch.mul(ca, r)
        sa = self.spatial_attention(f)
        r = torch.mul(sa, f)

        r = self.avg(r)
        r = r.view(r.size()[0], -1)
        r = self.dropout(r)
        r = self.fc(r)
        return torch.exp(r)
