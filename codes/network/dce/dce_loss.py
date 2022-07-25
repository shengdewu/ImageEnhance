import torch
import torch.nn.functional as tnf

__all__ = [
    'SpatialConsistencyLoss',
    'ExposureControlLoss',
    'ColorConstancyLoss',
    'IlluminationSmoothnessLoss'
]


class SpatialConsistencyLoss(torch.nn.Module):
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight_left = torch.nn.Parameter(kernel_left, requires_grad=False)
        self.weight_right = torch.nn.Parameter(kernel_right, requires_grad=False)
        self.weight_up = torch.nn.Parameter(kernel_up, requires_grad=False)
        self.weight_down = torch.nn.Parameter(kernel_down, requires_grad=False)
        self.pool = torch.nn.AvgPool2d(kernel_size=4)
        return

    def forward(self, x, enhance_x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        enhance_x_mean = torch.mean(enhance_x, dim=1, keepdim=True)

        x_local_region = self.pool(x_mean)
        enhance_x_local_region = self.pool(enhance_x_mean)

        x_left = tnf.conv2d(x_local_region, self.weight_left, padding=1)
        x_right = tnf.conv2d(x_local_region, self.weight_right, padding=1)
        x_up = tnf.conv2d(x_local_region, self.weight_up, padding=1)
        x_down = tnf.conv2d(x_local_region, self.weight_down, padding=1)

        enhance_x_left = tnf.conv2d(enhance_x_local_region, self.weight_left, padding=1)
        enhance_x_right = tnf.conv2d(enhance_x_local_region, self.weight_right, padding=1)
        enhance_x_up = tnf.conv2d(enhance_x_local_region, self.weight_up, padding=1)
        enhance_x_down = tnf.conv2d(enhance_x_local_region, self.weight_down, padding=1)

        d_left = torch.pow(x_left - enhance_x_left, 2)
        d_right = torch.pow(x_right - enhance_x_right, 2)
        d_up = torch.pow(x_up - enhance_x_up, 2)
        d_down = torch.pow(x_down - enhance_x_down, 2)
        return d_left + d_right + d_up + d_down


class ExposureControlLoss(torch.nn.Module):
    def __init__(self, local_region_size=16, e=0.6):
        super(ExposureControlLoss, self).__init__()
        self.well_level = torch.nn.Parameter(torch.FloatTensor([e]), requires_grad=False)
        self.pool = torch.nn.AvgPool2d(kernel_size=local_region_size)
        return

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_local_region = self.pool(x_mean)
        return torch.mean(torch.pow(x_local_region - self.well_level, 2))


class ColorConstancyLoss(torch.nn.Module):
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()
        return

    def forward(self, x):
        x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        r_mean, g_mean, b_mean = torch.split(x_mean, 1, dim=1)
        d_rg = torch.pow(r_mean - g_mean, 2)
        d_rb = torch.pow(r_mean - b_mean, 2)
        d_gb = torch.pow(g_mean - b_mean, 2)
        return torch.pow(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5)


class IlluminationSmoothnessLoss(torch.nn.Module):
    def __init__(self, tv_weight):
        super(IlluminationSmoothnessLoss, self).__init__()
        self.tv_weight = tv_weight
        return

    def forward(self, x):
        b, c, h, w = x.size()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w-1], 2).sum()
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h-1, :], 2).sum()
        count_h = (h - 1) * w
        count_w = (w - 1) * h
        return self.tv_weight * 2 * (h_tv / count_h + w_tv / count_w) / b


