import torch
import torch.nn.functional as F
from codes.network.build import BUILD_NETWORK_REGISTRY
import logging

__all__ = [
    'LPTNBasic'
]


class LapPyramidBicubic(torch.nn.Module):
    """

    """
    def __init__(self, num_high=3):
        super(LapPyramidBicubic, self).__init__()

        self.interpolate_mode = 'bicubic'
        self.num_high = num_high

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for i in range(self.num_high):
            down = torch.nn.functional.interpolate(current, size=(current.shape[2] // 2, current.shape[3] // 2), mode=self.interpolate_mode, align_corners=True)
            up = torch.nn.functional.interpolate(down, size=(current.shape[2], current.shape[3]), mode=self.interpolate_mode, align_corners=True)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            image = F.interpolate(image, size=(level.shape[2], level.shape[3]), mode=self.interpolate_mode, align_corners=True) + level
        return image


class LapPyramidConv(torch.nn.Module):
    def __init__(self, num_high=3):
        super(LapPyramidConv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def down_sample(self, x):
        return x[:, :, ::2, ::2]

    def up_sample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.down_sample(filtered)
            up = self.up_sample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = torch.nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.up_sample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = torch.nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, in_features, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class TransLow(torch.nn.Module):
    def __init__(self, num_residual_blocks):
        super(TransLow, self).__init__()

        model = [torch.nn.Conv2d(3, 16, 3, padding=1),
                 torch.nn.InstanceNorm2d(16),
                 torch.nn.LeakyReLU(),
                 torch.nn.Conv2d(16, 64, 3, padding=1),
                 torch.nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [torch.nn.Conv2d(64, 16, 3, padding=1),
                  torch.nn.LeakyReLU(),
                  torch.nn.Conv2d(16, 3, 3, padding=1)]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        out = torch.tanh(out)
        return out


class TransHighLK3(torch.nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(TransHighLK3, self).__init__()

        self.num_high = num_high

        model = [torch.nn.Conv2d(9, 64, 3, padding=1),
                 torch.nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [torch.nn.Conv2d(64, 3, 3, padding=1)]

        self.model = torch.nn.Sequential(*model)

        for i in range(self.num_high):
            trans_mask_block = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(16, 3, 1))
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        mask = self.model(x)

        for i in range(self.num_high):
            mask = torch.nn.functional.interpolate(mask, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3]))
            result_high_freq = torch.mul(pyr_original[-2-i], mask) + pyr_original[-2-i]
            trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            result_high_freq = trans_mask_block(result_high_freq)
            setattr(self, 'result_high_freq_{}'.format(str(i)), result_high_freq)

        for i in reversed(range(self.num_high)):
            result_high_freq = getattr(self, 'result_high_freq_{}'.format(str(i)))
            pyr_result.append(result_high_freq)

        pyr_result.append(fake_low)

        return pyr_result


class TransHigh(torch.nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(TransHigh, self).__init__()

        self.num_high = num_high

        model = [torch.nn.Conv2d(9, 64, 3, padding=1),
                 torch.nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [torch.nn.Conv2d(64, 1, 3, padding=1)]

        self.model = torch.nn.Sequential(*model)

        for i in range(self.num_high):
            trans_mask_block = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, 1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(16, 1, 1))
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        mask = self.model(x)

        for i in range(self.num_high):
            mask = torch.nn.functional.interpolate(mask, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3]))
            trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            mask = trans_mask_block(mask)
            result_high_freq = torch.mul(pyr_original[-2-i], mask)
            setattr(self, 'result_high_freq_{}'.format(str(i)), result_high_freq)

        for i in reversed(range(self.num_high)):
            result_high_freq = getattr(self, 'result_high_freq_{}'.format(str(i)))
            pyr_result.append(result_high_freq)

        pyr_result.append(fake_low)

        return pyr_result


@BUILD_NETWORK_REGISTRY.register()
class LPTNBasic(torch.nn.Module):
    def __init__(self, cfg):
        super(LPTNBasic, self).__init__()
        nrb_low = 5
        nrb_high = 3
        num_high = 3
        if cfg.MODEL.NETWORK.get('LPTN', None) is not None:
            lptn_cfg = cfg.MODEL.NETWORK.get('LPTN')
            nrb_low = lptn_cfg.get('NRB_LOW', 5)
            nrb_high = lptn_cfg.get('NRB_HIGH', 3)
            num_high = lptn_cfg.get('NUM_HIGH', 3)

        self.lap_pyramid = LapPyramidConv(num_high)
        trans_low = TransLow(nrb_low)
        trans_high = TransHighLK3(nrb_high, num_high=num_high)
        self.trans_low = trans_low.cuda()
        self.trans_high = trans_high.cuda()
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create network {}'.format(self.__class__))
        return

    def forward(self, real_a_full):

        pyr_a = self.lap_pyramid.pyramid_decom(img=real_a_full)
        fake_b_low = self.trans_low(pyr_a[-1])
        real_a_up = torch.nn.functional.interpolate(pyr_a[-1], size=(pyr_a[-2].shape[2], pyr_a[-2].shape[3]))
        fake_b_up = torch.nn.functional.interpolate(fake_b_low, size=(pyr_a[-2].shape[2], pyr_a[-2].shape[3]))
        high_with_low = torch.cat([pyr_a[-2], real_a_up, fake_b_up], 1)
        pyr_a_trans = self.trans_high(high_with_low, pyr_a, fake_b_low)
        fake_b_full = self.lap_pyramid.pyramid_recons(pyr_a_trans)

        return fake_b_full
