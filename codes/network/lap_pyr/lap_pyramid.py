import torch
import torch.nn.functional


class LapPyramidConv(torch.nn.Module):
    def __init__(self, level=3, device='cuda'):
        super(LapPyramidConv, self).__init__()

        self.level = level
        kernel = self.create_gauss_kernel(device=device)
        self.kernel = dict()
        self.kernel[kernel.device] = kernel
        return

    @staticmethod
    def create_gauss_kernel(device='cuda', channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def get_gauss_kernel(self, device):
        kernel = self.kernel.get(device, None)
        if kernel is None:
            print('in {} not found kernel'.format(device))
            kernel = self.kernel[list(self.kernel.keys())[0]].detach().clone().to(device)
            self.kernel[device] = kernel
        return kernel

    @staticmethod
    def down_sample(x):
        return x[:, :, ::2, ::2]

    def up_sample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        kernel = self.get_gauss_kernel(x.device)
        return self.conv_gauss(x_up, 4 * kernel)

    @staticmethod
    def conv_gauss(img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        lap = []
        gauss = [current]
        kernel = self.get_gauss_kernel(img.device)
        for _ in range(self.level):
            filtered = self.conv_gauss(current, kernel)
            down = self.down_sample(filtered)
            up = self.up_sample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = torch.nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            lap.append(diff)
            current = down
            gauss.append(current)
        lap.append(current)
        return lap, gauss

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.up_sample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = torch.nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image
    