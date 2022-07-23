import torch


class TV3D(torch.nn.Module):
    def __init__(self, dim=33, device='cuda'):
        super(TV3D, self).__init__()

        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float).to(device)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float).to(device)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float).to(device)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU().to(device)
        self.to(device)
        return

    def forward(self, lut: torch.nn.Module):
        module = lut
        if isinstance(lut, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            module = lut.module
        dif_r = module.lut[:, :, :, :-1] - module.lut[:, :, :, 1:]
        dif_g = module.lut[:, :, :-1, :] - module.lut[:, :, 1:, :]
        dif_b = module.lut[:, :-1, :, :] - module.lut[:, 1:, :, :]
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn

