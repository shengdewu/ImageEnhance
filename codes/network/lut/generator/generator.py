import torch
import numpy as np
from engine.checkpoint.functional import get_model_state_dict, load_model_state_dict

from trilinear.TrilinearInterpolationModel import TrilinearInterpolationModel


def generate_identity_lut(dim):
    lut3d = np.zeros((3, dim, dim, dim), dtype=np.float32)
    step = 1.0 / float(dim - 1)
    for b in range(dim):
        for g in range(dim):
            for r in range(dim):
                lut3d[0, b, g, r] = step * r
                lut3d[1, b, g, r] = step * g
                lut3d[2, b, g, r] = step * b
    return lut3d


class Generator3DLUT(torch.nn.Module):
    def __init__(self, dim=33, is_zero=False, device='cuda'):
        super(Generator3DLUT, self).__init__()

        if is_zero:
            parameter = torch.zeros(3, dim, dim, dim, dtype=torch.float)
        else:
            parameter = torch.from_numpy(generate_identity_lut(dim)).requires_grad_(True)

        self._lut = torch.nn.Parameter(parameter)
        self.trilinear_interpolation = TrilinearInterpolationModel()
        self.to(device)
        return

    @property
    def lut(self):
        return self._lut

    def forward(self, x):
        _, output = self.trilinear_interpolation(self._lut, x)
        return output


class Generator3DLUTSupplement(torch.nn.Module):
    def __init__(self, dim=33, nums=2, is_zero=True, device='cuda'):
        super(Generator3DLUTSupplement, self).__init__()
        self.generator_3d_lut = dict()  # index, lutmodel
        for i in range(nums):
            self.generator_3d_lut[i] = Generator3DLUT(dim, is_zero, device)
        return

    def parameters(self):
        parameters = list()
        for i, lut in self.generator_3d_lut.items():
            parameters.append(lut.parameters())
        return parameters

    def state_dict(self, offset=1):
        state_dict = dict()
        for i, lut in self.generator_3d_lut.items():
            state_dict[i+offset] = get_model_state_dict(lut)
        return state_dict

    def load_state_dict(self, state_dict:dict, offset=1):
        total_lut = len([key for key in state_dict.keys() if key >= offset])
        assert total_lut == len(self.generator_3d_lut), 'Generator_3DLUT_SUPPLEMENT owned number {} is not equal lut number {} that in the state_dict'.format(len(self.generator_3d_lut), total_lut)
        for i, lut in self.generator_3d_lut.items():
            load_model_state_dict(lut, state_dict[i+offset])
        return

    def enable_parallel(self):
        parallel = dict()
        for i, lut in self.generator_3d_lut.items():
            parallel[i] = torch.nn.parallel.DataParallel(lut)
        self.generator_3d_lut.clear()
        self.generator_3d_lut.update(parallel)
        return

    def enable_model_distributed(self, gpu_id):
        parallel = dict()
        for i, lut in self.generator_3d_lut.items():
            lut = torch.nn.SyncBatchNorm.convert_sync_batchnorm(lut)
            parallel[i] = torch.nn.parallel.DistributedDataParallel(lut, device_ids=[gpu_id])
        self.generator_3d_lut.clear()
        self.generator_3d_lut.update(parallel)
        return

    def train(self):
        for i, lut in self.generator_3d_lut.items():
            lut.train()
        return

    def eval(self):
        for i, lut in self.generator_3d_lut.items():
            lut.eval()
        return

    def foreach(self):
        for k, lut in self.generator_3d_lut.items():
            yield k, lut
        return

    def __len__(self):
        return len(self.generator_3d_lut)

    def __call__(self, x):
        output = dict()
        for i, lut in self.generator_3d_lut.items():
            output[i] = lut(x)
        return output