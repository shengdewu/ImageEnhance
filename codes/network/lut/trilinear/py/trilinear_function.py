import torch
import numpy as np
from .trilinear import *


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        """
        :param ctx:
        :param lut:
        :param x: normalized 0.0-1.9
        :return:
        """
        c, dim, dim1, dim2 = lut.size()
        bat, c, h, w = x.size()
        assert dim == dim1 and dim == dim2

        bins = 1.000001 / (dim - 1) #不采取整数的原因时防止当图像数据=1时越界（eg index = 1.0 * 32, left = index + 1 > 32

        output = np.zeros_like(x)
        x_numpy = x.numpy()
        for b in range(bat):
            trilinear_forward(x_numpy[b], lut.numpy(), output[b], w, h, bins)

        int_package = torch.IntTensor([dim, w, h, bat])
        float_package = torch.FloatTensor([bins])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, torch.from_numpy(output)

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        lut_dim, w, h, bat = int_package
        lut_dim, w, h, bat = int(lut_dim), int(w), int(h), int(bat)
        bins = float(float_package[0])

        lut_grad_numpy = lut_grad.numpy()
        x_numpy = x.numpy()
        x_grad_numpy = x_grad.numpy()
        for b in range(bat):
            trilinear_backword(x_numpy[b], x_grad_numpy[b], lut_grad_numpy, h, w, bins)

        return torch.from_numpy(lut_grad_numpy), x_grad
