import torch
# from .py.trilinear_function import TrilinearInterpolationFunction
from .cpp.trilinear_function import TrilinearInterpolationFunction


class TrilinearInterpolationModel(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolationModel, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)