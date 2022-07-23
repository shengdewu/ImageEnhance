from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

if torch.cuda.is_available():
    print('Including CUDA code.')
    setup(
        name='trilinear',
        ext_modules=[
            CUDAExtension('trilinear', [
                'torch_1_x/src/trilinear_cuda.cpp',
                'torch_1_x/src/trilinear_kernel.cu',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
else:
    print('NO CUDA is found. Fall back to CPU.')
    setup(name='trilinear',
        ext_modules=[CppExtension('trilinear', ['torch_1_x/src/trilinear.cpp'])],
        cmdclass={'build_ext': BuildExtension})
