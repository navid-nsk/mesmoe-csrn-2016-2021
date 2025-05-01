from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='csrn_cuda',
    ext_modules=[
        CUDAExtension('csrn_cuda', [
            'csrn_cuda.cpp',
            'csrn_cuda_kernels.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)