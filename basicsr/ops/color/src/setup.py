from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='color',
    ext_modules=[
        CUDAExtension(
            'color', 
            ['./src/color_cuda.cpp',
             './src/color_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2', \
                                         '-gencode=arch=compute_52,code=sm_52', \
                                         '-gencode=arch=compute_60,code=sm_60', \
                                         '-gencode=arch=compute_61,code=sm_61', \
                                         '-gencode=arch=compute_62,code=sm_62', \
                                         '-gencode=arch=compute_70,code=sm_70', \
                                         '-gencode=arch=compute_75,code=sm_75', \
                                         '-gencode=arch=compute_75,code=compute_75'
                                        ]
                               }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
