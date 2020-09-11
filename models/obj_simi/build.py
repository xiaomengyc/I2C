import os

from setuptools import setup
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension

sources = ['src/obj_simi_api.cpp']
with_cuda = True
defines = [('WITH_CUDA', None)]

this_file = os.path.dirname(os.path.realpath(__file__))


setup(
    cmdclass={'build_ext': BuildExtension},
    ext_modules= [
        CppExtension("obj_simi",
                  sources=sources,
                  language="c++",
                  include_dirs=torch.utils.cpp_extension.include_paths()
                  )
    ]
)
