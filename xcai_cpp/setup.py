from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='pad_tfm',
            ext_modules=[cpp_extension.CppExtension('pad_tfm', ['pad_tfm.cpp'])],
            cmdclass={'build_ext': cpp_extension.BuildExtension})

Extension(
       name='pad_tfm',
       sources=['pad_tfm.cpp'],
       include_dirs=cpp_extension.include_paths(),
       language='c++'
)
