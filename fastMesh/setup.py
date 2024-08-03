from matplotlib.style import library
from setuptools import setup, find_packages
import unittest,os 
from typing import List
from glob import glob 

from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
print(find_packages())


headers = [os.path.join(os.path.split(os.path.abspath(__file__))[0], 'include')]
headers += ["/usr/local/include/opencv4", "/usr/local/include"]
library_dir = ['./']
library_name = ["opencv_core", "opencv_imgproc","opencv_imgcodecs"]

src_files = list(glob(os.path.join("src/*.cu"))) + list(glob(os.path.join("src/*.cpp")))


# """
# -DBUILD_OPENGL -DCUDA_GVDB_COPYDATA_PTX=\"cuda_gvdb_copydata.ptx\" -DCUDA_GVDB_MODULE_PTX=\"cuda_gvdb_module.ptx\" -DGLEW_STATIC -DGVDB_EXPORTS -Dgvdb_EXPORTS
# """
ext_modules = [
    CUDAExtension(
    name='fastMesh', 
    sources=src_files + ['binding.cpp'],
    library_dirs=library_dir,
    libraries=library_name,
    include_dirs=headers,
    extra_compile_args=["-g", "-O0"],
    extra_link_args=['-g']),
]

INSTALL_REQUIREMENTS = ['numpy', 'torch']


setup(
    name='fastMesh',
    description='torch fastMesh',
    author='anon',
    version='0.1',
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)