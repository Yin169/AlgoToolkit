# setup.py
import os
from setuptools import setup, Extension
import pybind11

source_file = []
def getallsource(path="src"):
	if "/." in path:
		return
	if path.endswith(".hpp"):
		source_file.append(path)
		return
	child = os.listdir(path)
	for c in child:
		getallsource(os.path.join(path, c))
	return 

getallsource()
source_file = ["#include ../"+ i for i in source_file]

# Define the extension module
fastsolver_module = Extension(
    'fastsolver',
    sources=['python/pybind.cpp'],
	include_dirs=[pybind11.get_include(), os.getenv('OPENBLAS_INCLUDE', '/opt/homebrew/opt/openblas/include')],  # OpenBLAS 头文件路径
    library_dirs=[os.getenv('OPENBLAS_LIB', '/opt/homebrew/opt/openblas/lib')],  # OpenBLAS 库路径
    language='c++',
	extra_compile_args=['-std=c++17']
)

# Setup configuration
setup(
    name='fastsolver',
    version='0.1.1',
    description='Python bindings for FASTSolver',
    ext_modules=[fastsolver_module],
    install_requires=['pybind11'],
)