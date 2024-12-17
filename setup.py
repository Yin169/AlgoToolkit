from setuptools import setup, Extension
import pybind11
import sys

# Define the extension module
ext_modules = [
    Extension(
        'FASTSolver',
        ['pybind_interface.cpp'],  # Replace with your Pybind11 interface file
        include_dirs=[pybind11.get_include()],  # Pybind11 headers
        language='c++',
        extra_compile_args=['-std=c++17'],  # Use C++11 standard
    ),
]

# Setup configuration
setup(
    name='FASTSolver',
    version='0.1.0',
    description='Numerical solvers and linear algebra tools using Pybind11',
    author='NG YIN CHEANG',
    author_email='yincheang.ng@outlook.com',
    ext_modules=ext_modules,
    zip_safe=False,
)
