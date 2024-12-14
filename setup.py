from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "matrixobj",
        ["python/pybind.cpp"],
        depends=["src/MatrixObj.hpp", "src/VectorObj.hpp", "src/IterSolver.hpp", "src/SolverBase.hpp"]
    )
]

setup(
    name="algotools",
    version="1.0",
    author="NG YIN CHEANG",
    author_email="ycheang.ng@gmail.com",
    description="Python bindings for MatrixObj and related classes",
    url="https://github.com/Yin169/AlgorithmicToolkit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)