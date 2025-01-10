import os
import sys
import platform
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        # Install dependencies with conan
        subprocess.check_call(['conan', 'install', '.', '--build=missing'])
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE=Release',
            f'-DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake',
            '-DBUILD_PYTHON_BINDINGS=ON'
        ]

        if platform.system() == "Darwin":
            cmake_args += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15']

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', self.build_temp, '--config', 'Release'])

setup(
    name='fastsolver',
    version='0.1.0',
    author='NG YIN CHEANG',
    description='Fast numerical solver library',
    packages=['fastsolver'],
    package_dir={'fastsolver': 'python'},
    ext_modules=[CMakeExtension('fastsolver.core')],
    cmdclass={'build_ext': CMakeBuild},
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.15.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
    ],
)