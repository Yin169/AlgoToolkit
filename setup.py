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
        
        # Get OpenBLAS path from brew
        openblas_path = subprocess.check_output(['brew', '--prefix', 'openblas']).decode('utf-8').strip()
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE=Release',
            f'-DOpenBLAS_HOME={openblas_path}',
        ]
        
        build_args = ['--config', 'Release']
        
        if platform.system() == "Darwin":
            cmake_args += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15']
            build_args += ['--', '-j2']
        
        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

setup(
    name='fastsolver',
    version='0.1.0',
    author='Your Name',
    description='Fast numerical solver library',
    long_description='',
    ext_modules=[CMakeExtension('fastsolver')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.6',
)