from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import numpy

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "pbctools._cpp.pbctools_cpp",
        [
            "pbctools/_cpp/pbctools_cpp.cpp",
            "pbctools/_cpp/bindings.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Path to numpy headers
            numpy.get_include(),
        ],
        language='c++',
        cxx_std=17,
        define_macros=[
            ('VERSION_INFO', '"dev"'),
        ],
        extra_compile_args=[
            '-O3',
            '-fopenmp',  # Enable OpenMP
            '-march=native',
            '-DWITH_OPENMP',
        ],
        extra_link_args=[
            '-fopenmp',
        ],
    ),
]

setup(
    name="pbctools",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)