""" from https://github.com/jaywalnut310/glow-tts """

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

# Fix output directory to current directory
extensions = [
    Extension(
        "core",
        ["core.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='monotonic_align',
    ext_modules=cythonize(extensions),
)

