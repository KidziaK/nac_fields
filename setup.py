from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module
extensions = [
    Extension(
        "nac.clustering",
        ["nac/clustering.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"],
    )
]

setup(
    name="nac-fields",
    version="0.1.0",
    ext_modules=cythonize(extensions, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'language_level': 3,
    }),
    zip_safe=False,
    install_requires=[
        "numpy>=1.20.0",
        "cython>=0.29.0",
    ],
)
