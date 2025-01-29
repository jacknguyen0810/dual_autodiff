from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "dual_autodiff_x.dual",
        ["src/dual_autodiff_x/dual.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
    }),
    package_dir={"": "src"},
    packages=["dual_autodiff_x"],
    package_data={"dual_autodiff_x": ["*.so", "*.pyd"]},
    exclude_package_data={"dual_autodiff_x": ["*.pyx", "*.py"]},
    zip_safe=False,
)