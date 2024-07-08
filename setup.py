from setuptools import setup

setup(
    name='pyagg',
    version='1.0',
    packages=['pyagg'],
    install_requires=['pyopencl', 'numpy', 'py-algorand-sdk', 'pytest'],
    entry_points={'console_scripts': ['pyagg=pyagg:pyagg_cli', 'pyagg-optimize=pyagg:pyagg_optimize_cli']},
    package_data={'pyagg': ['kernel.cl', 'kernel.cl.LICENSE.md']},
    include_package_data=True,
)