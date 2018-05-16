from setuptools import (
    setup,
    find_packages,
)

setup(
    name='VincentNet',
    version='0.1dev',
    packages=find_packages(exclude=[
        'tests',
        '*.tests',
    ]),
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'mnist'
    ],

)