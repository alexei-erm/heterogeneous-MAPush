from setuptools import find_packages
from distutils.core import setup

setup(
    name='mapush',
    version='1.0.0',
    author='CMU Safe AI Lab',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='',
    description='multi-quadruped-pushing-environments',
    install_requires=['isaacgym',
                      'openrl',
                      'gymnasium==0.29.1',
                      'matplotlib',
                      'gym',
                      'debugpy',
                      'treelib',
                      'tensorboardX',
                      'pettingzoo',]
)