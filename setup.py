#!/usr/bin/env python3

import os
from setuptools import setup, find_packages

# get key package details from py_pkg/__version__.py
about = {
    '__title__': 'interpretdl',
    '__description__': 'interpretation of deep learning models',
    '__version__': '0.1.8'
}  # type: ignore
here = os.path.abspath(os.path.dirname(__file__))
# with open(os.path.join(here, 'py_pkg', '__version__.py')) as f:
#     exec(f.read(), about)

# load the README file and use it as the long_description for PyPI
#with open('README.md', 'r') as f:
#    readme = f.read()

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
long_description = 'Home-page: https://github.com/PaddlePaddle/InterpretDL \
Author: Baidu-BDL \
Author-email: autodl@baidu.com \
License: Apache 2.0 \
Description: InterpretDL, short for interpretation of deep learning models, is a model interpretation toolkit for PaddlePaddle models.'

setup(
    name=about['__title__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type='text/plain',
    version=about['__version__'],
    #url=about['__url__'],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=[
        'numpy', 'requests', 'scikit-image', 'scikit-learn', 'tqdm', 'pillow',
        'opencv-python', 'matplotlib', 'IPython'
    ],
    license='Apache 2.0',
    zip_safe=False,
    entry_points={
        'console_scripts': ['interpretdl=interpretdl.command:main'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='interpreters for Paddle models')
