#!/bin/bash

# upgrade pip
pip install --upgrade pip

# install paddlepaddle cpu version
pip install paddlepaddle  # -i https://mirror.baidu.com/pypi/simple

# install interpretdl dev. 
pip install -r requirements.txt  # -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .[dev]
