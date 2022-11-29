#!/bin/bash

# upgrade pip
pip install --upgrade pip

# install paddlepaddle cpu version
pip install paddlepaddle==2.3.2 -i https://mirror.baidu.com/pypi/simple

# install paddlepaddle cpu version from conda
# conda install paddlepaddle==2.4.0 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/

# install interpretdl dev. 
pip install -r requirements.txt  # -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .[dev]
