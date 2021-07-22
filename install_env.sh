#!/bin/bash

while getopts n: opt
do
    case ${opt} in
        n) env_name=${OPTARG};;
        \?) echo "Usage cmd [-n]";;
    esac
done

git clone https://github.com/zhengp0/limetr.git
git clone https://github.com/ihmeuw-msca/MRTool.git

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda create -n "$env_name" -y -c conda-forge python=3.7 cyipopt gmp h5py
conda activate "$env_name"

pip install --global-option=build_ext --global-option "-I$CONDA_PREFIX/include/" pycddlib
cd limetr && git checkout master && make install && cd ..
cd MRTool && python setup.py install && cd ..

pip install -e .[internal]

rm -rf limetr
rm -rf MRTool
