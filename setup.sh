#!/usr/bin/env bash

# to run:
# chmod +x setup.sh
# source setup.sh


conda create --name nlpfromscratch python=3.8

conda activate nlpfromscratch

# conda install pytorch==2.0.1 torchvision torchaudio -c pytorch
# Use below instead if you would be using GPU
conda install pytorch==2.0.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install torch torchvision torchaudio -U


python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
conda install openjdk=11

pip install lxml==4.6.3
pip install -e git+https://github.com/titipata/scipdf_parser.git#egg=scipdf
pip install spacy==3.6
pip install bs4
pip install chardet
pip install transformers
pip install scikit-learn
