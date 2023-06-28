#!/bin/bash

apt-get install ffmpeg libsm6 libxext6  -y

apt-get update && \
apt-get upgrade -y && \
apt-get install -y software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa && \
apt-get update && \
apt-get install -y --no-install-recommends \
ffmpeg libsm6 libxext6 git curl libgl1 libglib2.0-0 libgoogle-perftools-dev && \
python3.10-dev python3.10-tk python3-html5lib python3-apt python3-pip python3.10-distutils && \
apt-get autoremove -y && \
apt-get clean -y && \
rm -rf /var/lib/apt/lists/*
