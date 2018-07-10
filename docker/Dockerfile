FROM nvidia/opengl:1.0-glvnd-devel-ubuntu16.04

ENV http_proxy http://proxy.int.europe.naverlabs.com:3128
ENV https_proxy https://proxy.int.europe.naverlabs.com:3128
ENV HTTP_PROXY http://proxy.int.europe.naverlabs.com:3128
ENV HTTPS_PROXY https://proxy.int.europe.naverlabs.com:3128

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    cmake \
    pkg-config \
    libpng-dev \
    libjpeg-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libegl1-mesa-dev \
    ca-certificates \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.5.5
WORKDIR /opt
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-devel && \
    /opt/conda/bin/conda install -c mingfeima --name pytorch-py$PYTHON_VERSION mkldnn && \
    /opt/conda/bin/conda clean -ya && \
    /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install gym atari_py matplotlib jupyter tensorboardX
ENV PATH /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:/opt/conda/bin:$PATH

# Compile PyTorch
WORKDIR /opt
RUN git clone --single-branch -b master https://github.com/pytorch/pytorch && \
    cd pytorch && \
    git submodule update --init && \
    NO_CUDA=1 python setup.py install

WORKDIR /
