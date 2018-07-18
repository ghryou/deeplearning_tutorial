FROM ubuntu:16.04

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
    python3 \
    python3-dev \
    python3-future \
    python3-setuptools \
    python3-wheel \
    python3-tk \
    python-opengl \
    apt-utils \
    g++  \
    zlib1g-dev \
    xvfb \
    libav-tools \
    xorg-dev \
    libboost-all-dev \
    libsdl2-dev \
    swig \
    libopenblas-base \
    libatlas-dev \
    cython3  \
    tmux && \
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
    /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install atari_py matplotlib jupyter tensorboardX PyOpenGL piglet pyglet JSAnimation ipywidgets scipy gym[all]
ENV PATH /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:/opt/conda/bin:$PATH

# Compile PyTorch
WORKDIR /opt
RUN git clone --single-branch -b master https://github.com/pytorch/pytorch && \
    cd pytorch && \
    git submodule update --init && \
    NO_NNPACK=1 NO_DISTRIBUTED=1 NO_CUDA=1 NO_SYSTEM_NCCL=1 NO_CUDNN=1 python setup.py install

# Compile TorchVision
WORKDIR /opt
RUN git clone --single-branch -b master https://github.com/pytorch/vision && \
    cd vision && \
    python setup.py install

# Compile House3D
# WORKDIR /opt
# RUN git clone --single-branch -b master http://github.com/facebookresearch/House3D && \
#     cd House3D && \
#     git submodule update --init && \
#     cd renderer && \
#     SYSTEM=conda.linux PYTHON_CONFIG=/opt/conda/envs/pytorch-py$PYTHON_VERSION/bin/python3-config make && \
#     cd .. && \
#     pip install .

WORKDIR /
COPY run.sh /
RUN chmod +x /run.sh

# Juoyter notebook port
EXPOSE 8888

ENTRYPOINT ["/run.sh"]
CMD ["jupyter notebook --allow-root --ip 0.0.0.0"]
