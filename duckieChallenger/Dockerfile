# Definition of Submission container

# We start from tensorflow-gpu image
ARG UBUNTU_VERSION=16.04

ARG ARCH
ARG CUDA=10.0
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
SHELL ["/bin/bash", "-c"]

# let's create our workspace, we don't want to clutter the container
WORKDIR /${proj_dir}

ARG ARCH
ARG CUDA
ARG CUDNN=7.4.1.5-1
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=2.3.0

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        cuda-cublas-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

RUN [ ${ARCH} = ppc64le ] || (apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda${CUDA} \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*)

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# DO NOT CHANGE PYTHON VERSION
# - if you do also change ppa repository to pull python version from
ARG _PY_SUFFIX=3.7
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        software-properties-common \
        python3-pip \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y ${PYTHON} ${PYTHON}-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# pip has to be installed before setuptools, setuptools has to be installed before tensorflow
RUN ${PYTHON} -m pip install --no-cache-dir -U pip
RUN ${PYTHON} -m pip install --no-cache-dir -U setuptools
# also useful
RUN ${PYTHON} -m pip install --no-cache-dir ipython requests numpy pandas quandl
RUN ${PYTHON} -m pip install --no-cache-dir tensorflow-gpu==${TF_PACKAGE_VERSION}


# RUN apt-get update && apt-get install -y \
#    ${PYTHON} \
#    ${PYTHON}-pip

#RUN ${PIP} --no-cache-dir install --upgrade \
#    pip \
#    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.

RUN ${PIP} install ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

# from https://github.com/tensorflow/tensorflow/issues/10776
# this may not be needed if you use pytorch image
RUN cd /usr/local/cuda/lib64 \
    && mv stubs/libcuda.so ./ \
    && ln -s libcuda.so libcuda.so.1 \
    && ldconfig

# DO NOT MODIFY: your submission won't run if you do
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         gcc \
         libc-dev\
         git \
         bzip2 \
         libgl1-mesa-glx \
         python3-tk && \
     rm -rf /var/lib/apt/lists/*

# let's create our workspace, we don't want to clutter the container
WORKDIR /workspace


ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}

# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
COPY requirements.* ./
RUN pip3 install --use-feature=2020-resolver -r requirements.txt

# let's copy all our solution files to our workspace
# if you have more file use the COPY command to move them to the workspace
COPY solution.py /workspace
COPY FrankNet.h5 /workspace
COPY helperFncs.py /workspace
# we make the workspace our working directory
WORKDIR /workspace

# DO NOT MODIFY: your submission won't run if you do
ENV DUCKIETOWN_SERVER=evaluator

# let's see what you've got there...
ENTRYPOINT ["python", "solution.py"]
