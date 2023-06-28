# Base image
FROM nvcr.io/nvidia/pytorch:23.04-py3

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y software-properties-common \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 libgoogle-perftools-dev && \
    python3.10-dev python3.10-tk python3-html5lib python3-apt python3-pip python3.10-distutils && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

WORKDIR /app
RUN python3 -m pip install wheel

# Set python 3.10 and cuda 11.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 3 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    update-alternatives --set cuda /usr/local/cuda-11.8


# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Replace pillow with pillow-simd
RUN python3 -m pip uninstall -y pillow && \
    CC="cc -mavx2" python3 -m pip install -U --force-reinstall pillow-simd


RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer.so.7 && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7

RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser: /app
USER appuser
COPY --chown=appuser . .

COPY builder/setup.py /setup.py
RUN python3 setup.py

# Add src files (Worker Template)
ADD src .

ENV LD_PRELOAD=libtcmalloc.so
ENV PATH="$PATH:/home/appuser/.local/bin"


CMD python3 -u /handler.py
