# Base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
COPY builder/system_packages.sh /system_packages.sh
RUN bash /system_packages.sh

# Clone kohya-ss/sd-scripts
RUN git clone https://github.com/kohya-ss/sd-scripts.git && \
    cd sd-scripts && \
    git checkout 0cfcb5a49cf813547d728101cc05edf1a9b7d06c

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files (Worker Template)
ADD src .

CMD python3 -u /handler.py
