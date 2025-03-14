# Use the official NVIDIA CUDA 12.1 base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ubuntu-desktop \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    python3-pip \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install ZED SDK
RUN wget https://download.stereolabs.com/zedsdk/4.2/ubuntu22 -O /tmp/zed_sdk.run \
    && chmod +x /tmp/zed_sdk.run \
    && /tmp/zed_sdk.run --silent

# Verify cuDNN and TensorRT versions
RUN echo "cuDNN version: $(cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2)" && \
    echo "TensorRT version: $(dpkg -l | grep tensorrt | awk '{print $3}')"

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install Python packages from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install QGroundControl (optional, if needed in the container)
RUN wget https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage -O /opt/QGroundControl.AppImage \
    && chmod +x /opt/QGroundControl.AppImage

# Install VSCode Server (optional, for browser-based VSCode)
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Set up working directory
WORKDIR /workspace

# Expose ports for Jupyter, CVAT, and VSCode
EXPOSE 8888 8080 3000

# Default command (start a shell)
CMD ["/bin/bash"]
