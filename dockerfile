# Use official PyTorch image with CUDA 11.8 support
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory inside the container
WORKDIR /workspace

# Install system dependencies required for OpenCV, Git, and C++ building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase into the container
COPY . .

# Build and install custom CUDA submodules
RUN pip install --no-cache-dir ./submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir ./submodules/fused-ssim
RUN pip install --no-cache-dir ./submodules/simple-knn

# Default entry command
CMD ["/bin/bash"]
