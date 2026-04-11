# Multi-stage build for Runner service
# Build stage - includes full CUDA toolkit for compilation
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    g++ \
    ninja-build \
    libcurl4-openssl-dev \
    pkg-config \
    ccache \
    zstd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set up CUDA library paths for building
ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs

# Create symlink for CUDA driver stub (needed for linking during build)
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN cd / && git clone https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    git checkout tags/b8149 && \
    git checkout -b llmmll

WORKDIR /llama.cpp
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;86" \
    -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j6

# Runtime stage - minimal CUDA runtime
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    SHARED_VENV="/opt/venv/shared" \
    RUNNER_VENV="/opt/venv/shared" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/cuda/lib64" \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    HF_HOME=/root/.cache/huggingface \
    BNB_CUDA_VERSION=128 \
    TOKENIZERS_PARALLELISM=true \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_CUDA_ARCH_LIST="7.5;8.6" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_CACHE_PATH=/tmp/cuda_cache \
    CUDA_LAUNCH_BLOCKING=0 \
    GRPC_PORT="50052" \
    GRPC_MAX_WORKERS=10 \
    GRPC_MAX_MESSAGE_SIZE=104857600 \
    GRPC_MAX_CONCURRENT_RPCS=100 \
    GRPC_ENABLE_REFLECTION=true \
    GRPC_SECURITY_ENABLED=false \
    GRPC_USE_TLS=false \
    IMAGE_DIR="/root/images" \
    IMAGE_GENERATION_ENABLED="true" \
    MAX_IMAGE_SIZE="2048" \
    IMAGE_RETENTION_HOURS="24"

# Install only runtime dependencies (no build tools!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    binutils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Copy only the compiled llama.cpp from builder (no wheels needed)
COPY --from=builder /llama.cpp /llama.cpp

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create single shared virtual environment
RUN python3 -m venv ${SHARED_VENV}

# Install dependencies in shared environment
RUN . ${SHARED_VENV}/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir diskcache>=5.6.1 typing-extensions>=4.5.0 numpy>=1.20.0 jinja2>=2.11.3 && \
    deactivate

# Install runner package in development mode
RUN . ${SHARED_VENV}/bin/activate && \
    pip install --no-cache-dir -e /app && \
    deactivate

# Make scripts executable
RUN mkdir -p /tmp/cuda_cache

# Expose gRPC port
EXPOSE 50052

# Run runner gRPC server
CMD ["/opt/venv/shared/bin/python", "-m", "server.grpc"]