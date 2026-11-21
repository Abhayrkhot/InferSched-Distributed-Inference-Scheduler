# Build image for the real-model Engine backend (llama.cpp + CUDA).
# The control-plane dependencies match the CPU-only dev environment; the CUDA
# toolkit is added so llama.cpp can be built with GGML_CUDA=ON. The NVIDIA
# driver comes from the host at run time (docker run --gpus all).
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git curl cmake ninja-build pkg-config g++ \
      protobuf-compiler protobuf-compiler-grpc libprotobuf-dev \
      libgrpc++-dev libabsl-dev libpq-dev librdkafka-dev libhiredis-dev \
      python3 python3-venv python3-pip docker.io \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch 8.0.1 https://github.com/jtv/libpqxx.git /tmp/libpqxx \
    && cmake -S /tmp/libpqxx -B /tmp/libpqxx-build -G Ninja -DSKIP_BUILD_TEST=ON \
    && cmake --build /tmp/libpqxx-build --parallel \
    && cmake --install /tmp/libpqxx-build && rm -rf /tmp/libpqxx /tmp/libpqxx-build
ARG CUDA_SERIES=12-8
RUN curl -fsSL -o /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
      cuda-nvcc-${CUDA_SERIES} cuda-cudart-dev-${CUDA_SERIES} cuda-cccl-${CUDA_SERIES} \
      cuda-driver-dev-${CUDA_SERIES} cuda-nvml-dev-${CUDA_SERIES} libcublas-dev-${CUDA_SERIES} \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda
