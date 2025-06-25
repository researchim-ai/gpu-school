FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

LABEL maintainer="GPU-School researchim-ai"

# Обновляем систему и ставим зависимости
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        build-essential cmake git \
        ocl-icd-opencl-dev opencl-headers clinfo \
        libclblast-dev \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# По умолчанию контейнер запускается в интерактивном режиме
CMD ["/bin/bash"] 