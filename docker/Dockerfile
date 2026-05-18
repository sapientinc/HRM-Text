FROM python:3.13-bookworm

ARG CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
ARG CUDA_MD5=e281945bf4ea42788db29457ebc1e6ae

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

# Install necessary and utility packages
# NOTE: libxml2 is required by CUDA installer
# NOTE: The last line is InfiniBand drivers
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    wget \
    git \
    openssh-client \
    libxml2 \
    build-essential \
    ca-certificates \
    tmux \
    rsync \
    psmisc \
    iproute2 \
    iftop \
    htop \
    perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 \
    && rm -rf /var/lib/apt/lists/*

# Download and install CUDA
RUN wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL \
    && echo "$CUDA_MD5 cuda_installer.run" | md5sum -c - \
    && sh cuda_installer.run --silent --toolkit --override \
    && rm cuda_installer.run

# Install PyTorch, torchvision, torchaudio, and other requirements.
# Root requirements.txt stays the source of truth for normal Python packages.
# Limit extension building to 1/4 of CPU cores to prevent OOM errors.
COPY requirements.txt /tmp/requirements.txt
COPY docker/requirements /tmp/docker-requirements/
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL \
    && pip3 install --no-cache-dir packaging ninja wheel setuptools setuptools-scm \
    && MAX_JOBS=$(python -c "import os; print(max(1, os.cpu_count() // 4))") NVCC_THREADS=2 pip3 install --no-cache-dir --no-build-isolation -r /tmp/docker-requirements/torch_extensions.txt \
    && pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm -rf /tmp/docker-requirements /tmp/requirements.txt

WORKDIR /workspace

# Environment-only image: mount or clone HRM-Text at runtime.
CMD [ "/bin/bash" ]
