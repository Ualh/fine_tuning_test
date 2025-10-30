FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    CURL_CA_BUNDLE= \
    REQUESTS_CA_BUNDLE= \
    SSL_CERT_FILE= \
    PYTHONHTTPSVERIFY=0 \
    HF_HUB_DISABLE_SSL_VERIFY=1 \
    HF_HUB_ENABLE_XET=0 \
    GIT_SSL_NO_VERIFY=1

RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true \
 && sed -i '/developer\.download\.nvidia\.com/d' /etc/apt/sources.list || true \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg git git-lfs wget tzdata \
    python3 python3-pip python3-venv \
    build-essential cmake \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libz-dev libssl-dev libffi-dev libxml2-dev \
    libjpeg-dev unzip locales \
 && git lfs install \
 && locale-gen en_US.UTF-8 \
 && update-ca-certificates \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip certifi
RUN pip install numpy==1.24.4

RUN pip install --no-cache-dir --default-timeout=300 \
      --index-url https://download.pytorch.org/whl/cu121 \
      --trusted-host download.pytorch.org \
      torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121

WORKDIR /app
## Install Python dependencies from requirements.txt (includes llmcompressor)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CMD ["bash"]
