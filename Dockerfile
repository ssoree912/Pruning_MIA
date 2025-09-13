FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# 1) 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-venv python3.9-distutils \
    build-essential git curl wget ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgl1 libjpeg-turbo8 libpng16-16 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# 2) 프로젝트 작업 경로
WORKDIR /workspace

# 3) Python 3.9 가상환경 생성
RUN python3.9 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# 4) pip 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

# 5) 필요한 파이썬 패키지 설치 (PyTorch CU118 채널 사용)
#    requirements.txt를 이미지에 복사해 두고 설치
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt

# 6) 확인용(선택): 파이썬/토치 버전 및 CUDA 연결 상태 출력
RUN python -V && \
    python -c "import torch, torchvision, torchaudio; print(torch.__version__, torch.cuda.is_available()); print('tv', torchvision.__version__, 'ta', torchaudio.__version__)"

# 7) 기본 쉘
CMD ["bash"]