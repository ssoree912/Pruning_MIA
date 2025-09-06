#!/bin/bash

# Docker image name
IMAGE_NAME="pruning-mia"

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Run Docker container
echo "Starting Docker container..."
docker run -it --rm \
    --gpus all \
    --name $IMAGE_NAME-container \
    -v /Users/hwangsolhee/Desktop/prunning:/workspace \
    -v /home/user/Desktop/solhee/Datasets/CIFAR:/workspace/data \
    -w /workspace \
    -e NVIDIA_VISIBLE_DEVICES=0,1 \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    $IMAGE_NAME bash