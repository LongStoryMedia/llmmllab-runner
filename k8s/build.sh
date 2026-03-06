#!/bin/bash

set -e

REGISTRY=${REGISTRY:-192.168.0.71:31500}
TAG=${TAG:-latest}

# Build image for runner (runs on lsnode-3 with nvidia runtime)
# Build directly on lsnode-3 (AMD64) for best compatibility with GPU drivers
echo "Building runner image on lsnode-3 (AMD64)..."
ssh root@lsnode-3.local "cd /data/code-base/runner && docker build -t ${REGISTRY}/runner:${TAG} -f k8s/Dockerfile . --push"

echo "Runner image built and pushed: ${REGISTRY}/runner:${TAG}"