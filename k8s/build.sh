#!/bin/bash

set -e

REGISTRY=${REGISTRY:-192.168.0.71:31500}
TAG=${TAG:-latest}

# Build image for runner (runs on lsnode-3 with nvidia runtime)
# Build directly on the target node for best compatibility
docker build \
    -t ${REGISTRY}/runner:${TAG} \
    -f k8s/Dockerfile .

echo "Runner image built: ${REGISTRY}/runner:${TAG}"
echo "Push the image with: docker push ${REGISTRY}/runner:${TAG}"