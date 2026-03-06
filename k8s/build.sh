#!/bin/bash

set -e

REGISTRY=${REGISTRY:-192.168.0.71:31500}
TAG=${TAG:-latest}

# Build image for runner (runs on lsnode-3 with nvidia runtime)
# Build directly on lsnode-3 (AMD64) for best compatibility with GPU drivers
echo "Building runner image on lsnode-3 (AMD64)..."

# Create a temp directory, sync code, build, push, then clean up
ssh root@lsnode-3.local "
  TEMP_DIR=\$(mktemp -d)
  trap 'rm -rf \${TEMP_DIR}' EXIT
  echo \"Created temp directory: \${TEMP_DIR}\"

  echo \"Syncing code to temp directory...\"
  cp -r /data/code-base/* \${TEMP_DIR}/

  echo \"Building runner image...\"
  cd \${TEMP_DIR}/runner && docker build -t \${REGISTRY}/runner:\${TAG} -f k8s/Dockerfile . --push

  echo \"Runner image built and pushed: \${REGISTRY}/runner:\${TAG}\"
"

echo "Runner build complete: ${REGISTRY}/runner:${TAG}"