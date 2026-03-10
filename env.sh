#! /bin/bash

export RUNNER_MODELS_FILE_PATH="$PWD/.models.yaml"
export RUNNER_LLAMA_SERVER_EXECUTABLE="/home/lsm/llama.cpp/build/bin/llama-server"
export RUNNER_GPU_POWER_CAP_PCT="85"
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export CUDA_SCALE_LAUNCH_QUEUES=4x