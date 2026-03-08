# llmmllab-runner

Model execution service for llmmllab. Provides gRPC interface for pipeline management and model execution.

## Overview

The llmmllab-runner provides:

- gRPC service for pipeline management and execution
- Model execution pipelines (text, image, embeddings, multimodal)
- Local pipeline caching with memory-aware eviction
- GPU hardware management

## Architecture

```
runner/
├── pipeline_factory.py   # Pipeline creation with caching
├── pipeline_cache.py     # Local pipeline caching & memory management
├── pipelines/            # Pipeline implementations
│   ├── llamacpp/         # llama.cpp server-based pipelines
│   ├── txt2img/          # Text-to-image (Flux)
│   ├── img2img/          # Image-to-image (Flux)
│   └── imgtxt2txt/       # Multimodal (Qwen3-VL)
├── server_manager/       # Server process management
│   └── llamacpp.py       # llama.cpp server wrapper
├── utils/                # Utilities
│   ├── hardware_manager.py    # GPU memory management
│   ├── resizer.py             # Memory calculation
│   ├── intelligent_oom_recovery.py  # OOM recovery ML
│   └── model_loader.py        # Model config loading
├── server/               # gRPC server implementation
│   └── grpc.py           # gRPC server entry point
├── gen/                  # Generated code (do not edit)
│   └── python/           # Python generated code
│       └── runner/       # Runner gRPC modules
├── models/               # Pydantic models (auto-generated)
└── test/                 # Test suite
```

## Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install generated gRPC code (for local development)
pip install -e gen/python
```

## Usage

### Development

```bash
# Start gRPC server
make start

# Start with debug logging
make start-debug

# Run tests
make test

# Run linting
make lint
```

### Build & Deploy

```bash
# Build Docker image
./build-image.sh multi-arch
./build-image.sh lsnode-3

# Deploy to k8s
make deploy
```

## Dependencies

- **llmmllab-schemas** - YAML schema definitions (for models)
- **llmmllab-proto** - Protocol Buffer definitions (for gRPC)

## gRPC Service

The runner exposes a gRPC service (`RunnerService`) on port 50052:

- **Package**: `runner.v1`
- **Service**: `RunnerService`

### RPC Methods

| Method | Description |
|--------|-------------|
| `CreatePipeline` | Create a pipeline from a model profile |
| `ExecutePipeline` | Execute a pipeline with streaming output |
| `GenerateEmbeddings` | Generate embeddings for texts |
| `GetCacheStats` | Get pipeline cache statistics |
| `EvictPipeline` | Evict a pipeline from cache |

## Provider Types

- **Local** (cached, managed servers): `ModelProvider.LLAMA_CPP`, `ModelProvider.STABLE_DIFFUSION_CPP`
- **Remote** (transient, no caching): `ModelProvider.OPENAI`, `ModelProvider.ANTHROPIC`

## Pipeline Types

- **Text**: `ChatLlamaCppPipeline` (llama.cpp), `ChatOpenAI`, `ChatAnthropic`
- **Embedding**: `EmbedLlamaCppPipeline`
- **Image**: `FluxPipe` (txt2img), `FluxKontextPipe` (img2img)
- **Multimodal**: Qwen3-VL pipeline