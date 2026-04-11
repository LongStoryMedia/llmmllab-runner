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

### Local Development

For local development, models are loaded from `./.models.local.yaml` (or `./.models.yaml` if `.models.local.yaml` doesn't exist).

```bash
# Set up local environment (creates .env.local with RUNNER_LOCAL_MODE=1)
# Source the local environment
source .env.local

# Start gRPC server locally
make start-local

# Start with debug logging
make start-local-debug
```

### Production/Container

In production (Docker/k8s), models are loaded from `/app/.models.yaml`.

```bash
# Start gRPC server (requires RUNNER_MODELS_FILE_PATH or /app/.models.yaml)
make start

# Start with debug logging
make start-debug
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUNNER_MODELS_CONFIG` | Full path to models config file | `./.models.local.yaml` (if exists) or `/app/.models.yaml` |
| `LOG_LEVEL` | Logging level | `debug` (local) / `info` (prod) |

**Model file paths in config:**
Paths to actual model files (GGUF, clip models) in the `.models.yaml` config should be absolute paths.

### Tests

```bash
# Run all tests
make test

# Run unit tests
make test-unit

# Run integration tests
make test-integration

# Run with coverage
make test-cover
```

### Linting

```bash
# Run linting
make lint

# Auto-fix linting issues
make lint-fix
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