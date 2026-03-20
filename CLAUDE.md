# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the llmmllab-runner repository.

## Repository Structure

The llmmllab ecosystem consists of several interconnected repositories:

```
schemas/              # YAML schema definitions (standalone repo at ../schemas)
├── llmmllab-schemas  # Git submodule in each app, defines all data models

proto/                # Protocol Buffer definitions (at ../proto)
├── llmmllab-schemas  # Git submodule (schemas)
├── llmmllab-proto    # Generated proto messages
└── Makefile          # Target: `make messages`

runner/               # This repository (at ./)
├── llmmllab-schemas  # Git submodule
├── llmmllab-proto    # Git submodule (proto)
├── models/           # Generated Pydantic models
└── Makefile          # Targets: `make models`, `make proto`

composer/             # ../composer
├── llmmllab-schemas  # Git submodule
├── llmmllab-proto    # Git submodule (proto)
├── models/           # Generated Pydantic models
└── Makefile          # Targets: `make models`, `make proto`

server/               # ../server
├── llmmllab-schemas  # Git submodule
├── llmmllab-proto    # Git submodule (proto)
├── models/           # Generated Pydantic models
└── Makefile          # Targets: `make models`, `make proto`
```

## Code Generation Workflow

### ⚠️ CRITICAL: Never Edit Generated Files

- **`models/*.py`** - Generated from llmmllab-schemas YAML files
- **`llmmllab-proto/*.proto`** messages - Generated from llmmllab-schemas YAML files
- **`gen/python/**/*.py`** - Generated from proto files via protoc

All generated code must be regenerated from source schemas. Manual edits will be lost and cause inconsistencies.

### To Update a Model or Proto Message

1. **Update the YAML schema** in the schemas repository (`../schemas`)
2. **Commit and push** the schema changes to main
3. **In `../proto`**:
   ```bash
   git submodule update --init --recursive --remote
   make messages
   ```
   Commit and push the generated proto messages to main
4. **In each application** (`runner`, `composer`, `server`):
   ```bash
   git submodule update --init --recursive --remote
   make models
   ```
   Commit the generated models

### To Update gRPC Services

1. **Update the `.proto` service definition** in `../proto`
2. **Commit and push** to main
3. **In each affected application**:
   ```bash
   git submodule update --init --recursive --remote
   make proto
   ```
   Commit the regenerated gRPC code

### Commands

### Development

```bash
# Install dependencies (in addition to system requirements)
pip install -r requirements.txt

# Install generated gRPC code (for local development)
pip install -e gen/python

# Run tests
pytest test/  # Python tests
```

### Code Generation

```bash
# Regenerate Pydantic models from llmmllab-schemas
make models

# Regenerate gRPC code from proto files
make proto
```

### Model Management

```bash
# Models are loaded from YAML/JSON config
# Default locations: /app/.models.yaml, /app/.models.json
# Or set MODELS_FILE_PATH env var
```

### gRPC Package Structure

The runner's gRPC code is generated to `gen/python/runner/v1/`:

```
gen/python/
├── runner/
│   ├── __init__.py
│   └── v1/
│       ├── __init__.py
│       ├── composer_runner_pb2.py
│       └── composer_runner_pb2_grpc.py
└── setup.py          # Install as editable package
```

To use the generated code:
```python
from runner.v1 import composer_runner_pb2, composer_runner_pb2_grpc
```

## Architecture

### Core Components

```
runner/                 # Main package
├── pipeline_factory.py # Pipeline creation with caching
├── pipeline_cache.py   # Local pipeline caching & memory management
├── pipelines/          # Pipeline implementations
│   ├── llamacpp/       # llama.cpp server-based pipelines
│   ├── txt2img/        # Text-to-image (Flux)
│   ├── img2img/        # Image-to-image (Flux)
│   └── imgtxt2txt/     # Multimodal (Qwen3-VL)
├── server_manager/     # Server process management
│   └── llamacpp.py     # llama.cpp server wrapper
├── utils/              # Utilities
│   ├── hardware_manager.py    # GPU memory management
│   ├── resizer.py             # Memory calculation
│   ├── intelligent_oom_recovery.py  # OOM recovery ML
│   └── model_loader.py        # Model config loading
├── server/             # gRPC server implementation
│   └── grpc.py         # gRPC server entry point
├── gen/                # Generated code (do not edit)
│   └── python/         # Python generated code
│       └── runner/     # Runner gRPC modules
├── models/             # Pydantic models (auto-generated)
```

### Key Architectural Patterns

**Pipeline System**: The runner uses a pluggable pipeline pattern. `PipelineFactory` creates appropriate pipelines based on model task and provider. Local providers (llama.cpp, stable diffusion) use cached pipelines with memory-aware eviction; remote providers (OpenAI, Anthropic) create transient pipelines.

**Multi-Tier Memory Management**:
- `EnhancedHardwareManager`: GPU detection, thermal monitoring, power management
- `LocalPipelineCacheManager`: LRU-style cache with intelligent eviction based on priority, memory size, access frequency
- `IntelligentOOMRecovery`: ML-based prediction of optimal parameters for memory-constrained scenarios

**Server Management Pattern**: `BaseServerManager` abstracts server process lifecycle. `LlamaCppServerManager` manages llama.cpp server with automatic port allocation, health checking, and graceful shutdown.

**Memory Calculation**: `Resizer.calculate_memory_breakdown()` computes GPU memory requirements with components: model weights (GPU layers), KV cache, activation buffer, overhead, CLIP model.

**Schema-Driven Models**: All data models are auto-generated from YAML schemas (in llmmllab-schemas). Never edit `models/*.py` directly.

### Key Entry Points

| Component | Entry Point |
|-----------|-------------|
| Pipeline factory | `runner.pipeline_factory` (singleton) |
| Local cache manager | `runner.local_pipeline_cache` (singleton) |
| Hardware manager | `runner.utils.hardware_manager` (singleton) |
| Model loader | `runner.utils.model_loader.ModelLoader` |
| Llama.cpp pipeline | `runner.pipelines.llamacpp.chat.ChatLlamaCppPipeline` |
| gRPC server | `runner.server.grpc` ( RunnerServicer class) |
| gRPC client | `runner.v1.composer_runner_pb2_grpc.RunnerServiceStub` |

### Provider Types

- **Local** (cached, managed servers): `ModelProvider.LLAMA_CPP`, `ModelProvider.STABLE_DIFFUSION_CPP`
- **Remote** (transient, no caching): `ModelProvider.OPENAI`, `ModelProvider.ANTHROPIC`

### Pipeline Types

- **Text**: `ChatLlamaCppPipeline` (llama.cpp), `ChatOpenAI`, `ChatAnthropic`
- **Embedding**: `EmbedLlamaCppPipeline`
- **Image**: `FluxPipe` (txt2img), `FluxKontextPipe` (img2img)
- **Multimodal**: Qwen3-VL pipeline

### gRPC Architecture

The runner exposes a gRPC service (`RunnerService`) for inter-service communication:

- **Port**: 50052 (default)
- **Package**: `runner.v1`
- **Service**: `RunnerService`

**RPC Methods**:
- `CreatePipeline`: Create a pipeline from a model profile
- `ExecutePipeline`: Execute a pipeline with streaming output
- `GenerateEmbeddings`: Generate embeddings for texts
- `GetCacheStats`: Get pipeline cache statistics
- `EvictPipeline`: Evict a pipeline from cache

**Generated Code**:
- `runner.v1.composer_runner_pb2`: Protocol buffer message classes
- `runner.v1.composer_runner_pb2_grpc`: gRPC stub and servicer classes

**Package Structure**:
```
gen/python/
├── runner/
│   ├── __init__.py
│   └── v1/
│       ├── __init__.py
│       ├── composer_runner_pb2.py
│       └── composer_runner_pb2_grpc.py
└── setup.py          # Install as editable package
```

## Local Development

### Starting the Runner

```bash
RUNNER_MODELS_FILE_PATH="$PWD/.models.yaml" PYTHONPATH="gen/python:." \
  .venv/bin/python -m server.grpc --port 50052
```

### Known Issues

**Proto package mismatch**: The runner's `gen/python/runner/v1/` contains two sets of generated code:
- `composer_runner_pb2.py` — old package `composer_runner.v1` (from `composer_runner.proto`)
- `runner_pb2.py` — current package `runner.v1` (from canonical `runner/v1/runner.proto`)

The composer client calls `runner.v1.RunnerService`. If the runner imports from `composer_runner_pb2`, gRPC returns "Method not found!". Fix: in `server/grpc.py`, alias the import:

```python
from runner.v1 import (
    runner_pb2 as composer_runner_pb2,
    runner_pb2_grpc as composer_runner_pb2_grpc,
)
```

**Interceptor bug**: The `DeadlineInterceptor` in `server/interceptors.py` tries to set a read-only attribute (`handler.unary_unary = ...`). Workaround: start with `enable_interceptors=False`.

**ModelProfile Pydantic vs Proto mismatch**: The Pydantic `ModelProfile` requires fields (`id`, `user_id`, `name`, `parameters`, `system_prompt`, `type`) not present in the proto `ModelProfile`. The `CreatePipeline` handler must provide defaults.

**PipelineFactory import**: `pipeline_factory` is a module-level *instance* in `pipelines/pipeline_factory.py`, not the module itself. Import as `from pipelines.pipeline_factory import pipeline_factory`.

### Model Configuration

The runner needs a `.models.yaml` file to know about available models. Set via `RUNNER_MODELS_FILE_PATH` env var. The file must be a **flat YAML list** (not nested under a key):

```yaml
- id: "model-id"
  name: "Display Name"
  model: "model-id"
  task: "TextToText"
  modified_at: "2026-01-01T00:00:00Z"
  digest: "some-digest"
  provider: "llama_cpp"
  details:
    family: "qwen"
    parameter_size: "30B"
    quantization_level: "Q4_K_M"
```

Without this file, `get_pipeline` raises "Model not found".

## Dependencies

- **llmmllab-schemas** - YAML schema definitions (for models)
- **llmmllab-proto** - Protocol Buffer definitions (for gRPC)

## Build & Deploy

```bash
# Build image
./build-image.sh multi-arch
./build-image.sh lsnode-3

# Deploy to k8s
make deploy
```