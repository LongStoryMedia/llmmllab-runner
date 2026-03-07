# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development

```bash
# Install dependencies (in addition to system requirements)
pip install -r requirements.txt

# Install generated gRPC code (for local development)
pip install -e runner/gen/python

# Run tests
pytest test/  # Python tests
```

### Code Generation

```bash
# Regenerate gRPC code from proto files
./build.sh
```

### Model Management

```bash
# Models are loaded from YAML/JSON config
# Default locations: /app/.models.yaml, /app/.models.json
# Or set MODELS_FILE_PATH env var
```

### gRPC Package Structure

The runner's gRPC code is generated to `runner/gen/python/runner/v1/`:

```
runner/gen/python/
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

**Schema-Driven Models**: All data models are auto-generated from YAML schemas (in parent repo). Never edit `models/*.py` directly.

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
runner/gen/python/
├── runner/
│   ├── __init__.py
│   └── v1/
│       ├── __init__.py
│       ├── composer_runner_pb2.py
│       └── composer_runner_pb2_grpc.py
└── setup.py          # Install as editable package
```