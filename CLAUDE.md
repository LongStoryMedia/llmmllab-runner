# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development

```bash
# Install dependencies (in addition to system requirements)
pip install -r requirements.txt

# Run tests
pytest test/  # Python tests
```

### Model Management

```bash
# Models are loaded from YAML/JSON config
# Default locations: /app/.models.yaml, /app/.models.json
# Or set MODELS_FILE_PATH env var
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
└── models/             # Pydantic models (auto-generated)
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

### Provider Types

- **Local** (cached, managed servers): `ModelProvider.LLAMA_CPP`, `ModelProvider.STABLE_DIFFUSION_CPP`
- **Remote** (transient, no caching): `ModelProvider.OPENAI`, `ModelProvider.ANTHROPIC`

### Pipeline Types

- **Text**: `ChatLlamaCppPipeline` (llama.cpp), `ChatOpenAI`, `ChatAnthropic`
- **Embedding**: `EmbedLlamaCppPipeline`
- **Image**: `FluxPipe` (txt2img), `FluxKontextPipe` (img2img)
- **Multimodal**: Qwen3-VL pipeline