       I now have all the information needed. Here is the comprehensive report.

       ---
       Complete Codebase Map for Removal

       1. MEMORY ESTIMATION / "SUFFICIENT SPACE" CHECKS

       Core Memory Estimation Files

       /Users/lons7862/workspace/llmmll/llmmllab-runner/utils/resizer.py
       - Defines MemoryBreakdown (TypedDict) and Resizer class
       - Resizer.calculate_memory_breakdown() computes GPU memory requirements for LLM inference (model weights, KV cache, activation buffer, overhead, CLIP model)
       - Contains architecture heuristics (_get_model_architecture_details) for estimating layers/hidden sizes by model family
       - Contains quantization parsing (_parse_quantization_bits) and parameter size parsing (_parse_parameter_size)

       /Users/lons7862/workspace/llmmll/llmmllab-runner/utils/hardware_manager.py
       - Defines MemoryConfig (dataclass at line 30, NOT the Pydantic MemoryConfig in models/), CUDAContextManager, GPUProcessManager, MemoryManager, EnhancedHardwareManager
       - is_memory_related_error() -- standalone function for detecting OOM errors
       - EnhancedHardwareManager.check_memory_available() -- checks if required bytes are available across GPUs
       - EnhancedHardwareManager.clear_memory() -- clears GPU memory (kills processes, destroys CUDA contexts)
       - EnhancedHardwareManager.check_gpu_thermals() -- GPU temperature monitoring
       - hardware_manager singleton instance at line 728
       - NOTE: This file is used beyond just memory estimation. It manages GPU hardware, process management, thermal monitoring, and CUDA contexts. Some of this may need to be kept.

       /Users/lons7862/workspace/llmmll/llmmllab-runner/pipelines/pipeline_cache.py
       - LocalPipelineCacheManager.estimate_memory() (line 596) -- estimates memory for a model+profile using Resizer
       - LocalPipelineCacheManager._calculate_corrected_memory_breakdown() (line 654) -- delegates to self._resizer.calculate_memory_breakdown()
       - LocalPipelineCacheManager._ensure_memory() (line 671) -- multi-step eviction logic that calls hardware_manager.check_memory_available() repeatedly
       - LocalPipelineCacheManager._convert_model_parameters_to_optimal() (line 575) -- converts ModelParameters to OptimalParameters for use with Resizer
       - Uses self._resizer = Resizer() (line 137) and self._oom_recovery = IntelligentOOMRecovery() (line 143/151)
       - The entire get_or_create() method (line 167) integrates memory estimation, eviction, and OOM recovery

       Generated Model Files for Memory/GPU Stats

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/system_gpu_stats.py
       - Pydantic model SystemGPUStats -- used by IntelligentOOMRecovery for GPU stats

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/resource_usage.py
       - Pydantic model ResourceUsage with gpu_memory_mb field -- general metrics model, may not need removal

       ---
       2. ML-BASED PARAMETER ADJUSTMENT / OOM RECOVERY

       Core OOM Recovery File

       /Users/lons7862/workspace/llmmll/llmmllab-runner/utils/intelligent_oom_recovery.py
       - Defines IntelligentOOMRecovery class (1488 lines)
       - Uses sklearn.linear_model.Ridge, sklearn.preprocessing.StandardScaler, sklearn.metrics.mean_squared_error
       - ML model training: _train_models() trains Ridge regression for n_ctx, n_batch, n_ubatch, n_gpu_layers
       - predict_optimal_parameters_from_profile() -- primary prediction method
       - _apply_ml_optimization() -- applies ML predictions to base config
       - execute_recovery_strategy() -- structured retry: clear memory -> reduce batch -> move to CPU -> reduce context
       - record_success() / record_failure() -- records training data
       - optimize_parameters_for_hardware() -- binary search / exponential backoff / conservative increment parameter optimization
       - test_memory_preallocation() -- async GPU memory allocation test
       - estimate_memory_requirements() -- delegates to Resizer
       - Persistent training data: loads/saves JSON to /app/oom_recovery_data/
       - Imports from utils.resizer (Resizer, MemoryBreakdown) and utils.hardware_manager (hardware_manager, EnhancedHardwareManager)

       Generated Model Files (OOM/ML-specific)

       All of these are auto-generated Pydantic models used exclusively or primarily by the OOM recovery system:

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/optimal_parameters.py
       - OptimalParameters -- n_ctx, n_batch, n_ubatch, n_gpu_layers. Also used by Resizer, imgtxt2txt/qwen3_vl.py, and pipeline_cache.py

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/prediction_features.py
       - PredictionFeatures -- ML prediction input features. Used only by IntelligentOOMRecovery

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/learned_limits.py
       - LearnedLimits -- learned max limits from successful configs. Used only by IntelligentOOMRecovery

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/recovery_strategy.py
       - RecoveryStrategy -- recovery result with parameters and strategy name. Used only by IntelligentOOMRecovery

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/ml_model_performance.py
       - MLModelPerformance -- MSE metrics for ML model predictions. Used only by IntelligentOOMRecovery

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/oom_recovery_attempt_data.py
       - OOMRecoveryAttemptData -- single recovery attempt data. Used only by IntelligentOOMRecovery

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/model_configuration_data.py
       - ModelConfigurationData -- config parameters and results for model init. Used only by IntelligentOOMRecovery

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/parameter_optimization_config.py
       - ParameterOptimizationConfig -- optimization config with parameters and crash prevention. Used by UserConfig, ModelProfile, IntelligentOOMRecovery, default_configs.py, config_utils.py,
       _protobuf_translators_local.py

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/crash_prevention.py
       - CrashPrevention -- preallocation test, memory buffer, timeout, graceful degradation. Used by parameter_optimization_config.py, IntelligentOOMRecovery, UserConfig, ModelProfile, default_configs.py,
       _protobuf_translators_local.py

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/performance_parameter.py
       - PerformanceParameter -- single parameter optimization config. Used by parameter_optimization_config.py, IntelligentOOMRecovery, UserConfig, ModelProfile, default_configs.py,
       _protobuf_translators_local.py

       /Users/lons7862/workspace/llmmll/llmmllab-runner/models/parameter_tuning_strategy.py
       - ParameterTuningStrategy enum -- binary_search, exponential_backoff, conservative_increment. Used by performance_parameter.py, _protobuf_translators_local.py, UserConfig, ModelProfile,
       default_configs.py

       ---
       3. pipelines/external/

       /Users/lons7862/workspace/llmmll/llmmllab-runner/pipelines/external/__init__.py
       - Contains only a docstring noting that external pipelines (OpenAI, Anthropic) have been moved to pipelines/remote/
       - Exports nothing (__all__: list[str] = [])
       - No other files reference or import from this directory
       - This is effectively a dead/empty package

       ---
       4. pipelines/imgtxt2txt/

       /Users/lons7862/workspace/llmmll/llmmllab-runner/pipelines/imgtxt2txt/qwen3_vl.py
       - Defines Qwen3VLPipeline(BaseLlamaCppPipeline) for multimodal text+image generation using Qwen3 VL models
       - Imports BaseLlamaCppPipeline from pipelines.llamacpp, but this class is NOT exported from pipelines/llamacpp/__init__.py (only ChatLlamaCppPipeline and EmbedLlamaCppPipeline are exported), so this
       import will fail at runtime
       - Uses Qwen25VLChatHandler from llama_cpp.llama_chat_format for CLIP model handling
       - Imports OptimalParameters from models
       - No __init__.py exists in this directory
       - No other file in the codebase imports from pipelines.imgtxt2txt or references Qwen3VLPipeline
       - The PipelineFactory.create_pipeline() does not route to any imgtxt2txt pipeline

       ---
       5. ALL REFERENCES TO REMOVED CODE (files that need updating)

       Files that IMPORT from utils/intelligent_oom_recovery.py

       ┌────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────┐
       │                                          File                                          │    What it imports     │
       ├────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/pipelines/pipeline_cache.py (line 33) │ IntelligentOOMRecovery │
       └────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────┘

       Files that IMPORT from utils/resizer.py

       ┌──────────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────────┐
       │                                             File                                             │     What it imports      │
       ├──────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/utils/intelligent_oom_recovery.py (line 45) │ MemoryBreakdown, Resizer │
       ├──────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/pipelines/pipeline_cache.py (line 32)       │ Resizer                  │
       └──────────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────┘

       Files that IMPORT from utils/hardware_manager.py

       ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────┐
       │                                                 File                                                 │              What it imports              │
       ├──────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/utils/__init__.py (line 6)                          │ hardware_manager                          │
       ├──────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/utils/intelligent_oom_recovery.py (lines 42, 44)    │ hardware_manager, EnhancedHardwareManager │
       ├──────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/pipelines/pipeline_cache.py (line 31)               │ hardware_manager                          │
       ├──────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/test/integration/test_runner.py (lines 19, 52, 104) │ EnhancedHardwareManager, hardware_manager │
       └──────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────┘

       Files that reference OOM/ML-related Pydantic models

       ┌──────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┐
       │                                           File                                           │                                                  Models referenced
              │
       ├──────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┤
       │                                                                                          │ Imports and re-exports ALL of them: OptimalParameters, PredictionFeatures, LearnedLimits, RecoveryStrategy,
              │
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/models/__init__.py                      │ MLModelPerformance, OOMRecoveryAttemptData, ModelConfigurationData, ParameterOptimizationConfig,
       CrashPrevention,   │
       │                                                                                          │ PerformanceParameter, ParameterTuningStrategy, SystemGPUStats
              │
       ├──────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/models/user_config.py                   │ CrashPrevention, ParameterOptimizationConfig, ParameterTuningStrategy, PerformanceParameter --
              │
       │                                                                                          │ UserConfig.parameter_optimization field is type ParameterOptimizationConfig
              │
       ├──────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/models/model_profile.py                 │ CrashPrevention, ParameterOptimizationConfig, ParameterTuningStrategy, PerformanceParameter --
              │
       │                                                                                          │ ModelProfile.parameter_optimization field is type Optional[ParameterOptimizationConfig]
              │
       ├──────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/models/default_configs.py               │ ParameterOptimizationConfig, PerformanceParameter, ParameterTuningStrategy, CrashPrevention -- defines
              │
       │                                                                                          │ DEFAULT_PARAMETER_OPTIMIZATION_CONFIG
              │
       ├──────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/models/config_utils.py                  │ ParameterOptimizationConfig -- resolve_parameter_optimization_config() function
              │
       ├──────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/models/_protobuf_translators_local.py   │ ParameterOptimizationConfig, ParameterTuningStrategy, PerformanceParameter, CrashPrevention -- protobuf
       translation │
       │                                                                                          │  functions for all of these
              │
       ├──────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/models/recovery_strategy.py             │ OptimalParameters (used as a field type)
              │
       ├──────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/models/parameter_optimization_config.py │ CrashPrevention, ParameterTuningStrategy, PerformanceParameter
              │
       ├──────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/models/performance_parameter.py         │ ParameterTuningStrategy
              │
       └──────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┘

       Files that reference OptimalParameters specifically

       This model is more broadly used beyond just OOM recovery:

       ┌───────────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┐
       │                                               File                                                │                           Usage                            │
       ├───────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/utils/resizer.py (line 3, 228)                   │ Import and parameter type for calculate_memory_breakdown() │
       ├───────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/utils/intelligent_oom_recovery.py                │ Pervasive use as return types and parameter types          │
       ├───────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/pipelines/pipeline_cache.py (lines 26, 577, 589) │ Import, _convert_model_parameters_to_optimal() returns it  │
       ├───────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/pipelines/imgtxt2txt/qwen3_vl.py (line 16, 57)   │ Import and _initialize_llama() parameter type              │
       └───────────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────┘

       Files that use hardware_manager singleton extensively

       ┌──────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┐
       │                                     File                                     │                                                              Usage
              │
       ├──────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/pipelines/pipeline_cache.py │ hardware_manager.update_all_memory_stats(), .check_memory_available(), .clear_memory(), .get_gpu_process_info(),
       .gpu_count,    │
       │                                                                              │ .has_gpu, .check_gpu_thermals() -- used throughout the entire cache management and eviction logic
              │
       └──────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┘

       Generated protobuf files that reference related models

       ┌──────────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
       │                                             File                                             │                          Notes                           │
       ├──────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/gen/python/runner/v1/runner_pb2.py          │ Contains ParameterOptimizationConfig protobuf descriptor │
       ├──────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/gen/python/runner/v1/composer_runner_pb2.py │ Contains ParameterOptimizationConfig protobuf descriptor │
       └──────────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────┘

       Test files

       ┌──────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┐
       │                                       File                                       │                                                        What it tests
              │
       ├──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┤
       │ /Users/lons7862/workspace/llmmll/llmmllab-runner/test/integration/test_runner.py │ Tests hardware_manager existence and attributes (check_memory_available, get_device_mappings,
       update_all_memory_stats,      │
       │                                                                                  │ get_gpu_process_info)
              │
       └──────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       ───────┘

       ---
       Summary: Removal Impact Matrix

       Files to DELETE entirely:
       - utils/intelligent_oom_recovery.py -- ML-based OOM recovery (entire file)
       - utils/resizer.py -- memory breakdown calculator (entire file)
       - pipelines/external/__init__.py -- empty dead package
       - pipelines/imgtxt2txt/qwen3_vl.py -- unused/broken vision pipeline
       - models/prediction_features.py -- used only by OOM recovery
       - models/learned_limits.py -- used only by OOM recovery
       - models/recovery_strategy.py -- used only by OOM recovery
       - models/ml_model_performance.py -- used only by OOM recovery
       - models/oom_recovery_attempt_data.py -- used only by OOM recovery
       - models/model_configuration_data.py -- used only by OOM recovery

       Files to MODIFY (heavy changes):
       - pipelines/pipeline_cache.py -- remove Resizer, IntelligentOOMRecovery, _ensure_memory(), estimate_memory(), _calculate_corrected_memory_breakdown(), _convert_model_parameters_to_optimal(), and all OOM
        recovery integration in get_or_create()
       - utils/hardware_manager.py -- decide what to keep (GPU detection, process management may still be needed)
       - utils/__init__.py -- update exports if hardware_manager changes

       Files to MODIFY (schema-driven, need schema changes first):
       - models/__init__.py -- remove exports for all deleted model files
       - models/user_config.py -- remove parameter_optimization field (requires schema change)
       - models/model_profile.py -- remove parameter_optimization field (requires schema change)
       - models/default_configs.py -- remove DEFAULT_PARAMETER_OPTIMIZATION_CONFIG and related imports
       - models/config_utils.py -- remove resolve_parameter_optimization_config() function
       - models/_protobuf_translators_local.py -- remove all ParameterOptimizationConfig, CrashPrevention, PerformanceParameter, ParameterTuningStrategy translation functions
       - models/parameter_optimization_config.py -- delete (requires schema change)
       - models/crash_prevention.py -- delete (requires schema change)
       - models/performance_parameter.py -- delete (requires schema change)
       - models/parameter_tuning_strategy.py -- delete (requires schema change)

       Files to MODIFY (light changes):
       - test/integration/test_runner.py -- update or remove hardware_manager tests depending on what is kept
       - models/optimal_parameters.py -- may be deletable if Resizer and pipeline_cache no longer need it; however it is also used by qwen3_vl.py which is being deleted

       Note on OptimalParameters and SystemGPUStats: Both are generated models. OptimalParameters is used by Resizer, IntelligentOOMRecovery, pipeline_cache, and imgtxt2txt/qwen3_vl.py. If all those consumers
       are removed, OptimalParameters can also be deleted. SystemGPUStats is used only by IntelligentOOMRecovery and can be deleted.
  ⎿  Done (50 tool uses · 136.6k tokens · 3m 55s)

                                                                                                                                                                                      03:35 PM claude-opus-4-6
⏺ Clear picture. Let me execute this systematically. The key changes are:

  1. Delete utils/intelligent_oom_recovery.py, utils/resizer.py, pipelines/external/, pipelines/imgtxt2txt/
  2. Simplify pipelines/pipeline_cache.py — strip memory estimation, OOM recovery, GPU-weighted eviction
  3. Clean up imports and requirements