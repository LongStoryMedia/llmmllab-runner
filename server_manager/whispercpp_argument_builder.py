"""CLI argument builder for whisper-server (whisper.cpp).

Translates model configuration into command-line arguments for the
whisper-server subprocess.  Follows the same pattern as
``LlamaCppArgumentBuilder`` and ``SDCppArgumentBuilder``.
"""

from __future__ import annotations

from typing import List, Optional

from config import WHISPER_SERVER_EXECUTABLE
from models import Model


class WhisperCppArgumentBuilder:
    """Builds CLI arguments for whisper-server from model config."""

    def __init__(self, executable: Optional[str] = None):
        self.executable = executable or WHISPER_SERVER_EXECUTABLE

    def build_args(self, model: Model, port: int) -> List[str]:
        """Build command-line arguments for whisper-server.

        whisper-server accepts:
          --model <path>       model file (required)
          --host <addr>        listen address
          --port <port>        listen port
          --threads <n>        thread count
          --language <lang>    forced language (optional)
        """
        model_path = getattr(model.details, "gguf_file", None)
        if not model_path:
            raise ValueError(
                f"Model '{model.id}' is missing details.gguf_file for whisper-server"
            )

        args = [
            self.executable,
            "--model", model_path,
            "--host", "0.0.0.0",
            "--port", str(port),
        ]

        # Optional: thread count from model.parameters
        if model.parameters:
            n_threads = getattr(model.parameters, "n_threads", None)
            if n_threads:
                args.extend(["--threads", str(n_threads)])

        return args
