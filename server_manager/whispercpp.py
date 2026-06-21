"""Server manager for whisper-server (whisper.cpp).

Spawns and manages a whisper-server subprocess for audio transcription.
Follows the same BaseServerManager pattern as LlamaCppServerManager and
SDCppServerManager.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from config import WHISPER_SERVER_EXECUTABLE
from models import Model
from server_manager.base import BaseServerManager
from server_manager.whispercpp_argument_builder import WhisperCppArgumentBuilder


class WhisperCppServerManager(BaseServerManager):
    """Manages a whisper-server subprocess for audio transcription."""

    startup_timeout = 60  # Small model (466 MB), fast load

    def __init__(
        self,
        model: Model,
        session_id: Optional[str] = None,
        port: Optional[int] = None,
    ):
        super().__init__(
            model=model,
            session_id=session_id,
            port=port,
            startup_timeout=self.startup_timeout,
        )
        self._argument_builder = WhisperCppArgumentBuilder()

    # ------------------------------------------------------------------
    # BaseServerManager overrides
    # ------------------------------------------------------------------

    def _build_server_args(self) -> list[str]:
        """Build CLI arguments for whisper-server."""
        return self._argument_builder.build_args(self.model, self.port)

    def get_api_endpoint(self, path: str) -> str:
        """Map runner paths to whisper-server endpoints.

        whisper-server exposes its HTTP API at root (no /v1 prefix).
        Health check: GET /
        Transcription: POST /inference
        """
        # Health and metrics pass through directly
        if path in ("/health", "/metrics"):
            return path
        # All other paths map directly to whisper-server
        return path

    def _validate_context_size(self) -> bool:
        """Whisper has no concept of context window — always valid."""
        return True

    def _build_subprocess_env(self) -> Optional[dict]:
        """Optionally pin GPU via CUDA_VISIBLE_DEVICES."""
        if self.model.parameters:
            main_gpu = getattr(self.model.parameters, "main_gpu", None)
            if main_gpu is not None and main_gpu >= 0:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(main_gpu)
                return env
        return None

    def is_running(self) -> bool:
        """Check whisper-server health via GET /.

        whisper-server returns server info on GET / when healthy.
        Falls back to base class process check if HTTP fails.
        """
        try:
            resp = requests.get(
                f"{self.server_url}/",
                timeout=2,
            )
            return resp.status_code == 200
        except requests.RequestException:
            # Fall back to process check
            return super().is_running()
