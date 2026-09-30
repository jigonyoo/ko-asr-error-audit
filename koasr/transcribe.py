"""Deterministic Whisper transcription (faster-whisper, CPU int8).

language is pinned to Korean and temperature to 0 so the same audio gives the
same text on every run. Automatic language detection or sampling fallback would
make the baseline unreproducible.
"""
from __future__ import annotations

import numpy as np

DECODE = {
    "language": "ko",
    "task": "transcribe",
    "temperature": 0.0,
    "beam_size": 5,
    "condition_on_previous_text": False,
    "vad_filter": False,
}


class WhisperTranscriber:
    def __init__(self, model: str = "small", compute_type: str = "int8"):
        from faster_whisper import WhisperModel
        import faster_whisper

        self.model_name = model
        self.compute_type = compute_type
        self.backend_version = faster_whisper.__version__
        self._model = WhisperModel(model, device="cpu", compute_type=compute_type)

    def __call__(self, audio: np.ndarray) -> str:
        segments, _info = self._model.transcribe(audio, **DECODE)
        return " ".join(s.text.strip() for s in segments).strip()

    def describe(self) -> dict:
        return {
            "backend": "faster-whisper",
            "backend_version": self.backend_version,
            "model": self.model_name,
            "device": "cpu",
            "compute_type": self.compute_type,
            "decode": DECODE,
        }
