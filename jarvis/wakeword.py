"""Wake word detection using openWakeWord."""

from __future__ import annotations

import logging

import numpy as np

from jarvis.errors import WakeWordError

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detects 'Hey Jarvis' wake word in audio chunks."""

    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5):
        self._model_name = model_name
        self._threshold = threshold
        self._model = None

    def load(self) -> None:
        """Download models if needed and load the wake word model."""
        try:
            import openwakeword
            from openwakeword.model import Model

            openwakeword.utils.download_models()
            self._model = Model(
                wakeword_models=[self._model_name],
                inference_framework="onnx",
            )
            logger.info(
                "Wake word model loaded: %s (threshold=%.2f)",
                self._model_name,
                self._threshold,
            )
        except Exception as e:
            raise WakeWordError(f"Failed to load wake word model: {e}") from e

    def process_chunk(self, chunk: np.ndarray) -> bool:
        """Feed an 80ms audio chunk (int16). Returns True if wake word detected."""
        if self._model is None:
            raise WakeWordError("Model not loaded. Call load() first.")

        prediction = self._model.predict(chunk)
        score = prediction.get(self._model_name, 0.0)

        if score >= self._threshold:
            logger.info("Wake word detected (score=%.3f)", score)
            self._model.reset()
            return True

        return False

    def reset(self) -> None:
        """Clear internal prediction buffers."""
        if self._model is not None:
            self._model.reset()
