# ============================================================
# File: speech_to_text.py
# Module 1: Speech-to-Text using faster-whisper (fallback: openai-whisper)
# ============================================================

import argparse
import os
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("speech_to_text")

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a"}

try:
    from faster_whisper import WhisperModel
    _BACKEND = "faster_whisper"
except ImportError:
    try:
        import whisper
        _BACKEND = "openai_whisper"
    except ImportError:
        _BACKEND = None


class SpeechToTextError(Exception):
    """Custom exception for STT failures."""
    pass


class SpeechToText:
    """
    Speech-to-Text engine wrapping faster-whisper or openai-whisper.
    Loads the model once and reuses it across calls.
    """

    def __init__(self, model_size: str = "base") -> None:
        if _BACKEND is None:
            raise SpeechToTextError(
                "No Whisper backend found. Install 'faster-whisper' or 'openai-whisper'."
            )
        self.backend = _BACKEND
        self.model_size = model_size
        self.model = self._load_model()
        logger.info("Loaded STT backend: %s (model=%s)", self.backend, model_size)

    def _load_model(self):
        try:
            if self.backend == "faster_whisper":
                # CPU-friendly default; switch to "cuda" if GPU available
                return WhisperModel(self.model_size, device="cpu", compute_type="int8")
            else:
                return whisper.load_model(self.model_size)
        except Exception as exc:
            raise SpeechToTextError(f"Failed to load Whisper model: {exc}") from exc

    @staticmethod
    def _validate_file(audio_path: str) -> None:
        if not os.path.isfile(audio_path):
            raise SpeechToTextError(f"Audio file not found: {audio_path}")
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise SpeechToTextError(
                f"Unsupported audio format '{ext}'. Supported: {SUPPORTED_FORMATS}"
            )
        if os.path.getsize(audio_path) == 0:
            raise SpeechToTextError(f"Audio file is empty: {audio_path}")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file and return transcript, language, confidence, duration.
        """
        self._validate_file(audio_path)

        try:
            if self.backend == "faster_whisper":
                segments, info = self.model.transcribe(audio_path, beam_size=5)
                segments = list(segments)
                transcript = " ".join(seg.text.strip() for seg in segments).strip()

                # Average log-prob -> rough confidence proxy (0-1)
                if segments:
                    avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
                    confidence = max(0.0, min(1.0, 1.0 + avg_logprob))
                else:
                    confidence = 0.0

                duration = info.duration if info.duration else 0.0
                language = info.language

            else:  # openai_whisper
                result = self.model.transcribe(audio_path)
                transcript = result.get("text", "").strip()
                language = result.get("language", "unknown")
                confidence = None  # openai-whisper doesn't expose this directly
                duration = self._estimate_duration(audio_path)

            if not transcript:
                logger.warning("Empty transcript produced for %s", audio_path)

            return {
                "transcript": transcript,
                "language": language,
                "confidence": round(confidence, 3) if confidence is not None else None,
                "duration": round(float(duration), 2),
            }

        except SpeechToTextError:
            raise
        except Exception as exc:
            logger.exception("Transcription failed for %s", audio_path)
            raise SpeechToTextError(f"Transcription failed: {exc}") from exc

    @staticmethod
    def _estimate_duration(audio_path: str) -> float:
        try:
            import librosa
            return librosa.get_duration(path=audio_path)
        except Exception:
            return 0.0


# Singleton-style convenience wrapper
_engine: Optional[SpeechToText] = None


def transcribe_audio(audio_path: str, model_size: str = "base") -> Dict[str, Any]:
    """
    Convenience function as required by spec.

    Args:
        audio_path: path to .wav/.mp3/.m4a file
        model_size: whisper model size (tiny/base/small/medium/large)

    Returns:
        dict with transcript, language, confidence, duration
    """
    global _engine
    try:
        if _engine is None or _engine.model_size != model_size:
            _engine = SpeechToText(model_size=model_size)
        return _engine.transcribe(audio_path)
    except SpeechToTextError as exc:
        logger.error(str(exc))
        return {"transcript": "", "language": "unknown", "confidence": None, "duration": 0.0, "error": str(exc)}


if __name__ == "__main__":
    # Sample usage
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with Whisper (faster-whisper or openai-whisper)."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default="sample_audio.wav",
        help="Path to a .wav, .mp3, or .m4a audio file.",
    )
    parser.add_argument(
        "--model-size",
        default="base",
        help="Whisper model size (tiny/base/small/medium/large).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.audio_path):
        logger.error("Audio file not found: %s", args.audio_path)
        print(f"Audio file not found: {args.audio_path}")
        print("Usage: python speech_to_text.py <audio_path> [--model-size base]")
    else:
        output = transcribe_audio(args.audio_path, model_size=args.model_size)
        print(output)