# ============================================================
# File: audio_features.py
# Module 3: Audio Feature Extraction & Scoring Engine
# ============================================================

import argparse
import os
import re
import logging
from typing import Dict, Any, List

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio_features")

try:
    import librosa
except ImportError as exc:
    raise ImportError("librosa is required. Install via: pip install librosa") from exc

FILLER_WORDS: List[str] = [
    "um", "uh", "like", "you know", "actually", "basically", "so"
]


class AudioFeatureError(Exception):
    """Custom exception for audio feature extraction failures."""
    pass


class AudioFeatureExtractor:
    """
    Extracts acoustic features from audio and combines them with
    transcript-based metrics to compute fluency and confidence scores.
    """

    def __init__(self, top_db: int = 30, frame_length: int = 2048, hop_length: int = 512) -> None:
        self.top_db = top_db
        self.frame_length = frame_length
        self.hop_length = hop_length

    @staticmethod
    def _validate_file(audio_path: str) -> None:
        if not os.path.isfile(audio_path):
            raise AudioFeatureError(f"Audio file not found: {audio_path}")
        if os.path.getsize(audio_path) == 0:
            raise AudioFeatureError(f"Audio file is empty: {audio_path}")

    def _load_audio(self, audio_path: str):
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            if y.size == 0:
                raise AudioFeatureError("Loaded audio signal is empty.")
            return y, sr
        except AudioFeatureError:
            raise
        except Exception as exc:
            raise AudioFeatureError(f"Failed to load audio: {exc}") from exc

    def _compute_silence_and_pauses(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect non-silent intervals to derive silence ratio and pause count."""
        intervals = librosa.effects.split(y, top_db=self.top_db,
                                           frame_length=self.frame_length,
                                           hop_length=self.hop_length)
        total_duration = len(y) / sr if sr else 0.0

        if len(intervals) == 0:
            return {
                "silence_ratio": 1.0,
                "pause_count": 0,
                "pause_ratio": 1.0,
                "voiced_duration": 0.0,
            }

        voiced_samples = sum((end - start) for start, end in intervals)
        voiced_duration = voiced_samples / sr
        silence_duration = max(total_duration - voiced_duration, 0.0)

        # Pauses = gaps between consecutive voiced intervals
        pause_count = max(len(intervals) - 1, 0)
        silence_ratio = round(silence_duration / total_duration, 3) if total_duration > 0 else 0.0
        pause_ratio = silence_ratio  # treat silence ratio as overall pause ratio

        return {
            "silence_ratio": silence_ratio,
            "pause_count": pause_count,
            "pause_ratio": pause_ratio,
            "voiced_duration": round(voiced_duration, 2),
        }

    @staticmethod
    def _count_filler_words(transcript: str) -> int:
        if not transcript:
            return 0
        text = transcript.lower()
        count = 0
        for filler in FILLER_WORDS:
            # word-boundary safe matching, handles multi-word fillers like "you know"
            pattern = r"\b" + re.escape(filler) + r"\b"
            count += len(re.findall(pattern, text))
        return count

    @staticmethod
    def _compute_speaking_rate(transcript: str, duration: float) -> float:
        if duration <= 0 or not transcript:
            return 0.0
        word_count = len(transcript.split())
        minutes = duration / 60.0
        return round(word_count / minutes, 2) if minutes > 0 else 0.0

    @staticmethod
    def _compute_fluency_score(pause_ratio: float, filler_count: int,
                                speaking_rate: float, word_count: int) -> float:
        """
        Heuristic fluency score (0-100):
        - Penalize high pause ratio
        - Penalize excessive filler words relative to word count
        - Reward speaking rate close to ideal range (110-160 wpm)
        """
        score = 100.0

        # Pause penalty
        score -= min(pause_ratio * 100, 40)

        # Filler word penalty (relative to length of speech)
        if word_count > 0:
            filler_density = filler_count / word_count
            score -= min(filler_density * 200, 30)
        else:
            score -= 30

        # Speaking rate penalty (ideal range 110-160 wpm)
        if speaking_rate > 0:
            if speaking_rate < 90:
                score -= 15
            elif speaking_rate > 190:
                score -= 15

        return round(max(0.0, min(100.0, score)), 2)

    @staticmethod
    def _compute_confidence_score(rms_energy: float, zcr: float, fluency_score: float) -> float:
        """
        Heuristic confidence score (0-100) combining vocal energy,
        zero-crossing rate (clarity proxy), and fluency.
        """
        # Normalize rms_energy roughly (typical range 0.0 - 0.5)
        energy_component = min(rms_energy / 0.3, 1.0) * 40

        # Lower ZCR variability generally indicates steadier voicing; reward moderate ZCR
        zcr_component = max(0.0, 1.0 - min(zcr / 0.2, 1.0)) * 20

        fluency_component = (fluency_score / 100.0) * 40

        score = energy_component + zcr_component + fluency_component
        return round(max(0.0, min(100.0, score)), 2)

    def analyze(self, audio_path: str, transcript: str) -> Dict[str, Any]:
        """
        Extract acoustic + transcript-based features and compute fluency/confidence scores.
        """
        self._validate_file(audio_path)

        try:
            y, sr = self._load_audio(audio_path)
            duration = round(len(y) / sr, 2) if sr else 0.0

            rms = float(np.mean(librosa.feature.rms(y=y, frame_length=self.frame_length,
                                                      hop_length=self.hop_length)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(
                y, frame_length=self.frame_length, hop_length=self.hop_length)))

            pause_info = self._compute_silence_and_pauses(y, sr)

            filler_count = self._count_filler_words(transcript)
            word_count = len(transcript.split()) if transcript else 0
            speaking_rate = self._compute_speaking_rate(transcript, duration)

            fluency_score = self._compute_fluency_score(
                pause_ratio=pause_info["pause_ratio"],
                filler_count=filler_count,
                speaking_rate=speaking_rate,
                word_count=word_count,
            )

            confidence_score = self._compute_confidence_score(rms, zcr, fluency_score)

            return {
                "duration": duration,
                "rms_energy": round(rms, 4),
                "zero_crossing_rate": round(zcr, 4),
                "silence_ratio": pause_info["silence_ratio"],
                "pause_count": pause_info["pause_count"],
                "pause_ratio": pause_info["pause_ratio"],
                "speaking_rate": speaking_rate,
                "filler_count": filler_count,
                "fluency_score": fluency_score,
                "confidence_score": confidence_score,
            }

        except AudioFeatureError:
            raise
        except Exception as exc:
            logger.exception("Audio feature extraction failed for %s", audio_path)
            raise AudioFeatureError(f"Audio feature extraction failed: {exc}") from exc


def analyze_audio(audio_path: str, transcript: str) -> Dict[str, Any]:
    """
    Convenience function as required by spec.

    Args:
        audio_path: path to audio file
        transcript: transcribed text from speech_to_text module

    Returns:
        dict with duration, pause_ratio, filler_count, rms_energy, speaking_rate, fluency_score, etc.
    """
    try:
        extractor = AudioFeatureExtractor()
        return extractor.analyze(audio_path, transcript)
    except AudioFeatureError as exc:
        logger.error(str(exc))
        return {
            "duration": 0.0, "pause_ratio": 0.0, "filler_count": 0,
            "rms_energy": 0.0, "speaking_rate": 0.0, "fluency_score": 0.0,
            "error": str(exc),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract audio fluency features from an audio file."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default="sample_audio.wav",
        help="Path to a .wav, .mp3, or .m4a audio file.",
    )
    parser.add_argument(
        "--transcript",
        default="So, um, photosynthesis is like, you know, the process where plants make food.",
        help="Transcript text corresponding to the audio file.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.audio_path):
        logger.error("Audio file not found: %s", args.audio_path)
        print(f"Audio file not found: {args.audio_path}")
        print("Usage: python audio_features.py <audio_path> [--transcript 'text']")
    else:
        result = analyze_audio(args.audio_path, args.transcript)
        print(result)