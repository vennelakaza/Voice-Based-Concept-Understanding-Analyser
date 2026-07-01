# ============================================================
# File: semantic_engine.py
# Module 2: Semantic Understanding & Similarity Engine
# ============================================================

import logging
from typing import Dict, Any, Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_engine")

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError as exc:
    raise ImportError(
        "sentence-transformers is required. Install via: pip install sentence-transformers"
    ) from exc


class SemanticEvaluationError(Exception):
    """Custom exception for semantic evaluation failures."""
    pass


class SemanticEngine:
    """
    Wraps a SentenceTransformer model to compute semantic similarity
    between a reference concept and a user's spoken transcript.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Loaded SentenceTransformer model: %s", model_name)
        except Exception as exc:
            raise SemanticEvaluationError(f"Failed to load model '{model_name}': {exc}") from exc

    @staticmethod
    def _validate_inputs(reference_text: str, user_text: str) -> None:
        if not reference_text or not reference_text.strip():
            raise SemanticEvaluationError("Reference text is empty.")
        if not user_text or not user_text.strip():
            raise SemanticEvaluationError("User transcript is empty.")

    def _compute_similarity(self, reference_text: str, user_text: str) -> float:
        try:
            embeddings = self.model.encode(
                [reference_text, user_text], convert_to_tensor=True, normalize_embeddings=True
            )
            score = util.cos_sim(embeddings[0], embeddings[1]).item()
            return float(np.clip(score, -1.0, 1.0))
        except Exception as exc:
            raise SemanticEvaluationError(f"Embedding/similarity computation failed: {exc}") from exc

    @staticmethod
    def _classify_level(similarity_pct: float) -> str:
        if similarity_pct >= 80:
            return "Strong Understanding"
        elif similarity_pct >= 50:
            return "Moderate Understanding"
        else:
            return "Poor Understanding"

    @staticmethod
    def _generate_feedback(similarity_pct: float, level: str) -> str:
        if level == "Strong Understanding":
            return "The explanation covers most of the important concepts accurately."
        elif level == "Moderate Understanding":
            return (
                "The explanation captures some key ideas but misses important details. "
                "Consider elaborating further on the core concept."
            )
        else:
            return (
                "The explanation does not align well with the reference concept. "
                "Review the topic and try to include key terms and ideas."
            )

    def evaluate(self, reference_text: str, user_text: str) -> Dict[str, Any]:
        """
        Compare reference concept text with user transcript and score understanding.
        """
        self._validate_inputs(reference_text, user_text)
        cosine_score = self._compute_similarity(reference_text, user_text)

        # Convert cosine similarity (-1 to 1) to percentage (0 to 100)
        similarity_pct = round(((cosine_score + 1) / 2) * 100, 2)

        level = self._classify_level(similarity_pct)
        feedback = self._generate_feedback(similarity_pct, level)

        return {
            "similarity": similarity_pct,
            "level": level,
            "feedback": feedback,
        }


# Singleton-style convenience wrapper
_engine: Optional[SemanticEngine] = None


def evaluate_understanding(reference_text: str, user_text: str) -> Dict[str, Any]:
    """
    Convenience function as required by spec.

    Args:
        reference_text: ground-truth concept description
        user_text: user's spoken/transcribed explanation

    Returns:
        dict with similarity, level, feedback
    """
    global _engine
    try:
        if _engine is None:
            _engine = SemanticEngine()
        return _engine.evaluate(reference_text, user_text)
    except SemanticEvaluationError as exc:
        logger.error(str(exc))
        return {"similarity": 0.0, "level": "Error", "feedback": str(exc)}


if __name__ == "__main__":
    # Sample usage
    reference = "Photosynthesis is the process by which plants convert light energy into chemical energy."
    user = "Plants use sunlight to make food and produce oxygen through photosynthesis."
    result = evaluate_understanding(reference, user)
    print(result)