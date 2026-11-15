"""Optional embedding featurizers for `LinearDecisionModel` (needs the `encoders` extra)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

DEFAULT_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
}


def sentence_featurizer(name: str = "minilm", device: str = "cpu",
                        batch_size: int = 64) -> Callable[[Sequence[str]], np.ndarray]:
    """Return texts -> L2-normalised embeddings; model is loaded once, lazily."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DEFAULT_MODELS.get(name, name), device=device)

    def featurize(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), batch_size=batch_size,
                            normalize_embeddings=True, show_progress_bar=False)

    return featurize
