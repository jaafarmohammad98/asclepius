"""The local embedding model the meaning index uses: free, small, and run on this machine.

BAAI's bge-small-en-v1.5 (MIT licence, 384 dimensions, about 64 MB), run with ONNX Runtime
through fastembed. It is downloaded once from Hugging Face into the data folder's `models/`
(`hypothesis_engine.home`) by `asclepius-library index`; from then on it is loaded from there and nothing goes over
the network. Searching never downloads it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hypothesis_engine.home import DATA

MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MODELS = DATA / "models"


class ModelMissing(Exception):
    """The model is not in the models folder and downloading it was not allowed."""


class LocalEmbedder:
    """bge-small-en-v1.5 from `models`. With `download`, fetches it first if it is not there."""

    name = MODEL

    def __init__(self, models: Path = DEFAULT_MODELS, download: bool = False) -> None:
        from fastembed import TextEmbedding  # loads ONNX Runtime: only when a model is needed

        try:
            self._model = TextEmbedding(MODEL, cache_dir=str(models), local_files_only=not download)
        except Exception as error:  # fastembed raises a bare ValueError or OSError
            if download:
                raise
            raise ModelMissing(
                f"the embedding model is not in {models}; "
                "run `asclepius-library index` once with the network on to download it"
            ) from error

    def passages(self, texts: Sequence[str]) -> np.ndarray:
        """One unit vector per text, as rows."""
        # small batches pad less: about 28 passages a second on an 11-core laptop CPU, whatever the thread count
        return np.array(list(self._model.passage_embed(list(texts), batch_size=8)), dtype=np.float32)

    def query(self, text: str) -> np.ndarray:
        """The query's unit vector (bge adds its search instruction to queries, not passages)."""
        (vector,) = self._model.query_embed(text)
        return np.asarray(vector, dtype=np.float32)


def model_downloaded(models: Path = DEFAULT_MODELS) -> bool:
    """Whether the model loads from `models` without the network."""
    try:
        LocalEmbedder(models)
    except ModelMissing:
        return False
    return True
