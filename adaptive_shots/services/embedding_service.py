from typing import Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
import numpy as np

# This will be used in the future for accepting different embedding providers
class EmbeddingProvider(Protocol):
    """Protocol defining the interface for embedding providers."""
    def encode(self, text: str) -> np.ndarray:
        """Encode text into a vector representation."""
        pass

class SentenceTransformerEmbedding:
    """Concrete implementation of embedding provider using SentenceTransformer."""
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def _lazy_load_model(self) -> None:
        """Lazy load the model when needed."""
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)

    def encode(self, text: str) -> np.ndarray:
        """Encode text into a vector representation."""
        self._lazy_load_model()
        return self._model.encode(text).astype(np.float32)
