"""
Text embeddings for stylometric analysis.

Uses sentence transformers to create dense vector representations
of text that capture semantic and stylistic information.
"""

from typing import List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TextEmbedder:
    """
    Generate text embeddings using sentence transformers.

    These embeddings capture semantic meaning and can be used
    alongside traditional stylometric features.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the text embedder.

        Args:
            model_name: Sentence transformer model name
            device: Device to use ("cpu", "cuda", etc.)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Whether to L2-normalize embeddings

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )

        return np.array(embeddings)

    def embed_chunks(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 100,
        aggregation: str = "mean",
    ) -> np.ndarray:
        """
        Embed a long text by chunking and aggregating.

        Args:
            text: Long text to embed
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            aggregation: How to combine chunks ("mean", "max", "first")

        Returns:
            Aggregated embedding vector
        """
        if len(text) <= chunk_size:
            return self.embed(text)[0]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]

            if len(chunk.strip()) > 50:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text) - overlap:
                break

        if not chunks:
            return self.embed(text[:chunk_size])[0]

        chunk_embeddings = self.embed(chunks)

        if aggregation == "mean":
            return np.mean(chunk_embeddings, axis=0)
        elif aggregation == "max":
            return np.max(chunk_embeddings, axis=0)
        elif aggregation == "first":
            return chunk_embeddings[0]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray],
    ) -> float:
        """
        Calculate cosine similarity between two texts or embeddings.

        Args:
            text1: First text or embedding
            text2: Second text or embedding

        Returns:
            Cosine similarity score (0-1)
        """
        if isinstance(text1, str):
            emb1 = self.embed(text1)[0]
        else:
            emb1 = text1

        if isinstance(text2, str):
            emb2 = self.embed(text2)[0]
        else:
            emb2 = text2

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    def pairwise_similarity(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Calculate pairwise similarity matrix for a list of texts.

        Args:
            texts: List of texts

        Returns:
            Similarity matrix (n_texts, n_texts)
        """
        embeddings = self.embed(texts, normalize=True)

        similarity_matrix = np.dot(embeddings, embeddings.T)

        return similarity_matrix

    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Find most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        query_embedding = self.embed(query)[0]
        candidate_embeddings = self.embed(candidates, normalize=True)

        similarities = np.dot(candidate_embeddings, query_embedding)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.model.get_sentence_embedding_dimension()


class StyleEmbedder:
    """
    Specialized embedder that combines semantic embeddings with
    stylometric features for authorship attribution.
    """

    def __init__(
        self,
        text_embedder: Optional[TextEmbedder] = None,
        use_semantic: bool = True,
        semantic_weight: float = 0.3,
    ):
        """
        Initialize style embedder.

        Args:
            text_embedder: TextEmbedder instance
            use_semantic: Whether to include semantic embeddings
            semantic_weight: Weight for semantic component (0-1)
        """
        self.use_semantic = use_semantic and SENTENCE_TRANSFORMERS_AVAILABLE
        self.semantic_weight = semantic_weight

        if self.use_semantic:
            self.text_embedder = text_embedder or TextEmbedder()
        else:
            self.text_embedder = None

    def embed(
        self,
        text: str,
        stylometric_features: np.ndarray,
    ) -> np.ndarray:
        """
        Create combined style embedding.

        Args:
            text: Input text
            stylometric_features: Pre-extracted stylometric feature vector

        Returns:
            Combined embedding vector
        """
        style_weight = 1 - self.semantic_weight

        style_normalized = stylometric_features / (np.linalg.norm(stylometric_features) + 1e-8)
        style_component = style_normalized * style_weight

        if self.use_semantic and self.text_embedder:
            semantic_embedding = self.text_embedder.embed(text)[0]
            semantic_component = semantic_embedding * self.semantic_weight

            combined = np.concatenate([style_component, semantic_component])
        else:
            combined = style_component

        return combined

    def embed_batch(
        self,
        texts: List[str],
        stylometric_features: np.ndarray,
    ) -> np.ndarray:
        """
        Create combined embeddings for multiple texts.

        Args:
            texts: List of texts
            stylometric_features: Feature matrix (n_texts, n_features)

        Returns:
            Combined embedding matrix
        """
        style_weight = 1 - self.semantic_weight

        style_norms = np.linalg.norm(stylometric_features, axis=1, keepdims=True) + 1e-8
        style_normalized = stylometric_features / style_norms
        style_component = style_normalized * style_weight

        if self.use_semantic and self.text_embedder:
            semantic_embeddings = self.text_embedder.embed(texts)
            semantic_component = semantic_embeddings * self.semantic_weight

            combined = np.hstack([style_component, semantic_component])
        else:
            combined = style_component

        return combined
