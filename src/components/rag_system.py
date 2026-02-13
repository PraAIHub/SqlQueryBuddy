"""RAG (Retrieval Augmented Generation) system for schema-aware SQL generation"""
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import math
import re
import logging
from collections import Counter

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers"""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed text into vector space"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        pass


class SimpleEmbeddingProvider(EmbeddingProvider):
    """Bag-of-words embedding provider that works without external APIs.

    Uses TF-IDF-like weighting over a fixed vocabulary built from schema text.
    Sufficient for semantic retrieval over small schema vocabularies.
    """

    def __init__(self):
        self.vocabulary: List[str] = []
        self.idf: Dict[str, float] = {}
        self._documents: List[str] = []

    def build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary and IDF weights from a corpus of schema descriptions."""
        self._documents = texts
        word_set = set()
        doc_freq: Dict[str, int] = Counter()

        for text in texts:
            words = self._tokenize(text)
            unique_words = set(words)
            word_set.update(unique_words)
            for w in unique_words:
                doc_freq[w] += 1

        self.vocabulary = sorted(word_set)
        num_docs = len(texts) + 1  # smoothing
        self.idf = {
            w: math.log(num_docs / (1 + doc_freq.get(w, 0)))
            for w in self.vocabulary
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenizer: lowercase, split on non-alpha, remove short tokens."""
        return [w for w in re.findall(r"[a-z][a-z0-9_]*", text.lower()) if len(w) > 1]

    def embed(self, text: str) -> List[float]:
        """Embed text into a TF-IDF weighted bag-of-words vector."""
        if not self.vocabulary:
            return []
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vector = []
        for word in self.vocabulary:
            tfidf = (tf.get(word, 0) / total) * self.idf.get(word, 0.0)
            vector.append(tfidf)
        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return [self.embed(text) for text in texts]


class SchemaEmbedder:
    """Embeds database schema for semantic retrieval"""

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.schema_embeddings: Dict[str, List[float]] = {}
        self.schema_metadata: Dict[str, dict] = {}

    def embed_schema(self, schema: dict) -> None:
        """Embed database schema (tables, columns, relationships)"""
        for table_name, table_info in schema.items():
            # Embed table description
            table_desc = f"Table {table_name}: {table_info.get('description', '')}"
            self.schema_embeddings[f"table:{table_name}"] = self.embedding_provider.embed(
                table_desc
            )
            self.schema_metadata[f"table:{table_name}"] = {
                "type": "table",
                "name": table_name,
                "description": table_info.get("description", ""),
            }

            # Embed columns
            for column_name, column_info in table_info.get("columns", {}).items():
                col_desc = f"Column {column_name} in {table_name}: {column_info.get('description', '')} ({column_info.get('type', '')})"
                key = f"column:{table_name}.{column_name}"
                self.schema_embeddings[key] = self.embedding_provider.embed(col_desc)
                self.schema_metadata[key] = {
                    "type": "column",
                    "table": table_name,
                    "name": column_name,
                    "column_type": column_info.get("type", ""),
                }

    def retrieve_relevant_schema(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[str, dict, float]]:
        """Retrieve most relevant schema elements for a query"""
        # Simplified similarity calculation
        results = []
        for key, embedding in self.schema_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            results.append((key, self.schema_metadata[key], similarity))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a**2 for a in vec1) ** 0.5
        magnitude2 = sum(b**2 for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


class VectorDatabase(ABC):
    """Abstract base for vector databases"""

    @abstractmethod
    def store(self, key: str, vector: List[float], metadata: dict) -> None:
        """Store a vector with metadata"""
        pass

    @abstractmethod
    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[str, dict, float]]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored vectors"""
        pass


class InMemoryVectorDB(VectorDatabase):
    """Simple in-memory vector database for MVP"""

    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, dict] = {}

    def store(self, key: str, vector: List[float], metadata: dict) -> None:
        """Store vector in memory"""
        self.vectors[key] = vector
        self.metadata[key] = metadata

    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[str, dict, float]]:
        """Search for similar vectors"""
        results = []
        for key, vector in self.vectors.items():
            similarity = SchemaEmbedder._cosine_similarity(query_vector, vector)
            results.append((key, self.metadata[key], similarity))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """Clear all vectors"""
        self.vectors.clear()
        self.metadata.clear()


class FAISSVectorDB(VectorDatabase):
    """FAISS-backed vector database for production-grade similarity search.

    Uses Facebook AI Similarity Search (FAISS) for efficient
    vector indexing and retrieval with inner product similarity.
    """

    def __init__(self, dimension: int = 0):
        self._dimension = dimension
        self._index: Optional[object] = None
        self._keys: List[str] = []
        self._metadata: List[dict] = []
        self._initialized = False

    def _ensure_index(self, dimension: int) -> None:
        """Lazily initialize the FAISS index when dimension is known."""
        if self._initialized:
            return
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu is required for FAISSVectorDB. Install with: pip install faiss-cpu"
            )
        self._dimension = dimension
        # Use IndexFlatIP (inner product) after L2-normalizing vectors = cosine similarity
        self._index = faiss.IndexFlatIP(dimension)
        self._initialized = True
        logger.info(f"FAISS index initialized with dimension {dimension}")

    @staticmethod
    def _normalize(vector: List[float]) -> np.ndarray:
        """L2-normalize a vector so inner product equals cosine similarity."""
        arr = np.array(vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def store(self, key: str, vector: List[float], metadata: dict) -> None:
        """Store a vector in the FAISS index."""
        if not vector:
            return
        self._ensure_index(len(vector))
        normalized = self._normalize(vector)
        self._index.add(normalized)
        self._keys.append(key)
        self._metadata.append(metadata)

    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[str, dict, float]]:
        """Search for similar vectors using FAISS."""
        if not self._initialized or self._index.ntotal == 0 or not query_vector:
            return []

        normalized_query = self._normalize(query_vector)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(normalized_query, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            results.append(
                (self._keys[idx], self._metadata[idx], float(distances[0][i]))
            )
        return results

    def clear(self) -> None:
        """Clear the FAISS index."""
        if self._initialized:
            self._index.reset()
        self._keys.clear()
        self._metadata.clear()


class RAGSystem:
    """Complete RAG pipeline for schema-aware SQL generation"""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_db: VectorDatabase):
        self.embedding_provider = embedding_provider
        self.vector_db = vector_db
        self.schema_embedder = SchemaEmbedder(embedding_provider)

    def initialize_schema(self, schema: dict) -> None:
        """Initialize RAG system with database schema"""
        self.schema_embedder.embed_schema(schema)

        # Store schema embeddings in vector database
        for key, embedding in self.schema_embedder.schema_embeddings.items():
            metadata = self.schema_embedder.schema_metadata[key]
            self.vector_db.store(key, embedding, metadata)

    def retrieve_context(
        self,
        user_query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[dict]:
        """Retrieve relevant schema context for a user query.

        Args:
            user_query: Natural language query.
            top_k: Maximum number of results.
            similarity_threshold: Minimum similarity score to include.
        """
        query_embedding = self.embedding_provider.embed(user_query)
        results = self.vector_db.search(query_embedding, top_k=top_k)

        context = []
        for key, metadata, similarity in results:
            if similarity < similarity_threshold:
                continue
            context.append(
                {
                    "key": key,
                    "type": metadata.get("type"),
                    "name": metadata.get("name"),
                    "table": metadata.get("table"),
                    "description": metadata.get("description", ""),
                    "similarity": similarity,
                }
            )

        return context

    def get_schema_context_string(
        self, user_query: str, similarity_threshold: float = 0.0
    ) -> str:
        """Get relevant schema as formatted context string"""
        context = self.retrieve_context(
            user_query, similarity_threshold=similarity_threshold
        )

        if not context:
            return (
                "No relevant schema found for this query. "
                "The question may not relate to the database. "
                "Available tables: customers, products, orders, order_items."
            )

        lines = ["Relevant Schema Elements:"]

        for item in context:
            if item["type"] == "table":
                lines.append(f"- Table: {item['name']}")
            elif item["type"] == "column":
                lines.append(f"  - Column: {item['name']} (Table: {item['table']})")

        return "\n".join(lines)
