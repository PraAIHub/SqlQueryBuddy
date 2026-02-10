"""RAG (Retrieval Augmented Generation) system for schema-aware SQL generation"""
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod


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

    def retrieve_context(self, user_query: str, top_k: int = 5) -> List[dict]:
        """Retrieve relevant schema context for a user query"""
        query_embedding = self.embedding_provider.embed(user_query)
        results = self.vector_db.search(query_embedding, top_k=top_k)

        context = []
        for key, metadata, similarity in results:
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

    def get_schema_context_string(self, user_query: str) -> str:
        """Get relevant schema as formatted context string"""
        context = self.retrieve_context(user_query)
        lines = ["Relevant Schema Elements:"]

        for item in context:
            if item["type"] == "table":
                lines.append(f"- Table: {item['name']}")
            elif item["type"] == "column":
                lines.append(f"  - Column: {item['name']} (Table: {item['table']})")

        return "\n".join(lines)
