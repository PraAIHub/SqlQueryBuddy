"""Configuration management for SQL Query Buddy"""
import os
from typing import Literal
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # OpenAI Configuration
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", alias="OPENAI_MODEL")

    # Database Configuration
    database_type: Literal["sqlite", "postgresql", "mysql"] = Field(
        default="sqlite", alias="DATABASE_TYPE"
    )
    database_url: str = Field(default="sqlite:///retail.db", alias="DATABASE_URL")

    # Vector Database Configuration
    vector_db_type: Literal["faiss", "chroma"] = Field(default="faiss", alias="VECTOR_DB_TYPE")

    # Embeddings Configuration
    embeddings_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDINGS_MODEL"
    )

    # Application Configuration
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_rows_return: int = Field(default=1000, alias="MAX_ROWS_RETURN")
    query_timeout_seconds: int = Field(default=30, alias="QUERY_TIMEOUT_SECONDS")

    # RAG Configuration
    top_k_similar: int = Field(default=5, alias="TOP_K_SIMILAR")
    similarity_threshold: float = Field(default=0.5, alias="SIMILARITY_THRESHOLD")

    # FastAPI Configuration
    fastapi_host: str = Field(default="0.0.0.0", alias="FASTAPI_HOST")
    fastapi_port: int = Field(default=8000, alias="FASTAPI_PORT")

    # Gradio Configuration
    gradio_share: bool = Field(default=False, alias="GRADIO_SHARE")
    gradio_server_port: int = Field(default=7860, alias="GRADIO_SERVER_PORT")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Load settings from environment
settings = Settings()
