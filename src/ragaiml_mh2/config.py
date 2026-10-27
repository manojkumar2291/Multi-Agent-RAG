"""Project settings. Change defaults here or override them in the .env file."""

from functools import lru_cache
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Hold model names, Chroma location, chunk sizes and upload limits.

    Created by get_settings() below, using defaults, .env and environment values.
    Used in api.py: get_service() passes it to RAGService in service.py.
    The service reads these values when creating models and processing uploads.
    Tests create their own Settings in test_rag.py to use a temporary database.
    """

    # BaseSettings reads environment variables and .env values automatically.
    # For example, OLLAMA_MODEL in .env overrides ollama_model below.
    # Ignore unrelated .env entries so older settings do not break the app.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama generates answers. The embedding model converts text into vectors.
    ollama_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    # A collection is a named group of documents inside the Chroma database.
    chroma_directory: Path = Path("vector_store")
    chroma_collection: str = "ragaiml_multiagent"
    # Chunk sizes count characters. Overlap repeats text between nearby chunks.
    # Field adds validation: ge means >= and le means <=.
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0)
    max_upload_mb: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_overlap(self):
        # Pydantic runs this check after reading all settings.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Cache means read the settings once and reuse the result on later calls.
    return Settings()
