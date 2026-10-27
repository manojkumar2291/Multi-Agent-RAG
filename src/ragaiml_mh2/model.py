"""Create the models and the Chroma store used by the RAG service.

Importing this file does not create models or download anything.
RAGService.__init__() in service.py calls these functions when needed.
"""

from .config import Settings


def create_chat_model(settings: Settings):
    """Create the Ollama model shared by the planner, writer and reviewer.

    Called by RAGService.__init__() when no model was supplied by a test.
    """
    from langchain_ollama import ChatOllama

    # This model generates text. Its name and server URL come from config.py/.env.
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    return model


def create_embedding_model(settings: Settings):
    """Create the model that converts document text and queries into vectors.

    Called by create_vector_store() below. The first use may download the model.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return embedding_model


def create_vector_store(settings: Settings):
    """Open Chroma with the embedding model it needs to save and search text.

    Called by RAGService.__init__() when no test database was supplied.
    Chroma is a database, not a language model; it uses the embedding model.
    """
    from langchain_chroma import Chroma

    embedding_model = create_embedding_model(settings)
    vector_store = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embedding_model,
        # Saving on disk keeps the chunks available after a server restart.
        persist_directory=str(settings.chroma_directory),
    )
    return vector_store
