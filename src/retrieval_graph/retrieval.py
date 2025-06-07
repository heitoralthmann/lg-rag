"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Pinecone, MongoDB, and ChromaDB.

The retrievers support filtering results by user_id to ensure data isolation between users.
"""

import os
from contextlib import contextmanager
from typing import Generator

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from retrieval_graph.configuration import Configuration, IndexConfiguration

## Encoder constructors


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "cohere":
            from langchain_cohere import CohereEmbeddings

            return CohereEmbeddings(model=model)  # type: ignore
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_pinecone_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    search_kwargs = configuration.search_kwargs

    search_filter = search_kwargs.setdefault("filter", {})
    search_filter.update({"user_id": configuration.user_id})
    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_mongodb_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )
    search_kwargs = configuration.search_kwargs
    pre_filter = search_kwargs.setdefault("pre_filter", {})
    pre_filter["user_id"] = {"$eq": configuration.user_id}
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_chroma_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a ChromaDB instance."""
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    import chromadb

    # Create ChromaDB client - can be persistent or in-memory
    # Check if CHROMA_PERSIST_DIRECTORY is set for persistent storage
    persist_directory = os.environ.get("CHROMA_PERSIST_DIRECTORY")

    if persist_directory:
        # Persistent ChromaDB client
        chroma_client = chromadb.PersistentClient(path=persist_directory)
    else:
        # In-memory ChromaDB client
        chroma_client = chromadb.Client()

    # Collection name can be configured via environment variable
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "langchain_collection")

    # Initialize Chroma vector store
    vstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    # Create a custom retriever class to handle the where filter properly
    class ChromaRetrieverWrapper(VectorStoreRetriever):
        def __init__(self, vectorstore, user_id: str, search_kwargs: dict):
            # Don't pass where filter in search_kwargs to avoid conflicts
            clean_search_kwargs = {k: v for k, v in search_kwargs.items() if k != "where"}
            super().__init__(vectorstore=vectorstore, search_kwargs=clean_search_kwargs)
            # Store user_id and original_where as instance attributes
            self._user_id = user_id
            self._original_where = search_kwargs.get("where", {})

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list[Document]:
            # Build the where filter with user_id
            where_filter = self._original_where.copy()
            where_filter["user_id"] = {"$eq": self._user_id}

            # Use similarity_search directly with proper where filter
            return self.vectorstore.similarity_search(
                query=query,
                k=self.search_kwargs.get("k", 4),
                where=where_filter
            )

    retriever = ChromaRetrieverWrapper(
        vectorstore=vstore,
        user_id=configuration.user_id,
        search_kwargs=configuration.search_kwargs
    )

    yield retriever


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = IndexConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    user_id = configuration.user_id
    if not user_id:
        raise ValueError("Please provide a valid user_id in the configuration.")
    match configuration.retriever_provider:
        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "chroma":
            with make_chroma_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(Configuration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
