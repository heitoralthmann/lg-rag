#!/usr/bin/env python3
"""
Example script demonstrating how to use ChromaDB with the LangGraph RAG application.

This script shows how to:
1. Configure the application to use ChromaDB
2. Index some sample documents
3. Query the indexed documents

Before running this script, make sure to:
1. Install the dependencies: pip install -e .
2. Set your OpenAI API key: export OPENAI_API_KEY=your_key_here
3. Optionally set ChromaDB persistence directory: export CHROMA_PERSIST_DIRECTORY=/path/to/chroma/db
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from retrieval_graph import graph, index_graph

# Load environment variables
load_dotenv()

async def main():
    # Configuration for ChromaDB
    config = {
        "configurable": {
            "user_id": "example_user",
            "retriever_provider": "chroma",
            "embedding_model": "openai/text-embedding-3-small",
            "response_model": "openai/gpt-3.5-turbo",
            "query_model": "openai/gpt-3.5-turbo"
        }
    }

    # Sample documents to index
    documents = [
        Document(
            page_content="ChromaDB is an open-source AI application database. It provides vector search, document storage, and metadata filtering capabilities.",
            metadata={"source": "chroma_docs", "topic": "database"}
        ),
        Document(
            page_content="LangGraph is a framework for building stateful, multi-actor applications with LLMs. It allows you to create complex workflows and decision trees.",
            metadata={"source": "langgraph_docs", "topic": "framework"}
        ),
        Document(
            page_content="RAG (Retrieval Augmented Generation) combines the power of retrieval systems with generative AI to provide more accurate and contextual responses.",
            metadata={"source": "rag_guide", "topic": "ai_technique"}
        )
    ]

    print("🚀 Starting ChromaDB RAG Example")
    print("=" * 50)

    # Step 1: Index the documents
    print("\n📚 Indexing documents...")
    try:
        # Index all documents at once
        result = await index_graph.ainvoke({
            "docs": documents
        }, config=config)
        print(f"✅ Indexed {len(documents)} documents successfully!")
        for i, doc in enumerate(documents):
            print(f"   - Document {i+1}: {doc.page_content[:50]}...")
    except Exception as e:
        print(f"❌ Error indexing documents: {e}")
        return

    # Step 2: Query the indexed documents
    print("\n🔍 Querying the documents...")

    queries = [
        "What is ChromaDB?",
        "Tell me about LangGraph",
        "How does RAG work?",
        "What are the benefits of vector databases?"
    ]

    for query in queries:
        print(f"\n❓ Query: {query}")
        try:
            result = await graph.ainvoke({
                "messages": [("user", query)]
            }, config=config)

            print(f"💬 Response: {result['messages'][-1].content}")

            # Show retrieved documents if available
            if result.get("retrieved_docs"):
                print("📖 Retrieved documents:")
                for doc in result["retrieved_docs"]:
                    print(f"   - {doc.page_content[:100]}...")

        except Exception as e:
            print(f"❌ Error querying: {e}")

    print("\n✨ ChromaDB RAG Example completed!")
    print("\n💡 Tips:")
    print("   - Set CHROMA_PERSIST_DIRECTORY to persist data between runs")
    print("   - Set CHROMA_COLLECTION_NAME to use a custom collection name")
    print("   - ChromaDB will use in-memory storage if no persist directory is set")

if __name__ == "__main__":
    asyncio.run(main())
