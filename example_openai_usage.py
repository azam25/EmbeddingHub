#!/usr/bin/env python3
"""
Example: Using OpenAI Embeddings with EmbeddingHub

This script demonstrates how to use OpenAI embedding models
with the EmbeddingHub project.
"""

import os
from embeddings import EmbeddingModel, DocRetriever

def example_openai_embeddings():
    """Example using OpenAI's official API"""
    
    # Option 1: Use OpenAI's official API
    openai_model = EmbeddingModel(
        model_type="openai",
        model_name="text-embedding-ada-002",  # OpenAI's embedding model
        api_key="your-openai-api-key-here"   # Replace with your actual API key
    )
    
    # Test the embedding
    texts = ["Hello world", "This is a test document", "AI embeddings are powerful"]
    embeddings = openai_model.embed(texts)
    print(f"OpenAI embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")
    
    return openai_model

def example_openai_compatible_server():
    """Example using an OpenAI-compatible custom server"""
    
    # Option 2: Use OpenAI-compatible custom server
    custom_model = EmbeddingModel(
        model_type="openai-compatible",
        model_name="your-custom-model-name",  # Your custom model name
        base_url="https://your-server.com/v1",  # Your custom server URL
        api_key="your-api-key-here"  # Your server's API key
    )
    
    # Test the embedding
    texts = ["Hello world", "This is a test document", "AI embeddings are powerful"]
    embeddings = custom_model.embed(texts)
    print(f"Custom server embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")
    
    return custom_model

def example_with_doc_retriever():
    """Example using OpenAI embeddings with document retrieval"""
    
    # Create OpenAI embedding model
    openai_model = EmbeddingModel(
        model_type="openai",
        model_name="text-embedding-ada-002",
        api_key="your-openai-api-key-here"
    )
    
    # Create document retriever with OpenAI embeddings
    retriever = DocRetriever(
        embedding_model=openai_model,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    print("Document retriever created with OpenAI embeddings!")
    print("You can now add documents and perform semantic search.")
    
    return retriever

if __name__ == "__main__":
    print("=== OpenAI Embeddings with EmbeddingHub ===\n")
    
    # Set your OpenAI API key (or use environment variable)
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    try:
        # Example 1: OpenAI official API
        print("1. Testing OpenAI official API...")
        openai_model = example_openai_embeddings()
        print("✅ OpenAI embeddings working!\n")
        
        # Example 2: OpenAI-compatible custom server
        print("2. Testing OpenAI-compatible custom server...")
        # Uncomment the line below and configure your custom server
        # custom_model = example_openai_compatible_server()
        print("ℹ️  Custom server example commented out - configure your server first\n")
        
        # Example 3: Document retriever with OpenAI
        print("3. Creating document retriever with OpenAI embeddings...")
        # Uncomment the line below and add your API key
        # retriever = example_with_doc_retriever()
        print("ℹ️  Document retriever example commented out - add your API key first\n")
        
        print("🎉 All examples completed successfully!")
        print("\nTo use with your own API key:")
        print("1. Set OPENAI_API_KEY environment variable, or")
        print("2. Pass api_key parameter directly to EmbeddingModel")
        print("3. For custom servers, use model_type='openai-compatible'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure to:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Set your OpenAI API key")
        print("3. Check your internet connection")
