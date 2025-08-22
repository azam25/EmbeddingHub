#!/usr/bin/env python3
"""
Simple Performance Logging Example

This script shows the basic performance logging features
for chunk processing and vector database operations.
"""

from embeddings import EmbeddingModel, DocRetriever

def simple_example():
    """Simple example with performance logging"""
    
    print("🚀 Simple Performance Logging Example")
    print("=" * 50)
    
    # Create embedding model
    model = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        cache_dir="simple_cache"
    )
    
    # Create retriever
    retriever = DocRetriever(
        embedding_model=model,
        chunk_size=300,
        chunk_overlap=50
    )
    
    # Create a simple text file
    with open("test_doc.txt", "w") as f:
        f.write("This is a test document about artificial intelligence. "
                "It contains information about machine learning, neural networks, "
                "and deep learning algorithms. The document discusses various "
                "approaches to AI development and their applications in real-world scenarios.")
    
    print("📝 Created test document: test_doc.txt")
    
    # Process document (this will show detailed logging)
    print("\n🔄 Processing document...")
    retriever.add_source(
        file_path="test_doc.txt",
        source_name="test_source",
        tags=["ai", "test"]
    )
    
    # Search (this will show search performance logging)
    print("\n🔍 Performing search...")
    results = retriever.retrieve("machine learning", k=2)
    
    # Show results
    print(f"\n📊 Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. Score: {result['score']:.3f}")
        print(f"      Text: {result['text']}")
    
    # Show cache stats
    print("\n📈 Cache Statistics:")
    stats = model.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    import os
    if os.path.exists("test_doc.txt"):
        os.remove("test_doc.txt")
    if os.path.exists("simple_cache"):
        import shutil
        shutil.rmtree("simple_cache")
    
    print("\n✅ Example completed!")

if __name__ == "__main__":
    try:
        simple_example()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have sentence-transformers installed:")
        print("pip install sentence-transformers")
