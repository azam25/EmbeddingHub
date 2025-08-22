#!/usr/bin/env python3
"""
Performance Logging Demo - Comprehensive Performance Tracking

This script demonstrates the detailed performance logging for:
1. Document chunking time
2. Embedding generation time per chunk
3. Vector database ingestion time
4. Search query performance
5. Cache performance metrics
"""

import time
import os
from embeddings import EmbeddingModel, DocRetriever

def demo_performance_logging():
    """Demonstrate comprehensive performance logging"""
    
    print("🚀 EMBEDDINGHUB PERFORMANCE LOGGING DEMO")
    print("=" * 60)
    
    # Create embedding model with detailed logging
    print("\n1️⃣  Creating embedding model...")
    model = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        cache_dir="demo_cache",
        max_cache_size="100MB",
        cache_policy="lru"
    )
    
    # Create document retriever
    print("\n2️⃣  Creating document retriever...")
    retriever = DocRetriever(
        embedding_model=model,
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Create sample documents for testing
    print("\n3️⃣  Creating sample documents...")
    sample_docs = {
        "sample1.txt": "This is a sample document about artificial intelligence and machine learning. It contains information about neural networks, deep learning, and natural language processing.",
        "sample2.txt": "Another document discussing data science and analytics. This covers topics like statistical analysis, data visualization, and predictive modeling.",
        "sample3.txt": "A third document about software engineering and development practices. Topics include agile methodologies, testing strategies, and deployment pipelines."
    }
    
    # Write sample documents to disk
    for filename, content in sample_docs.items():
        with open(filename, 'w') as f:
            f.write(content)
        print(f"   📝 Created: {filename}")
    
    # Process documents with detailed logging
    print("\n4️⃣  Processing documents with performance logging...")
    print("-" * 60)
    
    for filename in sample_docs.keys():
        try:
            retriever.add_source(
                file_path=filename,
                source_name=f"source_{filename.split('.')[0]}",
                tags=["demo", "sample", "ai"],
                save_dir="demo_save"
            )
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
    
    # Test search performance
    print("\n5️⃣  Testing search performance...")
    print("-" * 60)
    
    test_queries = [
        "artificial intelligence",
        "data science",
        "software engineering",
        "machine learning algorithms"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = retriever.retrieve(query, k=3)
        
        print(f"   📊 Top results:")
        for i, result in enumerate(results[:2], 1):
            print(f"      {i}. Score: {result['score']:.3f}")
            print(f"         Text: {result['text'][:80]}...")
            print(f"         Source: {result['metadata']['source']}")
    
    # Test cache performance
    print("\n6️⃣  Testing cache performance...")
    print("-" * 60)
    
    # Repeat queries to show cache hits
    print("   🔄 Repeating queries to demonstrate cache performance...")
    for query in test_queries[:2]:  # Test first 2 queries
        print(f"\n   🔍 Repeat query: '{query}'")
        results = retriever.retrieve(query, k=2)
    
    # Show cache statistics
    print("\n7️⃣  Cache Performance Statistics...")
    print("-" * 60)
    
    stats = model.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Performance summary
    print("\n8️⃣  Overall Performance Summary...")
    print("-" * 60)
    
    print("   ✅ Document Processing:")
    print("      - Chunking time tracked per document")
    print("      - Embedding generation time per chunk")
    print("      - Vector database build time")
    print("      - Storage and metadata time")
    
    print("\n   ✅ Search Performance:")
    print("      - Query embedding generation time")
    print("      - Vector search time per source")
    print("      - Result processing time")
    print("      - Overall search throughput")
    
    print("\n   ✅ Cache Performance:")
    print("      - Cache hit/miss rates")
    print("      - Cache lookup time")
    print("      - Cache save time")
    print("      - Memory usage and efficiency")
    
    # Cleanup
    print("\n9️⃣  Cleaning up...")
    for filename in sample_docs.keys():
        if os.path.exists(filename):
            os.remove(filename)
            print(f"   🗑️  Deleted: {filename}")
    
    if os.path.exists("demo_cache"):
        import shutil
        shutil.rmtree("demo_cache")
        print("   🗑️  Deleted: demo_cache")
    
    if os.path.exists("demo_save"):
        import shutil
        shutil.rmtree("demo_save")
        print("   🗑️  Deleted: demo_save")
    
    print("\n🎉 Performance logging demo completed!")
    print("\n📊 Key Metrics Tracked:")
    print("   • Chunk processing time per document")
    print("   • Embedding generation time per chunk")
    print("   • Vector database ingestion time")
    print("   • Search query performance")
    print("   • Cache hit rates and performance")
    print("   • Overall throughput and efficiency")

def demo_batch_processing():
    """Demonstrate batch processing performance"""
    
    print("\n🔄 BATCH PROCESSING PERFORMANCE DEMO")
    print("=" * 60)
    
    # Create model
    model = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        cache_dir="batch_cache"
    )
    
    # Test different batch sizes
    batch_sizes = [1, 5, 10, 20]
    
    for batch_size in batch_sizes:
        print(f"\n📦 Testing batch size: {batch_size}")
        
        # Create test texts
        texts = [f"This is test text number {i} for batch processing performance analysis." for i in range(batch_size)]
        
        # Time batch processing
        start_time = time.time()
        embeddings = model.embed(texts)
        total_time = time.time() - start_time
        
        print(f"   📊 Batch size: {batch_size}")
        print(f"   ⏱️  Total time: {total_time:.4f}s")
        print(f"   ⚡ Per text: {total_time/batch_size:.4f}s")
        print(f"   🚀 Throughput: {batch_size/total_time:.2f} texts/second")
        print(f"   📐 Embedding shape: {embeddings.shape}")
    
    # Cleanup
    if os.path.exists("batch_cache"):
        import shutil
        shutil.rmtree("batch_cache")

if __name__ == "__main__":
    try:
        # Main performance logging demo
        demo_performance_logging()
        
        # Batch processing demo
        demo_batch_processing()
        
        print("\n🎯 Performance Logging Features:")
        print("1. Real-time chunk processing timing")
        print("2. Per-chunk embedding generation metrics")
        print("3. Vector database operation timing")
        print("4. Search query performance breakdown")
        print("5. Cache performance statistics")
        print("6. Throughput and efficiency metrics")
        print("7. Detailed performance summaries")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("\nMake sure you have the required dependencies:")
        print("pip install sentence-transformers faiss-cpu numpy")
