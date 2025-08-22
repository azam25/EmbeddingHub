#!/usr/bin/env python3
"""
Per-Chunk Timing Demo - Individual Chunk Performance Analysis

This script demonstrates the detailed per-chunk timing for:
1. Individual chunk embedding generation time
2. Individual chunk FAISS ingestion time
3. Performance analysis per chunk
"""

import time
import os
from embeddings import EmbeddingModel, DocRetriever

def create_test_documents():
    """Create test documents with varying chunk sizes"""
    
    documents = {
        "short_doc.txt": "This is a short document about AI.",
        "medium_doc.txt": "This is a medium length document about artificial intelligence and machine learning. It contains several sentences with more detailed information.",
        "long_doc.txt": "This is a much longer document about artificial intelligence, machine learning, and deep learning. It contains extensive information about neural networks, natural language processing, computer vision, and various AI applications. The document discusses different approaches to AI development and their practical implementations in real-world scenarios.",
        "mixed_doc.txt": "Short chunk. This is a medium length chunk with more content about AI and machine learning. This is a very long chunk that contains extensive information about artificial intelligence, neural networks, deep learning, natural language processing, computer vision, robotics, and various other AI applications and their practical implementations."
    }
    
    for filename, content in documents.items():
        with open(filename, 'w') as f:
            f.write(content)
        print(f"📝 Created: {filename} ({len(content)} chars)")
    
    return documents

def demo_per_chunk_timing():
    """Demonstrate per-chunk timing analysis"""
    
    print("🔍 PER-CHUNK TIMING DEMONSTRATION")
    print("=" * 60)
    
    # Create test documents
    print("\n1️⃣  Creating test documents...")
    documents = create_test_documents()
    
    # Create embedding model
    print("\n2️⃣  Creating embedding model...")
    model = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        cache_dir="per_chunk_cache",
        max_cache_size="50MB"
    )
    
    # Test different chunk sizes
    chunk_configs = [
        (100, 20),   # Small chunks
        (200, 50),   # Medium chunks
        (300, 100),  # Large chunks
    ]
    
    for chunk_size, overlap in chunk_configs:
        print(f"\n{'='*60}")
        print(f"📊 TESTING: Chunk size={chunk_size}, Overlap={overlap}")
        print(f"{'='*60}")
        
        # Create retriever with current config
        retriever = DocRetriever(
            embedding_model=model,
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        
        # Process each document
        for filename in documents.keys():
            print(f"\n🔄 Processing: {filename}")
            print("-" * 40)
            
            try:
                retriever.add_source(
                    file_path=filename,
                    source_name=f"source_{filename.split('.')[0]}_{chunk_size}",
                    tags=["demo", "per_chunk", f"size_{chunk_size}"],
                    save_dir=f"demo_save_{chunk_size}"
                )
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test search to show query performance
        print(f"\n🔍 Testing search performance...")
        results = retriever.retrieve("artificial intelligence", k=3)
        print(f"   📊 Found {len(results)} results")
        
        # Cleanup for this config
        for filename in documents.keys():
            source_name = f"source_{filename.split('.')[0]}_{chunk_size}"
            if source_name in retriever.sources:
                del retriever.sources[source_name]
    
    # Show cache statistics
    print(f"\n📈 CACHE PERFORMANCE STATISTICS:")
    print("-" * 40)
    stats = model.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Performance analysis summary
    print(f"\n🎯 PER-CHUNK TIMING ANALYSIS SUMMARY:")
    print("-" * 40)
    print("✅ What gets measured:")
    print("   • Individual chunk embedding generation time")
    print("   • Individual chunk FAISS ingestion time")
    print("   • Per-chunk character count and processing details")
    print("   • Cache performance per chunk")
    print("   • Overall throughput and efficiency metrics")
    
    print("\n📊 Key Metrics:")
    print("   • Chunk processing time (text splitting)")
    print("   • Per-chunk embedding generation time")
    print("   • Per-chunk FAISS index ingestion time")
    print("   • Metadata preparation time")
    print("   • Storage and persistence time")
    
    # Cleanup
    print(f"\n🧹 Cleaning up...")
    for filename in documents.keys():
        if os.path.exists(filename):
            os.remove(filename)
            print(f"   🗑️  Deleted: {filename}")
    
    # Clean up cache and save directories
    cleanup_dirs = ["per_chunk_cache"]
    for chunk_size, _ in chunk_configs:
        cleanup_dirs.append(f"demo_save_{chunk_size}")
    
    for dir_name in cleanup_dirs:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)
            print(f"   🗑️  Deleted: {dir_name}")
    
    print(f"\n🎉 Per-chunk timing demo completed!")

def demo_individual_chunk_analysis():
    """Show detailed analysis of individual chunks"""
    
    print(f"\n🔬 INDIVIDUAL CHUNK ANALYSIS DEMO")
    print("=" * 60)
    
    # Create a single document with known content
    test_content = """This is the first chunk about artificial intelligence. It discusses the basics of AI and machine learning.

This is the second chunk that goes deeper into neural networks and deep learning algorithms. It contains more technical information.

This is the third chunk about natural language processing and computer vision applications. It covers practical use cases and implementations.

This is the fourth chunk discussing the future of AI and emerging technologies. It explores trends and predictions for the field."""
    
    with open("analysis_doc.txt", "w") as f:
        f.write(test_content)
    
    print("📝 Created analysis document with 4 distinct chunks")
    
    # Create model and retriever
    model = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        cache_dir="analysis_cache"
    )
    
    retriever = DocRetriever(
        embedding_model=model,
        chunk_size=150,  # Small chunks to see individual processing
        chunk_overlap=0   # No overlap for clear separation
    )
    
    # Process document (this will show detailed per-chunk timing)
    print("\n🔄 Processing document with per-chunk analysis...")
    retriever.add_source(
        file_path="analysis_doc.txt",
        source_name="analysis_source",
        tags=["analysis", "demo"]
    )
    
    # Test search
    print("\n🔍 Testing search...")
    results = retriever.retrieve("neural networks", k=2)
    
    # Cleanup
    if os.path.exists("analysis_doc.txt"):
        os.remove("analysis_doc.txt")
    if os.path.exists("analysis_cache"):
        import shutil
        shutil.rmtree("analysis_cache")
    
    print("✅ Individual chunk analysis completed!")

if __name__ == "__main__":
    try:
        # Main per-chunk timing demo
        demo_per_chunk_timing()
        
        # Individual chunk analysis demo
        demo_individual_chunk_analysis()
        
        print(f"\n🎯 Key Features Demonstrated:")
        print("1. Per-chunk embedding generation timing")
        print("2. Per-chunk FAISS ingestion timing")
        print("3. Individual chunk performance metrics")
        print("4. Cache performance per chunk")
        print("5. Overall system throughput analysis")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("\nMake sure you have the required dependencies:")
        print("pip install sentence-transformers faiss-cpu numpy")
