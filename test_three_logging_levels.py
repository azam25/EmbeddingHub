#!/usr/bin/env python3
"""
Test Three Logging Levels - Demonstrate True, False, and "Block" logging

This script shows the three different logging levels:
1. True - Verbose logging (shows everything including per-chunk details)
2. False - Concise logging (shows summaries but hides per-chunk details)
3. "Block" - Silent logging (shows nothing at all)
"""

import time
import os
from embeddings import EmbeddingModel, DocRetriever

def test_three_logging_levels():
    """Test all three logging levels"""
    
    print("🔊 TESTING THREE LOGGING LEVELS")
    print("=" * 70)
    
    # Create test document
    test_content = """This is the first chunk about artificial intelligence and machine learning. It contains information about neural networks and deep learning algorithms.

This is the second chunk discussing natural language processing and computer vision applications. It covers practical use cases and implementations.

This is the third chunk about the future of AI and emerging technologies. It explores trends and predictions for the field of artificial intelligence.

This is the fourth chunk covering machine learning algorithms and their applications in real-world scenarios. It discusses various approaches and methodologies."""
    
    with open("test_doc.txt", "w") as f:
        f.write(test_content)
    
    print("📝 Created test document with 4 chunks")
    
    # Test 1: Verbose logging (True) - Shows everything
    print(f"\n{'='*70}")
    print("🧪 TEST 1: verbose_logging=True (VERBOSE - Shows everything)")
    print(f"{'='*70}")
    
    model1 = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        cache_dir="test_cache_verbose",
        verbose_logging=True
    )
    
    retriever1 = DocRetriever(
        embedding_model=model1,
        chunk_size=200,
        chunk_overlap=50,
        verbose_logging=True
    )
    
    print("🔄 Processing document with verbose logging...")
    retriever1.add_source(
        file_path="test_doc.txt",
        source_name="verbose_source",
        tags=["test", "verbose"]
    )
    
    print("\n🔍 Testing search with verbose logging...")
    results1 = retriever1.retrieve("artificial intelligence", k=2)
    
    # Test 2: Concise logging (False) - Shows summaries, hides per-chunk details
    print(f"\n{'='*70}")
    print("🧪 TEST 2: verbose_logging=False (CONCISE - Shows summaries, hides per-chunk)")
    print(f"{'='*70}")
    
    model2 = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        cache_dir="test_cache_concise",
        verbose_logging=False
    )
    
    retriever2 = DocRetriever(
        embedding_model=model2,
        chunk_size=200,
        chunk_overlap=50,
        verbose_logging=False
    )
    
    print("🔄 Processing document with concise logging...")
    retriever2.add_source(
        file_path="test_doc.txt",
        source_name="concise_source",
        tags=["test", "concise"]
    )
    
    print("\n🔍 Testing search with concise logging...")
    results2 = retriever2.retrieve("artificial intelligence", k=2)
    
    # Test 3: Silent logging ("Block") - Shows nothing at all
    print(f"\n{'='*70}")
    print("🧪 TEST 3: verbose_logging='Block' (SILENT - Shows nothing)")
    print(f"{'='*70}")
    
    model3 = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        cache_dir="test_cache_silent",
        verbose_logging="Block"
    )
    
    retriever3 = DocRetriever(
        embedding_model=model3,
        chunk_size=200,
        chunk_overlap=50,
        verbose_logging="Block"
    )
    
    print("🔄 Processing document with silent logging...")
    retriever3.add_source(
        file_path="test_doc.txt",
        source_name="silent_source",
        tags=["test", "silent"]
    )
    
    print("\n🔍 Testing search with silent logging...")
    results3 = retriever3.retrieve("artificial intelligence", k=2)
    
    # Test 4: Runtime switching between levels
    print(f"\n{'='*70}")
    print("🧪 TEST 4: Runtime Logging Control")
    print(f"{'='*70}")
    
    model4 = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        cache_dir="test_cache_runtime",
        verbose_logging="Block"  # Start silent
    )
    
    retriever4 = DocRetriever(
        embedding_model=model4,
        chunk_size=200,
        chunk_overlap=50,
        verbose_logging="Block"  # Start silent
    )
    
    print("🔄 Processing document with silent logging...")
    retriever4.add_source(
        file_path="test_doc.txt",
        source_name="runtime_source1",
        tags=["test", "runtime1"]
    )
    
    print("\n🔄 Switching to concise logging...")
    retriever4.set_logging(False)
    
    print("🔄 Processing another document with concise logging...")
    retriever4.add_source(
        file_path="test_doc.txt",
        source_name="runtime_source2",
        tags=["test", "runtime2"]
    )
    
    print("\n🔄 Switching to verbose logging...")
    retriever4.set_logging(True)
    
    print("🔄 Processing another document with verbose logging...")
    retriever4.add_source(
        file_path="test_doc.txt",
        source_name="runtime_source3",
        tags=["test", "runtime3"]
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 THREE LOGGING LEVELS SUMMARY")
    print(f"{'='*70}")
    
    print("✅ verbose_logging=True (VERBOSE):")
    print("   • Shows per-chunk embedding timing")
    print("   • Shows per-chunk FAISS ingestion timing")
    print("   • Shows detailed performance breakdowns")
    print("   • Shows individual text details")
    print("   • Shows comprehensive performance summaries")
    
    print("\n✅ verbose_logging=False (CONCISE):")
    print("   • Hides per-chunk timing details")
    print("   • Shows overall performance summaries")
    print("   • Shows completion messages")
    print("   • Shows throughput and efficiency metrics")
    print("   • Professional, clean output")
    
    print("\n✅ verbose_logging='Block' (SILENT):")
    print("   • Shows absolutely nothing")
    print("   • Complete silence during processing")
    print("   • Only shows completion message")
    print("   • Perfect for production automation")
    print("   • Zero console output")
    
    print("\n🎯 Key Differences:")
    print("   • True:     'Chunk 1: 0.0064s (241 chars)' + Full summaries")
    print("   • False:    '✅ Total embedding time: 0.0983s' + Summaries only")
    print("   • 'Block':  '✅ Added 4 chunks to source...' (nothing else)")
    
    print("\n💡 Use Cases:")
    print("   • True:     Development, debugging, performance analysis")
    print("   • False:    Production with monitoring, user-facing operations")
    print("   • 'Block':  Automated scripts, CI/CD, background processing")
    
    # Cleanup
    print(f"\n🧹 Cleaning up...")
    if os.path.exists("test_doc.txt"):
        os.remove("test_doc.txt")
    
    cleanup_dirs = ["test_cache_verbose", "test_cache_concise", "test_cache_silent", "test_cache_runtime"]
    for dir_name in cleanup_dirs:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)
            print(f"   🗑️  Deleted: {dir_name}")
    
    print(f"\n🎉 Three logging levels test completed!")

if __name__ == "__main__":
    try:
        test_three_logging_levels()
    except Exception as e:
        print(f"❌ Error during test: {e}")
        print("\nMake sure you have the required dependencies:")
        print("pip install sentence-transformers faiss-cpu numpy")
