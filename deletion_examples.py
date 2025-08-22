#!/usr/bin/env python3
"""
Deletion Examples - Demonstrate deletion capabilities in EmbeddingHub

This script shows how to:
1. Delete specific sources from FAISS Vector DB
2. Delete FAISS index files
3. Delete chunks from different storage backends
4. Rebuild indexes after deletions
"""

import os
import time
from embeddings import EmbeddingModel
from retriever import DocRetriever
from storage_interface import FileSystemBackend, MongoDBBackend

def create_test_data(retriever, base_name: str):
    """Create test documents for demonstration"""
    test_docs = [
        f"This is document {base_name}_1 about artificial intelligence and machine learning.",
        f"This is document {base_name}_2 about natural language processing and computer vision.",
        f"This is document {base_name}_3 about deep learning algorithms and neural networks."
    ]
    
    for i, content in enumerate(test_docs, 1):
        filename = f"test_{base_name}_{i}.txt"
        with open(filename, "w") as f:
            f.write(content)
        
        retriever.add_source(
            file_path=filename,
            source_name=f"{base_name}_source_{i}",
            tags=[base_name, f"doc_{i}"]
        )
    
    return [f"test_{base_name}_{i}.txt" for i in range(1, 4)]

def cleanup_test_files(filenames):
    """Clean up test files"""
    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)

def example_delete_operations():
    """Demonstrate deletion operations"""
    print("🗑️  DELETION OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    # Create embedding model
    embedder = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        verbose_logging=False
    )
    
    # Create retriever
    retriever = DocRetriever(
        embedding_model=embedder,
        chunk_size=200,
        chunk_overlap=50,
        verbose_logging=False
    )
    
    # Create test data
    print("📝 Creating test documents...")
    test_files = create_test_data(retriever, "demo")
    
    print(f"✅ Created {len(retriever.sources)} sources")
    print(f"📊 Sources: {list(retriever.sources.keys())}")
    
    # Test retrieval before deletion
    print("\n🔍 Testing retrieval before deletion...")
    results = retriever.retrieve("artificial intelligence", k=3)
    print(f"✅ Retrieved {len(results)} results")
    
    # Save to disk
    print("\n💾 Saving to disk...")
    retriever.save_to_disk("deletion_demo_storage")
    
    # Example 1: Delete specific source
    print(f"\n{'='*60}")
    print("🧪 Example 1: Delete Specific Source")
    print(f"{'='*60}")
    
    source_to_delete = "demo_source_2"
    print(f"🗑️  Deleting source: {source_to_delete}")
    
    success = retriever.delete_source(source_to_delete)
    if success:
        print(f"✅ Source '{source_to_delete}' deleted")
        print(f"📊 Remaining sources: {list(retriever.sources.keys())}")
    
    # Test retrieval after deletion
    print("\n🔍 Testing retrieval after deletion...")
    results = retriever.retrieve("natural language", k=3)
    print(f"✅ Retrieved {len(results)} results")
    
    # Example 2: Rebuild index after deletion
    print(f"\n{'='*60}")
    print("🧪 Example 2: Rebuild Index After Deletion")
    print(f"{'='*60}")
    
    print("🔨 Rebuilding FAISS index...")
    success = retriever.rebuild_index()
    if success:
        print("✅ Index rebuilt successfully")
        
        # Test retrieval with rebuilt index
        print("\n🔍 Testing retrieval with rebuilt index...")
        results = retriever.retrieve("deep learning", k=3)
        print(f"✅ Retrieved {len(results)} results")
    
    # Example 3: Delete storage files
    print(f"\n{'='*60}")
    print("🧪 Example 3: Delete Storage Files")
    print(f"{'='*60}")
    
    print("🗑️  Deleting storage files...")
    success = retriever.delete_storage_files("deletion_demo_storage")
    if success:
        print("✅ Storage files deleted")
    
    # Example 4: Delete all sources
    print(f"\n{'='*60}")
    print("🧪 Example 4: Delete All Sources")
    print(f"{'='*60}")
    
    print("🗑️  Deleting all sources...")
    success = retriever.delete_all_sources()
    if success:
        print("✅ All sources deleted")
        print(f"📊 Sources count: {len(retriever.sources)}")
    
    # Cleanup test files
    cleanup_test_files(test_files)
    
    # Cleanup storage directory
    if os.path.exists("deletion_demo_storage"):
        import shutil
        shutil.rmtree("deletion_demo_storage")
        print("🧹 Cleaned up test storage directory")

def example_mongodb_deletion():
    """Demonstrate deletion with MongoDB backend"""
    print(f"\n{'='*60}")
    print("🧪 MongoDB Backend Deletion Example")
    print(f"{'='*60}")
    
    try:
        # Create MongoDB backend
        mongo_backend = MongoDBBackend(
            connection_string="mongodb://localhost:27017",
            database="embeddinghub",
            collection="deletion_demo"
        )
        
        # Create embedding model
        embedder = EmbeddingModel(
            model_type="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            verbose_logging=False
        )
        
        # Create retriever with MongoDB backend
        retriever = DocRetriever(
            embedding_model=embedder,
            chunk_size=200,
            chunk_overlap=50,
            verbose_logging=False,
            storage_backend=mongo_backend
        )
        
        # Create test data
        print("📝 Creating test documents for MongoDB...")
        test_files = create_test_data(retriever, "mongo")
        
        print(f"✅ Created {len(retriever.sources)} sources")
        
        # Save to MongoDB
        print("💾 Saving to MongoDB...")
        retriever.save_to_disk("mongo_demo_path")
        
        # Delete a source
        print("🗑️  Deleting source: mongo_source_1")
        retriever.delete_source("mongo_source_1")
        
        # Rebuild index
        print("🔨 Rebuilding index...")
        retriever.rebuild_index()
        
        # Save updated state
        print("💾 Saving updated state to MongoDB...")
        retriever.save_to_disk("mongo_demo_path")
        
        # Delete storage files from MongoDB
        print("🗑️  Deleting from MongoDB...")
        retriever.delete_storage_files("mongo_demo_path")
        
        # Cleanup
        cleanup_test_files(test_files)
        mongo_backend.close()
        
        print("✅ MongoDB deletion example completed")
        
    except ImportError as e:
        print(f"❌ MongoDB not available: {e}")
        print("Install with: pip install pymongo")
    except Exception as e:
        print(f"❌ MongoDB error: {e}")

def example_storage_management():
    """Demonstrate storage management operations"""
    print(f"\n{'='*60}")
    print("🧪 Storage Management Operations")
    print(f"{'='*60}")
    
    # Create embedding model
    embedder = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        verbose_logging=False
    )
    
    # Create retriever
    retriever = DocRetriever(
        embedding_model=embedder,
        chunk_size=200,
        chunk_overlap=50,
        verbose_logging=False
    )
    
    # Create test data
    print("📝 Creating test documents...")
    test_files = create_test_data(retriever, "storage")
    
    # Save to disk
    print("💾 Saving to disk...")
    retriever.save_to_disk("storage_management_demo")
    
    # Show storage info
    print("\n📊 Storage Information:")
    info = retriever.get_storage_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Load from disk
    print("\n📂 Loading from disk...")
    new_retriever = DocRetriever(
        embedding_model=embedder,
        verbose_logging=False
    )
    new_retriever.load_from_disk("storage_management_demo")
    
    # Show loaded info
    print("\n📊 Loaded Storage Information:")
    info = new_retriever.get_storage_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    cleanup_test_files(test_files)
    if os.path.exists("storage_management_demo"):
        import shutil
        shutil.rmtree("storage_management_demo")

def main():
    """Run all deletion examples"""
    print("🚀 EmbeddingHub Deletion Operations Examples")
    print("=" * 80)
    
    # Run examples
    example_delete_operations()
    example_mongodb_deletion()
    example_storage_management()
    
    print(f"\n{'='*80}")
    print("🎉 All deletion examples completed!")
    print("\n💡 Key Deletion Features:")
    print("   • delete_source() - Remove specific sources")
    print("   • delete_all_sources() - Clear all sources")
    print("   • delete_storage_files() - Remove from storage backend")
    print("   • rebuild_index() - Rebuild FAISS index after deletions")
    print("   • Works with all storage backends (File, MongoDB, PostgreSQL)")
    print("   • Proper cleanup and error handling")

if __name__ == "__main__":
    main()
