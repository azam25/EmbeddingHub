#!/usr/bin/env python3
"""
Storage Backend Examples - Demonstrate different storage backends for FAISS persistence

This script shows how to use:
1. FileSystemBackend (default) - Local disk storage
2. MongoDBBackend - MongoDB storage
3. PostgreSQLBackend - PostgreSQL storage
4. Custom backends - How to extend the system
"""

from embeddings import EmbeddingModel
from retriever import DocRetriever
from storage_interface import FileSystemBackend, MongoDBBackend, PostgreSQLBackend

def example_filesystem_backend():
    """Example using the default file system backend"""
    print("🔧 Example 1: File System Backend (Default)")
    print("=" * 60)
    
    # Create embedding model
    embedder = EmbeddingModel(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        verbose_logging=False
    )
    
    # Create retriever with file system backend
    retriever = DocRetriever(
        embedding_model=embedder,
        chunk_size=500,
        chunk_overlap=100,
        verbose_logging=False,
        storage_backend=FileSystemBackend()  # Explicit file system backend
    )
    
    # Add some test data
    test_text = "This is a test document about artificial intelligence and machine learning."
    with open("test_doc.txt", "w") as f:
        f.write(test_text)
    
    retriever.add_source("test_doc.txt", "test_source", tags=["test", "ai"])
    
    # Save to disk
    print("💾 Saving to file system...")
    retriever.save_to_disk("storage_examples/filesystem_storage")
    
    # Load from disk
    print("📂 Loading from file system...")
    new_retriever = DocRetriever(
        embedding_model=embedder,
        storage_backend=FileSystemBackend()
    )
    new_retriever.load_from_disk("storage_examples/filesystem_storage")
    
    # Test retrieval
    results = new_retriever.retrieve("artificial intelligence", k=1)
    print(f"✅ Retrieved {len(results)} results")
    
    # Show storage info
    info = new_retriever.get_storage_info()
    print(f"📊 Storage Info: {info}")

def example_mongodb_backend():
    """Example using MongoDB backend"""
    print("\n🔧 Example 2: MongoDB Backend")
    print("=" * 60)
    
    try:
        # Create MongoDB backend
        mongo_backend = MongoDBBackend(
            connection_string="mongodb://localhost:27017",
            database="embeddinghub",
            collection="faiss_storage"
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
            chunk_size=500,
            chunk_overlap=100,
            verbose_logging=False,
            storage_backend=mongo_backend
        )
        
        # Add test data
        test_text = "This is another test document about natural language processing."
        with open("test_doc2.txt", "w") as f:
            f.write(test_text)
        
        retriever.add_source("test_doc2.txt", "mongo_source", tags=["test", "nlp"])
        
        # Save to MongoDB
        print("💾 Saving to MongoDB...")
        retriever.save_to_disk("mongo_storage_path")  # Path is used as document identifier
        
        # Load from MongoDB
        print("📂 Loading from MongoDB...")
        new_retriever = DocRetriever(
            embedding_model=embedder,
            storage_backend=mongo_backend
        )
        new_retriever.load_from_disk("mongo_storage_path")
        
        # Test retrieval
        results = new_retriever.retrieve("natural language", k=1)
        print(f"✅ Retrieved {len(results)} results")
        
        # Show storage info
        info = new_retriever.get_storage_info()
        print(f"📊 Storage Info: {info}")
        
        # Clean up MongoDB connection
        mongo_backend.close()
        
    except ImportError as e:
        print(f"❌ MongoDB not available: {e}")
        print("Install with: pip install pymongo")
    except Exception as e:
        print(f"❌ MongoDB error: {e}")

def example_postgresql_backend():
    """Example using PostgreSQL backend"""
    print("\n🔧 Example 3: PostgreSQL Backend")
    print("=" * 60)
    
    try:
        # Create PostgreSQL backend
        postgres_backend = PostgreSQLBackend(
            connection_string="postgresql://username:password@localhost:5432/embeddinghub",
            table_prefix="faiss_storage"
        )
        
        # Create embedding model
        embedder = EmbeddingModel(
            model_type="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            verbose_logging=False
        )
        
        # Create retriever with PostgreSQL backend
        retriever = DocRetriever(
            embedding_model=embedder,
            chunk_size=500,
            chunk_overlap=100,
            verbose_logging=False,
            storage_backend=postgres_backend
        )
        
        # Add test data
        test_text = "This is a PostgreSQL test document about database systems."
        with open("test_doc3.txt", "w") as f:
            f.write(test_text)
        
        retriever.add_source("test_doc3.txt", "postgres_source", tags=["test", "database"])
        
        # Save to PostgreSQL
        print("💾 Saving to PostgreSQL...")
        retriever.save_to_disk("postgres_storage_path")
        
        # Load from PostgreSQL
        print("📂 Loading from PostgreSQL...")
        new_retriever = DocRetriever(
            embedding_model=embedder,
            storage_backend=postgres_backend
        )
        new_retriever.load_from_disk("postgres_storage_path")
        
        # Test retrieval
        results = new_retriever.retrieve("database systems", k=1)
        print(f"✅ Retrieved {len(results)} results")
        
        # Show storage info
        info = new_retriever.get_storage_info()
        print(f"📊 Storage Info: {info}")
        
        # Clean up PostgreSQL connection
        postgres_backend.close()
        
    except ImportError as e:
        print(f"❌ PostgreSQL not available: {e}")
        print("Install with: pip install sqlalchemy psycopg2-binary")
    except Exception as e:
        print(f"❌ PostgreSQL error: {e}")

def example_custom_backend():
    """Example of creating a custom storage backend"""
    print("\n🔧 Example 4: Custom Storage Backend")
    print("=" * 60)
    
    from storage_interface import StorageBackend
    
    class RedisBackend(StorageBackend):
        """Custom Redis storage backend example"""
        
        def __init__(self, redis_url: str, prefix: str = "faiss:"):
            self.redis_url = redis_url
            self.prefix = prefix
            self._redis = None
        
        def _get_redis(self):
            """Get Redis connection"""
            if self._redis is None:
                try:
                    import redis
                    self._redis = redis.from_url(self.redis_url)
                except ImportError:
                    raise ImportError("redis is required. Install with: pip install redis")
            return self._redis
        
        def save_index(self, index, path: str) -> bool:
            """Save FAISS index to Redis"""
            try:
                import faiss
                import io
                
                # Convert index to bytes
                index_bytes = io.BytesIO()
                faiss.write_index(index, index_bytes)
                index_bytes.seek(0)
                
                # Save to Redis
                redis_client = self._get_redis()
                key = f"{self.prefix}index:{path}"
                redis_client.set(key, index_bytes.getvalue())
                return True
            except Exception as e:
                print(f"Redis save error: {e}")
                return False
        
        def load_index(self, path: str):
            """Load FAISS index from Redis"""
            try:
                import faiss
                import io
                
                redis_client = self._get_redis()
                key = f"{self.prefix}index:{path}"
                data = redis_client.get(key)
                
                if data:
                    index_bytes = io.BytesIO(data)
                    return faiss.read_index(index_bytes)
                return None
            except Exception as e:
                print(f"Redis load error: {e}")
                return None
        
        def save_metadata(self, metadata: dict, path: str) -> bool:
            """Save metadata to Redis as JSON"""
            try:
                import json
                redis_client = self._get_redis()
                key = f"{self.prefix}metadata:{path}"
                redis_client.set(key, json.dumps(metadata))
                return True
            except Exception as e:
                print(f"Redis metadata save error: {e}")
                return False
        
        def load_metadata(self, path: str) -> dict:
            """Load metadata from Redis"""
            try:
                import json
                redis_client = self._get_redis()
                key = f"{self.prefix}metadata:{path}"
                data = redis_client.get(key)
                return json.loads(data) if data else {}
            except Exception as e:
                print(f"Redis metadata load error: {e}")
                return {}
        
        def exists(self, path: str) -> bool:
            """Check if key exists in Redis"""
            try:
                redis_client = self._get_redis()
                key = f"{self.prefix}index:{path}"
                return redis_client.exists(key) > 0
            except Exception:
                return False
        
        def delete(self, path: str) -> bool:
            """Delete keys from Redis"""
            try:
                redis_client = self._get_redis()
                index_key = f"{self.prefix}index:{path}"
                metadata_key = f"{self.prefix}metadata:{path}"
                
                deleted = 0
                if redis_client.delete(index_key):
                    deleted += 1
                if redis_client.delete(metadata_key):
                    deleted += 1
                
                return deleted > 0
            except Exception as e:
                print(f"Redis delete error: {e}")
                return False
    
    print("✅ Custom Redis backend created!")
    print("📝 This shows how to extend the storage system with any backend you need")
    print("🔌 You can implement backends for:")
    print("   • AWS S3, Google Cloud Storage")
    print("   • Azure Blob Storage")
    print("   • Elasticsearch")
    print("   • Any other storage system")

def main():
    """Run all storage backend examples"""
    print("🚀 Storage Backend Examples for EmbeddingHub")
    print("=" * 80)
    
    # Run examples
    example_filesystem_backend()
    example_mongodb_backend()
    example_postgresql_backend()
    example_custom_backend()
    
    print("\n🎉 All examples completed!")
    print("\n💡 Key Benefits of the Storage Backend System:")
    print("   • Easy to switch between storage systems")
    print("   • Consistent API across all backends")
    print("   • Simple to extend with custom backends")
    print("   • Production-ready with proper error handling")
    print("   • Automatic cleanup and connection management")

if __name__ == "__main__":
    main()
