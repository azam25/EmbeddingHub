from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import datetime

class StorageBackend(ABC):
    """Abstract storage backend interface for FAISS index and metadata persistence"""
    
    @abstractmethod
    def save_index(self, index: Any, path: str) -> bool:
        """Save FAISS index to storage"""
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> Any:
        """Load FAISS index from storage"""
        pass
    
    @abstractmethod
    def save_metadata(self, metadata: Dict[str, Any], path: str) -> bool:
        """Save metadata to storage"""
        pass
    
    @abstractmethod
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """Load metadata from storage"""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if storage path exists"""
        pass
    
    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete storage path"""
        pass

class FileSystemBackend(StorageBackend):
    """Default file system storage backend"""
    
    def save_index(self, index: Any, path: str) -> bool:
        """Save FAISS index to disk"""
        try:
            import faiss
            faiss.write_index(index, path)
            return True
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            return False
    
    def load_index(self, path: str) -> Any:
        """Load FAISS index from disk"""
        try:
            import faiss
            return faiss.read_index(path)
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None
    
    def save_metadata(self, metadata: Dict[str, Any], path: str) -> bool:
        """Save metadata to disk as pickle"""
        try:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(metadata, f)
            return True
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return False
    
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """Load metadata from disk"""
        try:
            import pickle
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}
    
    def exists(self, path: str) -> bool:
        """Check if file exists"""
        return Path(path).exists()
    
    def delete(self, path: str) -> bool:
        """Delete file"""
        try:
            Path(path).unlink()
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False

class MongoDBBackend(StorageBackend):
    """MongoDB storage backend for FAISS index and metadata"""
    
    def __init__(self, connection_string: str, database: str, collection: str):
        """
        Initialize MongoDB backend
        
        Args:
            connection_string: MongoDB connection string
            database: Database name
            collection: Collection name
        """
        self.connection_string = connection_string
        self.database = database
        self.collection = collection
        self._client = None
        self._db = None
        self._coll = None
    
    def _get_connection(self):
        """Get MongoDB connection"""
        if self._client is None:
            try:
                from pymongo import MongoClient
                self._client = MongoClient(self.connection_string)
                self._db = self._client[self.database]
                self._coll = self._db[self.collection]
            except ImportError:
                raise ImportError("pymongo is required for MongoDB backend. Install with: pip install pymongo")
            except Exception as e:
                raise Exception(f"Failed to connect to MongoDB: {e}")
        return self._coll
    
    def save_index(self, index: Any, path: str) -> bool:
        """Save FAISS index to MongoDB as binary data"""
        try:
            import faiss
            import io
            
            # Convert FAISS index to bytes
            index_bytes = io.BytesIO()
            faiss.write_index(index, index_bytes)
            index_bytes.seek(0)
            
            # Save to MongoDB
            coll = self._get_connection()
            doc = {
                "type": "faiss_index",
                "path": path,
                "data": index_bytes.getvalue(),
                "created_at": datetime.datetime.utcnow()
            }
            
            # Upsert based on path
            coll.replace_one({"path": path}, doc, upsert=True)
            return True
            
        except Exception as e:
            print(f"Error saving FAISS index to MongoDB: {e}")
            return False
    
    def load_index(self, path: str) -> Any:
        """Load FAISS index from MongoDB"""
        try:
            import faiss
            import io
            
            coll = self._get_connection()
            doc = coll.find_one({"type": "faiss_index", "path": path})
            
            if doc and "data" in doc:
                # Convert bytes back to FAISS index
                index_bytes = io.BytesIO(doc["data"])
                return faiss.read_index(index_bytes)
            return None
            
        except Exception as e:
            print(f"Error loading FAISS index from MongoDB: {e}")
            return None
    
    def save_metadata(self, metadata: Dict[str, Any], path: str) -> bool:
        """Save metadata to MongoDB"""
        try:
            coll = self._get_connection()
            doc = {
                "type": "metadata",
                "path": path,
                "data": metadata,
                "created_at": datetime.datetime.utcnow()
            }
            
            # Upsert based on path
            coll.replace_one({"path": path}, doc, upsert=True)
            return True
            
        except Exception as e:
            print(f"Error saving metadata to MongoDB: {e}")
            return False
    
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """Load metadata from MongoDB"""
        try:
            coll = self._get_connection()
            doc = coll.find_one({"type": "metadata", "path": path})
            
            if doc and "data" in doc:
                return doc["data"]
            return {}
            
        except Exception as e:
            print(f"Error loading metadata from MongoDB: {e}")
            return {}
    
    def exists(self, path: str) -> bool:
        """Check if document exists in MongoDB"""
        try:
            coll = self._get_connection()
            return coll.count_documents({"path": path}) > 0
        except Exception:
            return False
    
    def delete(self, path: str) -> bool:
        """Delete document from MongoDB"""
        try:
            coll = self._get_connection()
            result = coll.delete_many({"path": path})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting from MongoDB: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._coll = None

class PostgreSQLBackend(StorageBackend):
    """PostgreSQL storage backend for FAISS index and metadata"""
    
    def __init__(self, connection_string: str, table_prefix: str = "faiss_storage"):
        """
        Initialize PostgreSQL backend
        
        Args:
            connection_string: PostgreSQL connection string
            table_prefix: Prefix for table names
        """
        self.connection_string = connection_string
        self.table_prefix = table_prefix
        self._engine = None
        self._init_tables()
    
    def _init_tables(self):
        """Initialize database tables"""
        try:
            from sqlalchemy import create_engine, text
            
            self._engine = create_engine(self.connection_string)
            
            # Create tables if they don't exist
            with self._engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_prefix}_indexes (
                        path VARCHAR(500) PRIMARY KEY,
                        data BYTEA,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_prefix}_metadata (
                        path VARCHAR(500) PRIMARY KEY,
                        data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
                
        except ImportError:
            raise ImportError("sqlalchemy and psycopg2 are required for PostgreSQL backend. Install with: pip install sqlalchemy psycopg2-binary")
        except Exception as e:
            raise Exception(f"Failed to initialize PostgreSQL: {e}")
    
    def save_index(self, index: Any, path: str) -> bool:
        """Save FAISS index to PostgreSQL as binary data"""
        try:
            import faiss
            import io
            
            # Convert FAISS index to bytes
            index_bytes = io.BytesIO()
            faiss.write_index(index, index_bytes)
            index_bytes.seek(0)
            
            # Save to PostgreSQL
            with self._engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text(f"""
                    INSERT INTO {self.table_prefix}_indexes (path, data)
                    VALUES (:path, :data)
                    ON CONFLICT (path) DO UPDATE SET 
                        data = EXCLUDED.data,
                        created_at = CURRENT_TIMESTAMP
                """), {"path": path, "data": index_bytes.getvalue()})
                conn.commit()
            return True
            
        except Exception as e:
            print(f"Error saving FAISS index to PostgreSQL: {e}")
            return False
    
    def load_index(self, path: str) -> Any:
        """Load FAISS index from PostgreSQL"""
        try:
            import faiss
            import io
            
            with self._engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text(f"""
                    SELECT data FROM {self.table_prefix}_indexes 
                    WHERE path = :path
                """), {"path": path}).fetchone()
                
                if result and result[0]:
                    # Convert bytes back to FAISS index
                    index_bytes = io.BytesIO(result[0])
                    return faiss.read_index(index_bytes)
            return None
            
        except Exception as e:
            print(f"Error loading FAISS index from PostgreSQL: {e}")
            return None
    
    def save_metadata(self, metadata: Dict[str, Any], path: str) -> bool:
        """Save metadata to PostgreSQL as JSONB"""
        try:
            import json
            
            with self._engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text(f"""
                    INSERT INTO {self.table_prefix}_metadata (path, data)
                    VALUES (:path, :data)
                    ON CONFLICT (path) DO UPDATE SET 
                        data = EXCLUDED.data,
                        created_at = CURRENT_TIMESTAMP
                """), {"path": path, "data": json.dumps(metadata)})
                conn.commit()
            return True
            
        except Exception as e:
            print(f"Error saving metadata to PostgreSQL: {e}")
            return False
    
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """Load metadata from PostgreSQL"""
        try:
            with self._engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text(f"""
                    SELECT data FROM {self.table_prefix}_metadata 
                    WHERE path = :path
                """), {"path": path}).fetchone()
                
                if result and result[0]:
                    return json.loads(result[0])
            return {}
            
        except Exception as e:
            print(f"Error loading metadata from PostgreSQL: {e}")
            return {}
    
    def exists(self, path: str) -> bool:
        """Check if document exists in PostgreSQL"""
        try:
            with self._engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text(f"""
                    SELECT COUNT(*) FROM {self.table_prefix}_indexes 
                    WHERE path = :path
                """), {"path": path}).fetchone()
                return result[0] > 0
        except Exception:
            return False
    
    def delete(self, path: str) -> bool:
        """Delete document from PostgreSQL"""
        try:
            with self._engine.connect() as conn:
                from sqlalchemy import text
                result1 = conn.execute(text(f"""
                    DELETE FROM {self.table_prefix}_indexes WHERE path = :path
                """), {"path": path})
                
                result2 = conn.execute(text(f"""
                    DELETE FROM {self.table_prefix}_metadata WHERE path = :path
                """), {"path": path})
                
                conn.commit()
                return result1.rowcount > 0 or result2.rowcount > 0
                
        except Exception as e:
            print(f"Error deleting from PostgreSQL: {e}")
            return False
    
    def close(self):
        """Close PostgreSQL connection"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
