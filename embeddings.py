# embeddings.py
import hashlib
import numpy as np
import requests
import time
import os
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict
import pickle
import warnings

class EmbeddingModel:
    """Complete embedding model with support for local models, OpenAI, and custom servers with cache control"""
    
    def __init__(
        self,
        model_type: str = "sentence-transformers",
        model_name: str = "all-MiniLM-L6-v2",
        model_path: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: str = "embedding_cache",
        max_cache_size: str = "1GB",  # "500MB", "2GB", "1TB"
        cache_policy: str = "lru",    # "lru" or "fifo"
        verbose_logging: Union[bool, str] = True  # True=verbose, False=concise, "Block"=silent
    ):
        """
        Initialize with full cache control
        
        Args:
            model_type: "sentence-transformers", "lmstudio", "ollama", "vllm", "openai", "openai-compatible"
            model_name: Name of the model to use
            model_path: Path to local model (for sentence-transformers)
            base_url: Base URL for API-based models
            api_key: API key for authentication
            cache_dir: Directory for cache storage
            max_cache_size: Cache size limit (e.g., "500MB", "1GB", "2GB")
            cache_policy: Cleanup policy ("lru" or "fifo")
            verbose_logging: True=verbose, False=concise, "Block"=silent (default: True)
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.base_url = base_url.rstrip('/') if base_url else None
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_bytes = self._parse_size(max_cache_size)
        self.cache_policy = cache_policy.lower()
        
        # Track cache metadata and in-memory index for fast lookups
        self.cache_metadata = {}
        self.cache_index = set()  # Fast O(1) lookup for cache keys
        
        # Logging control
        self.verbose_logging = verbose_logging
        
        self._load_cache_metadata()
        
        if self.model_type == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        elif self.model_type in ["lmstudio", "ollama", "vllm"]:
            if not self.base_url:
                raise ValueError(f"{model_type} requires base_url")
        elif self.model_type == "openai":
            if not self.api_key:
                raise ValueError("OpenAI requires api_key")
            # Use OpenAI's default endpoint
            self.base_url = "https://api.openai.com/v1"
        elif self.model_type == "openai-compatible":
            if not self.base_url:
                raise ValueError("openai-compatible requires base_url")
            if not self.api_key:
                raise ValueError("openai-compatible requires api_key")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _parse_size(self, size_str: str) -> int:
        """Convert '1GB' to bytes"""
        size_str = size_str.upper()
        if "MB" in size_str:
            return int(float(size_str.replace("MB", "")) * 1024**2)
        elif "GB" in size_str:
            return int(float(size_str.replace("GB", "")) * 1024**3)
        elif "TB" in size_str:
            return int(float(size_str.replace("TB", "")) * 1024**4)
        raise ValueError("Size must be in MB/GB/TB")

    def _load_cache_metadata(self):
        """Initialize cache tracking with fast in-memory index"""
        for f in self.cache_dir.glob("*.npy"):
            stat = f.stat()
            self.cache_metadata[f] = {
                "size": stat.st_size,
                "last_used": stat.st_mtime,
                "created": stat.st_ctime
            }
            # Add to fast lookup index (extract key from filename)
            cache_key = f.stem  # filename without .npy extension
            self.cache_index.add(cache_key)

    def _enforce_cache_limit(self):
        """Automatically clean up old cache files"""
        current_size = sum(m["size"] for m in self.cache_metadata.values())
        if current_size <= self.max_bytes:
            return

        # Sort by policy (LRU or FIFO)
        sort_key = "last_used" if self.cache_policy == "lru" else "created"
        sorted_files = sorted(
            self.cache_metadata.items(),
            key=lambda x: x[1][sort_key]
        )

        # Delete oldest files
        deleted = 0
        for file_path, _ in sorted_files:
            if current_size <= self.max_bytes:
                break
            file_path.unlink()
            current_size -= self.cache_metadata[file_path]["size"]
            # Remove from in-memory index
            cache_key = file_path.stem
            self.cache_index.discard(cache_key)
            del self.cache_metadata[file_path]
            deleted += 1

        print(f"Cache cleanup: Deleted {deleted} files (New size: {self._format_bytes(current_size)})")

    def _format_bytes(self, bytes: int) -> str:
        """Convert bytes to human-readable format"""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes < 1024:
                return f"{bytes:.2f}{unit}"
            bytes /= 1024
        return f"{bytes:.2f}TB"

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings with automatic cache management"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Performance tracking
        total_start = time.time()
        num_texts = len(texts)
        
        cache_key = self._get_cache_key(texts)
        cache_path = self.cache_dir / f"{cache_key}.npy"
        
        # Time cache lookup
        cache_start = time.time()
        if cache_key in self.cache_index:
            # Fast O(1) lookup - no filesystem call needed!
            if cache_path.exists():  # Double-check file still exists
                self.cache_metadata[cache_path]["last_used"] = time.time()
                self._update_cache_stats(hit=True)  # Track cache hit
                
                # Time cache loading
                load_start = time.time()
                result = np.load(cache_path)
                load_time = time.time() - load_start
                
                total_time = time.time() - total_start
                if not self.is_logging_blocked():
                    print(f"   🎯 CACHE HIT: {num_texts} text(s) loaded in {load_time:.4f}s (Total: {total_time:.4f}s)")
                return result
            else:
                # File was deleted externally, remove from index
                self.cache_index.discard(cache_key)
        
        # Cache miss - track performance
        self._update_cache_stats(hit=False)
        cache_time = time.time() - cache_start
        
        # Time embedding generation
        embedding_start = time.time()
        if self.model_type == "sentence-transformers":
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        else:
            embeddings = self._embed_via_api(texts)
        embedding_time = time.time() - embedding_start
        
        # Time cache saving
        save_start = time.time()
        np.save(cache_path, embeddings)
        self.cache_metadata[cache_path] = {
            "size": cache_path.stat().st_size,
            "last_used": time.time(),
            "created": time.time()
        }
        # Add to fast lookup index
        self.cache_index.add(cache_key)
        self._enforce_cache_limit()
        save_time = time.time() - save_start
        
        # Calculate total time
        total_time = time.time() - total_start
        
        # Show embedding summary based on logging level
        if not self.is_logging_blocked():
            print(f"   🧠 EMBEDDING GENERATED: {num_texts} text(s)")
            print(f"      ┌─ Cache lookup:   {cache_time:>8.4f}s")
            print(f"      ├─ Embedding gen:  {embedding_time:>8.4f}s")
            print(f"      ├─ Cache save:     {save_time:>8.4f}s")
            print(f"      └─ TOTAL:          {total_time:>8.4f}s")
            print(f"      📊 Per text:       {total_time/num_texts:.4f}s")
            print(f"      ⚡ Throughput:     {num_texts/total_time:.2f} texts/second")
            
            # Show individual text lengths only if verbose logging is enabled
            if self.verbose_logging is True and num_texts <= 5:
                print(f"      📝 Text details:")
                for i, text in enumerate(texts):
                    print(f"         Text {i+1}: {len(text)} chars")
        
        return embeddings

    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate MD5 hash of texts"""
        text_blob = "|||".join(texts).encode('utf-8')
        return hashlib.md5(text_blob).hexdigest()

    def _embed_via_api(self, texts: List[str]) -> np.ndarray:
        """Handle API-based embedding generation"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        if self.model_type == "lmstudio":
            payload = {
                "input": texts,
                "model": self.model_name or "local-model"
            }
            endpoint = f"{self.base_url}/embeddings"
        elif self.model_type == "ollama":
            payload = {
                "model": self.model_name,
                "prompt": texts[0]  # Ollama handles one string at a time
            }
            endpoint = f"{self.base_url}/api/embeddings"
        elif self.model_type == "vllm":
            payload = {
                "input": texts,
                "model": self.model_name
            }
            endpoint = f"{self.base_url}/embeddings"
        elif self.model_type in ["openai", "openai-compatible"]:
            payload = {
                "input": texts,
                "model": self.model_name or "text-embedding-ada-002"
            }
            endpoint = f"{self.base_url}/embeddings"
        
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if self.model_type == "ollama":
            return np.array([data["embedding"]])
        elif self.model_type in ["openai", "openai-compatible"]:
            return np.array([item["embedding"] for item in data["data"]])
        return np.array([item["embedding"] for item in data["data"]])

    def clear_cache(self):
        """Completely reset the cache"""
        for f in self.cache_dir.glob("*.npy"):
            f.unlink()
        self.cache_metadata = {}
        self.cache_index.clear()  # Clear in-memory index

    def get_cache_stats(self) -> Dict[str, Union[int, str, float]]:
        """Get cache performance statistics"""
        total_files = len(self.cache_index)
        total_size = sum(m["size"] for m in self.cache_metadata.values())
        cache_hit_rate = getattr(self, '_cache_hits', 0) / max(1, getattr(self, '_cache_requests', 0))
        
        return {
            "total_files": total_files,
            "total_size": self._format_bytes(total_size),
            "max_size": self._format_bytes(self.max_bytes),
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "cache_policy": self.cache_policy,
            "index_size": len(self.cache_index)
        }

    def optimize_cache_index(self):
        """Rebuild cache index for consistency (useful after external file operations)"""
        self.cache_index.clear()
        for f in self.cache_dir.glob("*.npy"):
            self.cache_index.add(f.stem)
        print(f"Cache index rebuilt: {len(self.cache_index)} entries")

    def _update_cache_stats(self, hit: bool):
        """Track cache performance metrics"""
        if not hasattr(self, '_cache_hits'):
            self._cache_hits = 0
            self._cache_requests = 0
        
        self._cache_requests += 1
        if hit:
            self._cache_hits += 1

    def set_logging(self, verbose: Union[bool, str]):
        """Enable or disable logging
        
        Args:
            verbose: True=verbose, False=concise, "Block"=silent
        """
        self.verbose_logging = verbose
        if verbose == "Block":
            status = "blocked (silent)"
        elif verbose is True:
            status = "enabled (verbose)"
        else:
            status = "enabled (concise)"
        print(f"🔊 Logging {status}")

    def is_logging_enabled(self) -> bool:
        """Check if verbose logging is enabled"""
        return self.verbose_logging is True
    
    def is_logging_blocked(self) -> bool:
        """Check if logging is completely blocked (silent mode)"""
        return self.verbose_logging == "Block"
    
    def is_logging_concise(self) -> bool:
        """Check if logging is in concise mode (summaries only)"""
        return self.verbose_logging is False



class VectorDB:
    """Enhanced FAISS vector storage with auto-compaction support"""
    
    def __init__(self, 
                 storage_dir: str = "faiss_db",
                 embedding_model: Optional['EmbeddingModel'] = None,
                 auto_compact: bool = True,
                 compact_threshold: float = 0.3):
        """
        Initialize vector database with auto-compaction
        
        Args:
            storage_dir: Directory for FAISS index storage
            embedding_model: Pre-configured embedding model
            auto_compact: Enable automatic compaction
            compact_threshold: Fragmentation ratio (deleted/total) to trigger compaction
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.index_file = self.storage_dir / "index.faiss"
        self.metadata_file = self.storage_dir / "metadata.pkl"
        
        self.file_to_ids: Dict[str, List[int]] = {}
        self.id_to_chunk: Dict[int, Tuple[str, str]] = {}
        self.deleted_ids = set()
        
        self.embedding_model = embedding_model or EmbeddingModel()
        self.auto_compact = auto_compact
        self.compact_threshold = compact_threshold
        self._initialize_db()

    def _check_compaction_needed(self) -> bool:
        """Check if compaction should trigger automatically"""
        if not self.auto_compact or not self.deleted_ids:
            return False
            
        fragmentation = len(self.deleted_ids) / max(1, self.index.ntotal)
        return fragmentation >= self.compact_threshold

    def _auto_compact(self):
        """Conditionally run compaction based on fragmentation"""
        if self._check_compaction_needed():
            warnings.warn(
                f"Auto-compacting due to high fragmentation ({len(self.deleted_ids)}/{self.index.ntotal} vectors deleted)",
                UserWarning
            )
            self.compact_storage()

    def compact_storage(self):
        """Rebuild storage to remove fragmentation"""
        # 1. Export active data
        active_ids = set(self.id_to_chunk.keys())
        embeddings = np.vstack([
            self.index.reconstruct(id)
            for id in sorted(active_ids)
        ])
        
        # 2. Rebuild index
        self.index = faiss.IndexFlatL2(self.index.d)
        self.index.add(embeddings)
        
        # 3. Remap metadata
        id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(active_ids))}
        self.id_to_chunk = {
            id_map[old_id]: data
            for old_id, data in self.id_to_chunk.items()
        }
        self.file_to_ids = {
            f: [id_map[old_id] for old_id in ids]
            for f, ids in self.file_to_ids.items()
        }
        self.deleted_ids.clear()
        
        self._save_db()

    def remove_document(self, file_path: str) -> bool:
        """Enhanced removal with auto-compaction check"""
        if file_path not in self.file_to_ids:
            return False
            
        removed_ids = set(self.file_to_ids[file_path])
        self.deleted_ids.update(removed_ids)
        
        # Original removal logic here...
        # ... [previous remove_document implementation]
        
        self._auto_compact()  # Check if compaction needed
        return True

    
    def add_document(self, file_path: str, chunks: List[str], embeddings: Optional[np.ndarray] = None):
        """
        Add document chunks to the database
        Args:
            file_path: Path to source document
            chunks: List of text chunks from document
            embeddings: Optional pre-computed embeddings
                      (will generate if not provided)
        """
        """Add document with pre-compaction check"""
        
        if self._check_compaction_needed():
            self.compact_storage()
            
        if embeddings is None:
            embeddings = self.embedding_model.embed(chunks)
            
        if file_path in self.file_to_ids:
            self.remove_document(file_path)
        
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        # Track metadata
        new_ids = list(range(start_id, start_id + len(chunks)))
        self.file_to_ids[file_path] = new_ids
        
        for idx, chunk in zip(new_ids, chunks):
            self.id_to_chunk[idx] = (file_path, chunk)
        
        self._save_db()
        return new_ids

    def remove_document(self, file_path: str) -> bool:
        """Remove all vectors for a specific file"""
        if file_path not in self.file_to_ids:
            return False
        
        # Get IDs to remove
        ids_to_remove = set(self.file_to_ids[file_path])
        
        # Create mask of vectors to keep
        mask = np.ones(self.index.ntotal, dtype=bool)
        mask[list(ids_to_remove)] = False
        
        # Rebuild index without removed vectors
        remaining_embeddings = self.index.reconstruct_n(0, self.index.ntotal)[mask]
        new_index = faiss.IndexFlatL2(self.index.d)
        new_index.add(remaining_embeddings)
        
        # Update metadata
        self.index = new_index
        del self.file_to_ids[file_path]
        self.id_to_chunk = {
            new_id: self.id_to_chunk[old_id]
            for new_id, old_id in enumerate(sorted(set(self.id_to_chunk.keys()) - ids_to_remove))
            if old_id in self.id_to_chunk
        }
        
        # Update file_to_ids references
        for f, ids in self.file_to_ids.items():
            self.file_to_ids[f] = [
                new_id for old_id in ids 
                if (new_id := self._get_new_id(old_id, ids_to_remove)) is not None
            ]
        
        self._save_db()
        return True

    def _get_new_id(self, old_id: int, removed_ids: set) -> Optional[int]:
        """Helper for ID remapping after deletion"""
        if old_id in removed_ids:
            return None
        return len([id for id in self.id_to_chunk if id < old_id and id not in removed_ids])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Search database using text query
        Args:
            query: Search query text
            top_k: Number of results to return
        Returns:
            List of (chunk_text, score, file_path) tuples
        """
        query_embedding = self.embedding_model.embed([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [
            (self.id_to_chunk[idx][1], float(distances[0][i]), self.id_to_chunk[idx][0])
            for i, idx in enumerate(indices[0])
            if idx in self.id_to_chunk
        ]

    def get_document_chunks(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Retrieve all chunks for a specific document
        Args:
            file_path: Path to source document
        Returns:
            List of (chunk_text, vector_id) tuples
        """
        if file_path not in self.file_to_ids:
            return []
        return [
            (self.id_to_chunk[id][1], id)
            for id in self.file_to_ids[file_path]
            if id in self.id_to_chunk
        ]
