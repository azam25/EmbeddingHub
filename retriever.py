import numpy as np
import faiss
import pickle
import os
import time
from typing import List, Dict, Optional, Literal, Any, Union
from .embeddings import EmbeddingModel
from .storage_interface import StorageBackend, FileSystemBackend

class DocRetriever:
    def __init__(self, 
                embedding_model: EmbeddingModel,
                chunk_size: int = 1000,
                chunk_overlap: int = 200,
                chunking_strategy: str = "recursive",
                verbose_logging: Union[bool, str] = True,  # True=verbose, False=concise, "Block"=silent
                storage_backend: Optional[StorageBackend] = None,
                **chunking_kwargs):
        self.embedder = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.chunking_kwargs = chunking_kwargs
        self.verbose_logging = verbose_logging
        self.sources = {}
        
        # Initialize storage backend (default to file system)
        self.storage_backend = storage_backend or FileSystemBackend()

    def set_logging(self, verbose: Union[bool, str]):
        """Enable or disable logging for both retriever and embedding model
        
        Args:
            verbose: True=verbose, False=concise, "Block"=silent
        """
        self.verbose_logging = verbose
        self.embedder.set_logging(verbose)
        if verbose == "Block":
            status = "blocked (silent)"
        elif verbose is True:
            status = "enabled (verbose)"
        else:
            status = "enabled (concise)"
        print(f"🔊 Retriever logging {status}")

    def is_logging_enabled(self) -> bool:
        """Check if verbose logging is enabled"""
        return self.verbose_logging is True
    
    def is_logging_blocked(self) -> bool:
        """Check if logging is completely blocked (silent mode)"""
        return self.verbose_logging == "Block"
    
    def is_logging_concise(self) -> bool:
        """Check if logging is in concise mode (summaries only)"""
        return self.verbose_logging is False

    def add_source(self, file_path: str, source_name: str, save_dir: Optional[str] = None, tags: Optional[List[str]] = None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.is_logging_blocked():
            print(f"\n🔄 Processing document: {os.path.basename(file_path)}")
            print(f"📊 Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        # Time chunking process
        chunking_start = time.time()
        from .utils import load_and_chunk_pdf
        chunks = load_and_chunk_pdf(
            file_path=file_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            strategy=self.chunking_strategy,
            **self.chunking_kwargs
        )
        chunking_time = time.time() - chunking_start
        
        if not chunks:
            raise ValueError(f"No chunks generated from {file_path}")

        if not self.is_logging_blocked():
            print(f"✅ Generated {len(chunks)} chunks in {chunking_time:.4f}s")
            print(f"📝 Average chunk length: {sum(len(chunk['text']) for chunk in chunks) / len(chunks):.0f} characters")

        # Time metadata preparation
        metadata_start = time.time()
        for chunk in chunks:
            chunk["metadata"]["source_name"] = source_name
            chunk["metadata"]["file_name"] = os.path.basename(file_path)
        metadata_time = time.time() - metadata_start

        # Time embedding generation with per-chunk details
        if not self.is_logging_blocked():
            print(f"\n🧠 Generating embeddings for {len(chunks)} chunks...")
            if self.verbose_logging is True:
                print("   📊 Per-chunk embedding timing:")
        
        texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = []
        total_embedding_time = 0
        
        # Process each chunk individually to get per-chunk timing
        for i, text in enumerate(texts):
            chunk_start = time.time()
            chunk_embedding = self.embedder.embed([text])  # Single chunk
            chunk_time = time.time() - chunk_start
            total_embedding_time += chunk_time
            
            chunk_embeddings.append(chunk_embedding[0])  # Extract single embedding
            if self.verbose_logging:
                print(f"      Chunk {i+1}: {chunk_time:.4f}s ({len(text)} chars)")
        
        # Convert to numpy array
        embeddings = np.array(chunk_embeddings).astype('float32')
        
        # Show embedding summary based on logging level
        if not self.is_logging_blocked():
            print(f"   ✅ Total embedding time: {total_embedding_time:.4f}s")
            print(f"   📊 Embedding shape: {embeddings.shape}")
            print(f"   ⚡ Average time per chunk: {total_embedding_time/len(chunks):.4f}s")

        # Time vector database operations with per-chunk ingestion
        if not self.is_logging_blocked():
            print(f"\n🗄️  Building vector database...")
            if self.verbose_logging is True:
                print("   📊 Per-chunk FAISS ingestion timing:")
        
        if not self.sources:
            self.embedding_dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(embeddings.shape[1])
        
        # Ingest each chunk individually to get per-chunk timing
        total_ingestion_time = 0
        for i, embedding in enumerate(embeddings):
            ingestion_start = time.time()
            
            # Normalize single embedding
            single_embedding = embedding.reshape(1, -1)
            faiss.normalize_L2(single_embedding)
            
            # Add to index
            index.add(single_embedding)
            
            ingestion_time = time.time() - ingestion_start
            total_ingestion_time += ingestion_time
            
            if self.verbose_logging:
                print(f"      Chunk {i+1}: {ingestion_time:.4f}s (dim: {embedding.shape[0]})")
        
        # Show ingestion summary based on logging level
        if not self.is_logging_blocked():
            print(f"   ✅ Total ingestion time: {total_ingestion_time:.4f}s")
            print(f"   ⚡ Average ingestion per chunk: {total_ingestion_time/len(chunks):.4f}s")

        # Time metadata storage
        storage_start = time.time()
        self.sources[source_name] = {
            "index": index,
            "docstore": {i: chunk for i, chunk in enumerate(chunks)},
            "tags": set(tags) if tags else set(),
            "file_name": os.path.basename(file_path)
        }
        storage_time = time.time() - storage_start

        # Calculate total time
        total_time = time.time() - chunking_start
        
        # Print comprehensive performance summary based on logging level
        if not self.is_logging_blocked():
            if self.verbose_logging is True:
                print(f"\n📈 PERFORMANCE SUMMARY for '{source_name}':")
                print(f"   ┌─ Chunking:        {chunking_time:>8.4f}s")
                print(f"   ├─ Metadata prep:   {metadata_time:>8.4f}s")
                print(f"   ├─ Embedding gen:   {total_embedding_time:>8.4f}s")
                print(f"   ├─ Vector DB build: {total_ingestion_time:>8.4f}s")
                print(f"   ├─ Storage:         {storage_time:>8.4f}s")
                print(f"   └─ TOTAL:           {total_time:>8.4f}s")
                print(f"   📊 Throughput:      {len(chunks)/total_time:.2f} chunks/second")
                print(f"   🏷️  Tags:           {tags or 'None'}")
                
                # Per-chunk performance analysis
                print(f"\n🔍 PER-CHUNK PERFORMANCE ANALYSIS:")
                print(f"   📊 Total chunks processed: {len(chunks)}")
                print(f"   ⚡ Embedding throughput: {len(chunks)/total_embedding_time:.2f} chunks/second")
                print(f"   🗄️  Ingestion throughput: {len(chunks)/total_ingestion_time:.2f} chunks/second")
                print(f"   🎯 Overall efficiency: {len(chunks)/total_time:.2f} chunks/second")
            else:
                # Concise mode - show summary but not per-chunk details
                print(f"\n📈 PERFORMANCE SUMMARY for '{source_name}':")
                print(f"   ┌─ Chunking:        {chunking_time:>8.4f}s")
                print(f"   ├─ Metadata prep:   {metadata_time:>8.4f}s")
                print(f"   ├─ Embedding gen:   {total_embedding_time:>8.4f}s")
                print(f"   ├─ Vector DB build: {total_ingestion_time:>8.4f}s")
                print(f"   ├─ Storage:         {storage_time:>8.4f}s")
                print(f"   └─ TOTAL:           {total_time:>8.4f}s")
                print(f"   📊 Throughput:      {len(chunks)/total_time:.2f} chunks/second")
                print(f"   🏷️  Tags:           {tags or 'None'}")
        else:
            # Silent operation - just show completion
            print(f"✅ Added {len(chunks)} chunks to source '{source_name}' in {total_time:.4f}s")

        if save_dir is not None:
            save_start = time.time()
            self.save(save_dir)
            save_time = time.time() - save_start
            if not self.is_logging_blocked():
                print(f"   💾 Save to disk:   {save_time:>8.4f}s")

    def get_source_names_by_complex_tags(self, tag_groups: List[List[str]]) -> List[str]:
        """
        tag_groups: List of lists. Outer list is AND, inner list is OR.
        e.g., [["finance", "2024"], ["rfp"]] means (finance OR 2024) AND (rfp)
        """
        matched = []
        for source_name, source_info in self.sources.items():
            source_tags = source_info.get('tags', set())
            if all(any(tag in source_tags for tag in group) for group in tag_groups):
                matched.append(source_name)
        return matched

    def get_source_names_by_file_name(self, file_name: str) -> List[str]:
        """
        Returns all source_names where file_name matches (case-insensitive, supports partial match).
        """
        file_name = file_name.lower()
        matched = []
        for source_name, source_info in self.sources.items():
            src_file_name = source_info.get("file_name", "").lower()
            if file_name in src_file_name:
                matched.append(source_name)
        return matched
    
    def save_to_disk(self, save_dir: str) -> bool:
        """
        Save FAISS index and metadata to persistent storage
        
        Args:
            save_dir: Directory to save the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import os
            from pathlib import Path
            
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            
            # Save FAISS index
            index_path = save_path / "faiss_index.faiss"
            if hasattr(self, 'index') and self.index is not None:
                success = self.storage_backend.save_index(self.index, str(index_path))
                if not success:
                    if not self.is_logging_blocked():
                        print(f"❌ Failed to save FAISS index to {index_path}")
                    return False
            
            # Save metadata
            metadata_path = save_path / "metadata.pkl"
            metadata = {
                'sources': self.sources,
                'embedding_dim': getattr(self, 'embedding_dim', None),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'chunking_strategy': self.chunking_strategy
            }
            
            success = self.storage_backend.save_metadata(metadata, str(metadata_path))
            if not success:
                if not self.is_logging_blocked():
                    print(f"❌ Failed to save metadata to {metadata_path}")
                return False
            
            if not self.is_logging_blocked():
                print(f"✅ Successfully saved to {save_dir}")
            return True
            
        except Exception as e:
            if not self.is_logging_blocked():
                print(f"❌ Error saving to disk: {e}")
            return False
    
    def load_from_disk(self, load_dir: str) -> bool:
        """
        Load FAISS index and metadata from persistent storage
        
        Args:
            load_dir: Directory to load the data from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from pathlib import Path
            
            load_path = Path(load_dir)
            
            # Load metadata first
            metadata_path = load_path / "metadata.pkl"
            if not self.storage_backend.exists(str(metadata_path)):
                if not self.is_logging_blocked():
                    print(f"❌ Metadata file not found: {metadata_path}")
                return False
            
            metadata = self.storage_backend.load_metadata(str(metadata_path))
            if not metadata:
                if not self.is_logging_blocked():
                    print(f"❌ Failed to load metadata from {metadata_path}")
                return False
            
            # Restore metadata
            self.sources = metadata.get('sources', {})
            self.embedding_dim = metadata.get('embedding_dim')
            self.chunk_size = metadata.get('chunk_size', self.chunk_size)
            self.chunk_overlap = metadata.get('chunk_overlap', self.chunk_overlap)
            self.chunking_strategy = metadata.get('chunking_strategy', self.chunking_strategy)
            
            # Load FAISS index
            index_path = load_path / "faiss_index.faiss"
            if self.storage_backend.exists(str(index_path)):
                self.index = self.storage_backend.load_index(str(index_path))
                if self.index is None:
                    if not self.is_logging_blocked():
                        print(f"❌ Failed to load FAISS index from {index_path}")
                    return False
            else:
                if not self.is_logging_blocked():
                    print(f"⚠️  FAISS index not found: {index_path}")
                return False
            
            if not self.is_logging_blocked():
                print(f"✅ Successfully loaded from {load_dir}")
                print(f"📊 Loaded {len(self.sources)} sources with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            if not self.is_logging_blocked():
                print(f"❌ Error loading from disk: {e}")
            return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the current storage backend
        
        Returns:
            Dict containing storage backend information
        """
        backend_name = self.storage_backend.__class__.__name__
        info = {
            'backend_type': backend_name,
            'sources_count': len(self.sources),
            'has_index': hasattr(self, 'index') and self.index is not None,
            'index_size': getattr(self, 'index', None).ntotal if hasattr(self, 'index') and self.index is not None else 0
        }
        
        if hasattr(self.storage_backend, 'connection_string'):
            info['connection_string'] = self.storage_backend.connection_string
        
        return info
    
    def delete_source(self, source_name: str) -> bool:
        """
        Delete a source and its associated chunks from FAISS index and storage
        
        Args:
            source_name: Name of the source to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if source_name not in self.sources:
                if not self.is_logging_blocked():
                    print(f"❌ Source '{source_name}' not found")
                return False
            
            source_info = self.sources[source_name]
            file_path = source_info.get("file_path", "")
            
            if not self.is_logging_blocked():
                print(f"🗑️  Deleting source '{source_name}' ({file_path})")
            
            # Remove from sources dict
            del self.sources[source_name]
            
            # Note: FAISS doesn't support direct deletion, so we'll rebuild the index
            # This is a limitation of FAISS - we need to recreate without the deleted chunks
            if hasattr(self, 'index') and self.index is not None:
                if not self.is_logging_blocked():
                    print(f"⚠️  FAISS index will be rebuilt on next save (FAISS limitation)")
            
            if not self.is_logging_blocked():
                print(f"✅ Source '{source_name}' deleted successfully")
            return True
            
        except Exception as e:
            if not self.is_logging_blocked():
                print(f"❌ Error deleting source '{source_name}': {e}")
            return False
    
    def delete_all_sources(self) -> bool:
        """
        Delete all sources and clear the FAISS index
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_logging_blocked():
                print(f"🗑️  Deleting all sources ({len(self.sources)} sources)")
            
            # Clear sources
            self.sources.clear()
            
            # Clear FAISS index
            if hasattr(self, 'index') and self.index is not None:
                self.index = None
                if not self.is_logging_blocked():
                    print(f"🗑️  FAISS index cleared")
            
            if not self.is_logging_blocked():
                print(f"✅ All sources deleted successfully")
            return True
            
        except Exception as e:
            if not self.is_logging_blocked():
                print(f"❌ Error deleting all sources: {e}")
            return False
    
    def delete_storage_files(self, storage_path: str) -> bool:
        """
        Delete storage files from the storage backend
        
        Args:
            storage_path: Path to delete from storage
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_logging_blocked():
                print(f"🗑️  Deleting storage files from '{storage_path}'")
            
            # Delete from storage backend
            success = self.storage_backend.delete(storage_path)
            
            if success:
                if not self.is_logging_blocked():
                    print(f"✅ Storage files deleted successfully")
            else:
                if not self.is_logging_blocked():
                    print(f"⚠️  No storage files found to delete")
            
            return success
            
        except Exception as e:
            if not self.is_logging_blocked():
                print(f"❌ Error deleting storage files: {e}")
            return False
    
    def rebuild_index(self) -> bool:
        """
        Rebuild FAISS index from current sources (useful after deletions)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.sources:
                if not self.is_logging_blocked():
                    print(f"⚠️  No sources to rebuild index from")
                return False
            
            if not self.is_logging_blocked():
                print(f"🔨 Rebuilding FAISS index from {len(self.sources)} sources...")
            
            # Clear existing index
            if hasattr(self, 'index') and self.index is not None:
                self.index = None
            
            # Rebuild index by re-adding all sources
            total_chunks = 0
            for source_name, source_info in self.sources.items():
                file_path = source_info.get("file_path", "")
                if os.path.exists(file_path):
                    # Re-add the source to rebuild the index
                    self._add_to_index(source_name, file_path)
                    total_chunks += len(source_info.get("chunks", []))
            
            if not self.is_logging_blocked():
                print(f"✅ FAISS index rebuilt successfully with {total_chunks} chunks")
            
            return True
            
        except Exception as e:
            if not self.is_logging_blocked():
                print(f"❌ Error rebuilding index: {e}")
            return False
    
    def _add_to_index(self, source_name: str, file_path: str):
        """Helper method to add source to FAISS index (internal use)"""
        try:
            from .utils import load_and_chunk_pdf
            
            # Load and chunk the document
            chunks = load_and_chunk_pdf(
                file_path=file_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                strategy=self.chunking_strategy,
                **self.chunking_kwargs
            )
            
            if not chunks:
                return
            
            # Generate embeddings
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedder.embed(texts).astype('float32')
            
            # Initialize index if needed
            if not hasattr(self, 'index') or self.index is None:
                self.embedding_dim = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
            
            # Add to index
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            
            # Update source info
            self.sources[source_name]["chunks"] = chunks
            
        except Exception as e:
            if not self.is_logging_blocked():
                print(f"❌ Error adding {source_name} to index: {e}")

    def retrieve(self, 
                query: str, 
                k: int = 5, 
                source_names: Optional[List[str]] = None,
                tags: Optional[List[str]] = None,
                tag_operator: Literal['OR', 'AND'] = 'OR',
                tag_groups: Optional[List[List[str]]] = None,
                file_name: Optional[str] = None
                ) -> List[Dict[str, Any]]:
        """
        Enhanced retrieve supporting:
        - source_names (as before)
        - simple tags with AND/OR (as before)
        - complex tag groups: [["tag1","tag2"],["tag3"]] = (tag1 OR tag2) AND (tag3)
        - file_name: retrieve by filename (case-insensitive, partial ok)
        Priority: tag_groups > tags > file_name > source_names
        """
        if not self.sources:
            raise ValueError("No sources added - add documents first")

        # --- Tag group logic (most flexible, highest priority) ---
        if tag_groups:
            source_names = self.get_source_names_by_complex_tags(tag_groups)
            if not source_names:
                print(f"No sources found for complex tag groups {tag_groups}")
                return []

        # --- Simple tags (AND/OR logic) ---
        elif tags:
            tag_set = set(tags)
            matched = []
            for source_name, source_info in self.sources.items():
                source_tags = source_info.get('tags', set())
                if tag_operator == 'OR':
                    if tag_set & source_tags:
                        matched.append(source_name)
                elif tag_operator == 'AND':
                    if tag_set <= source_tags:
                        matched.append(source_name)
            source_names = matched
            if not source_names:
                print(f"No sources found for tags {tags} using {tag_operator} logic.")
                return []

        # --- File name logic ---
        elif file_name:
            source_names = self.get_source_names_by_file_name(file_name)
            if not source_names:
                print(f"No sources found for file_name containing '{file_name}'")
                return []

        # --- Default: all sources if none of the above is given
        if source_names is None:
            source_names = list(self.sources.keys())

        # --- Standard retrieval ---
        if self.verbose_logging:
            print(f"\n🔍 Processing search query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            print(f"📊 Searching in {len(source_names)} source(s): {source_names}")
            print(f"🎯 Target results: {k}")
        
        # Time query embedding generation
        query_start = time.time()
        query_embedding = np.array([self.embedder.embed(query)[0]]).astype('float32')
        faiss.normalize_L2(query_embedding)
        query_embedding_time = time.time() - query_start
        
        # Always show query embedding summary
        print(f"✅ Query embedding generated in {query_embedding_time:.4f}s")
        print(f"📊 Query embedding shape: {query_embedding.shape}")

        # Time vector search operations
        search_start = time.time()
        results = []
        total_search_time = 0
        
        for source_name in source_names:
            if source_name not in self.sources:
                raise ValueError(f"Source '{source_name}' not found")

            source = self.sources[source_name]
            source_search_start = time.time()
            scores, indices = source["index"].search(query_embedding, k)
            source_search_time = time.time() - source_search_start
            total_search_time += source_search_time
            
            if self.verbose_logging:
                print(f"   🔍 {source_name}: {source_search_time:.4f}s")

            for score, idx in zip(scores[0], indices[0]):
                if idx in source["docstore"]:
                    results.append({
                        **source["docstore"][idx],
                        "score": float(score)
                    })

        search_time = time.time() - search_start
        
        # Time result processing
        processing_start = time.time()
        results.sort(key=lambda x: x["score"], reverse=True)
        final_results = results[:k]
        processing_time = time.time() - processing_start
        
        # Show search performance summary based on logging level
        if not self.is_logging_blocked():
            print(f"\n📈 SEARCH PERFORMANCE SUMMARY:")
            print(f"   ┌─ Query embedding:  {query_embedding_time:>8.4f}s")
            print(f"   ├─ Vector search:    {search_time:>8.4f}s")
            print(f"   ├─ Result processing: {processing_time:>8.4f}s")
            print(f"   └─ TOTAL:            {query_embedding_time + search_time + processing_time:>8.4f}s")
            print(f"   📊 Results found:    {len(final_results)}")
            print(f"   ⚡ Search speed:     {k/(query_embedding_time + search_time + processing_time):.2f} results/second")
        
        return final_results

    def save(self, save_dir: str):
        """Save all sources (indices, docstores, tags, file_name) to disk."""
        os.makedirs(save_dir, exist_ok=True)

        for source_name, source in self.sources.items():
            faiss.write_index(source["index"], os.path.join(save_dir, f"{source_name}.index"))
            with open(os.path.join(save_dir, f"{source_name}_docstore.pkl"), "wb") as f:
                pickle.dump(source["docstore"], f)
            with open(os.path.join(save_dir, f"{source_name}_tags.pkl"), "wb") as f:
                pickle.dump(list(source["tags"]), f)
            with open(os.path.join(save_dir, f"{source_name}_file_name.txt"), "w") as f:
                f.write(source["file_name"])

        with open(os.path.join(save_dir, "retriever_meta.pkl"), "wb") as f:
            pickle.dump({
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "source_names": list(self.sources.keys())
            }, f)

    @classmethod
    def load(cls, save_dir: str, embedding_model: EmbeddingModel):
        """Load retriever state with tags and file_name from disk."""
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Directory not found: {save_dir}")

        with open(os.path.join(save_dir, "retriever_meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        retriever = cls(
            embedding_model=embedding_model,
            chunk_size=meta["chunk_size"],
            chunk_overlap=meta["chunk_overlap"]
        )

        for source_name in meta["source_names"]:
            index_path = os.path.join(save_dir, f"{source_name}.index")
            docstore_path = os.path.join(save_dir, f"{source_name}_docstore.pkl")
            tags_path = os.path.join(save_dir, f"{source_name}_tags.pkl")
            file_name_path = os.path.join(save_dir, f"{source_name}_file_name.txt")

            if os.path.exists(tags_path):
                with open(tags_path, "rb") as f:
                    tags = set(pickle.load(f))
            else:
                tags = set()
            if os.path.exists(file_name_path):
                with open(file_name_path, "r") as f:
                    file_name = f.read().strip()
            else:
                file_name = ""

            retriever.sources[source_name] = {
                "index": faiss.read_index(index_path),
                "docstore": pickle.load(open(docstore_path, "rb")),
                "tags": tags,
                "file_name": file_name
            }

        return retriever