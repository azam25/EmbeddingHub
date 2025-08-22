<h1 align="center" style="font-size: 56px; font-weight: 800; margin: 0; line-height: 1.1;">
  <span style="background: linear-gradient(90deg,#7C3AED,#06B6D4,#10B981,#F59E0B,#EF4444); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent;">EmbeddingHub</span>
</h1>
<p align="center" style="font-size: 16px; color: #111111; margin: 6px 0 24px;">— Centralizing AI Embeddings for Smarter Enterprises.</p>

---
![EmbeddingHub Key Capabilities](image/capabilities.png)
### Why EmbeddingHub?
EmbeddingHub is a production-grade Python library that centralizes document chunking, embedding generation, caching, and vector search into a clean, configurable API. It supports local and remote embedding providers (including OpenAI and OpenAI-compatible servers), fast FAISS indexing, and a flexible logging system tailored for development and production.


### Key Technical Features
- **Multi-provider Embeddings**: `sentence-transformers`, `LM Studio`, `Ollama`, `vLLM`, `OpenAI`, and `OpenAI-compatible` servers.
- **Vector Database**: High-performance FAISS with `IndexFlatIP`/`IndexFlatL2`, L2-normalization, and scalable ingestion.
- **Smart Chunking**: Semantic and recursive splitters with configurable chunk size/overlap.
- **File Handling**: PDFs, DOCX, PPTX, CSVs, ZIPs, and more via utility loaders.
- **Robust Caching**:
  - File-based `.npy` cache with MD5 keys
  - In-memory index for O(1) cache lookup
  - Cache cleanup via LRU/FIFO and size limits
  - Cache statistics and index repair
- **Performance Logging (3 Levels)**:
  - `True` (verbose): full details, per-chunk timings
  - `False` (concise): summaries only
  - `"Block"` (silent): no logs at all
- **Retrieval Layer**: Search with source names, simple/complex tag filters, and filename filtering.
- **Complete CRUD Operations**: Create, Read, Update, **Delete** with full source and index management.
- **Multi-Backend Storage**: File system, MongoDB, PostgreSQL, and extensible custom backends.


### Business Benefits
- **Faster Time-to-Value**: Drop-in vectors + chunking + retrieval without stitching multiple tools.
- **Cost Efficient**: Aggressive caching and O(1) lookups cut repeated embedding costs.
- **Scales with You**: FAISS-backed search and efficient ingestion support large corpora.
- **Ops Friendly**: Three-tier logging for dev, production, and automation.
- **Cloud or Local**: Use local models or managed APIs (OpenAI or your own OpenAI-compatible server).
- **Complete Data Lifecycle**: Full CRUD operations including secure deletion and index management.
- **Multi-Backend Flexibility**: Choose storage that fits your infrastructure (files, databases, cloud).

### Architecture 

![EmbeddingHub Architecture](image/ArchitectureEmbeddingHub.png)

## Installation
```bash
pip install -r requirements.txt
# Core extras you may need
pip install sentence-transformers faiss-cpu numpy requests PyPDF2 python-docx python-pptx pandas
```

If you plan to use OpenAI or OpenAI-compatible providers:
```bash
pip install openai>=1.0.0
```


## Quick Start
```python
from embeddings import EmbeddingModel
from retriever import DocRetriever

# 1) Choose your embedding backend
embedder = EmbeddingModel(
    model_type="sentence-transformers",   # "lmstudio", "ollama", "vllm", "openai", "openai-compatible"
    model_name="all-MiniLM-L6-v2",
    cache_dir="embedding_cache",
    max_cache_size="1GB",                 # e.g., "500MB", "2GB"
    cache_policy="lru",                   # or "fifo"
    verbose_logging=False                   # True | False | "Block"
)

# 2) Build a retriever
doc_retriever = DocRetriever(
    embedding_model=embedder,
    chunk_size=1000,
    chunk_overlap=200,
    chunking_strategy="recursive",        # or "semantic"
    verbose_logging=False                  # True | False | "Block"
)

# 3) Add a document source (PDF/TXT/ZIP supported via utils)
doc_retriever.add_source(
    file_path="Data/XYZ-Germany.zip",
    source_name="ZIP-RFP",
    tags=["rfp", "xyz", "2024"]
)

# 4) Retrieve relevant chunks
results = doc_retriever.retrieve(
    query="EUV lithography roadmap",
    k=5,
    tags=["rfp"]
)

for r in results:
    print(r["score"], r["text"][:120].replace("\n", " ") + "...")
```


## Providers and Configuration
### Local: sentence-transformers
```python
embedder = EmbeddingModel(
    model_type="sentence-transformers",
    model_name="all-MiniLM-L6-v2",
    verbose_logging=False
)
```

### LM Studio
```python
embedder = EmbeddingModel(
    model_type="lmstudio",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                  # placeholder
    model_name="your-local-embedding-model",
    verbose_logging=False
)
```

### Ollama
```python
embedder = EmbeddingModel(
    model_type="ollama",
    base_url="http://localhost:11434",
    api_key="ollama",                      # placeholder
    model_name="nomic-embed-text",
    verbose_logging=False
)
```

### OpenAI
```python
embedder = EmbeddingModel(
    model_type="openai",
    api_key="YOUR_OPENAI_KEY",
    model_name="text-embedding-3-small",   # or your choice
    verbose_logging=False
)
```

### OpenAI-Compatible (self-hosted)
```python
embedder = EmbeddingModel(
    model_type="openai-compatible",
    base_url="https://your-server/v1",
    api_key="YOUR_SERVER_KEY",
    model_name="text-embedding-3-small",
    verbose_logging=False
)
```


## Logging Levels
- `True` (verbose): full details, per-chunk timings, and text length snapshots
- `False` (concise): high-level summaries only (no per-chunk details)
- `"Block"` (silent): no logs at all (ideal for automation/CI)

Change at runtime:
```python
# Switch globally on the retriever (it propagates to the embedder)
doc_retriever.set_logging(True)     # verbose
doc_retriever.set_logging(False)    # concise
doc_retriever.set_logging("Block")  # silent
```


## Caching System
- `.npy` files saved to `cache_dir` keyed by MD5 of input text(s)
- In-memory `set` for O(1) cache membership checks
- Two-stage check: in-memory -> filesystem existence
- Policy-based cleanup: LRU or FIFO with `max_cache_size`
- Utilities:
```python
# Clear everything
embedder.clear_cache()

# Stats
stats = embedder.get_cache_stats()
print(stats)

# Rebuild index if you manually change files
embedder.optimize_cache_index()
```


## Chunking
Use `chunk_size`, `chunk_overlap`, and `chunking_strategy` to control segmentation.
- `chunking_strategy="recursive"`: token/character aware recursive splitter
- `chunking_strategy="semantic"`: semantic boundaries using tokenizers

```python
doc_retriever = DocRetriever(
    embedding_model=embedder,
    chunk_size=800,
    chunk_overlap=120,
    chunking_strategy="semantic",
    verbose_logging=False
)
```


## Vector Database (FAISS)
- Inner Product (`IndexFlatIP`) with optional L2-normalized vectors
- Fast single/mini-batch ingestion with per-chunk normalization
- Designed for high-throughput ingestion and retrieval

Retrieval example:
```python
results = doc_retriever.retrieve(
    query="risk assessment for supplier quality",
    k=5,
    tags=["rfp", "quality"],
)
```


## Advanced Retrieval Filters
- By tags (OR/AND) and complex groups
- By `source_name`
- By partial `file_name`

```python
# Complex tag groups: ("finance" OR "2024") AND ("rfp")
results = doc_retriever.retrieve(
    query="payment terms and liabilities",
    k=5,
    tag_groups=[["finance", "2024"], ["rfp"]]
)

# By file name (partial, case-insensitive)
results = doc_retriever.retrieve(
    query="laser power",
    k=3,
    file_name="XYZ Customer"
)
```

## Source and Index Management
EmbeddingHub provides complete CRUD operations for managing your document sources and FAISS indexes:

### **Delete Operations**
```python
# Delete specific source and its chunks
retriever.delete_source("document_name")

# Delete all sources and clear index
retriever.delete_all_sources()

# Delete storage files from any backend
retriever.delete_storage_files("storage_path")

# Rebuild index after deletions (handles FAISS limitations)
retriever.rebuild_index()
```

### **Storage Management**
```python
# Get storage information
info = retriever.get_storage_info()
print(f"Backend: {info['backend_type']}")
print(f"Sources: {info['sources_count']}")
print(f"Index size: {info['index_size']}")

# Save current state
retriever.save_to_disk("my_storage")

# Load from storage
retriever.load_from_disk("my_storage")
```

### **Why Rebuild Index?**
FAISS doesn't support direct vector deletion, so the system:
1. **Removes sources** from metadata
2. **Clears the index**
3. **Rebuilds from remaining sources**
4. **Saves updated state** to storage backend


## End-to-End Example
```python
from embeddings import EmbeddingModel
from retriever import DocRetriever

embedder = EmbeddingModel(
    model_type="openai-compatible",
    base_url="https://enterprise-embeddings.your-domain/v1",
    api_key="YOUR_KEY",
    model_name="text-embedding-3-small",
    cache_dir="embedding_cache",
    max_cache_size="2GB",
    cache_policy="lru",
    verbose_logging=False
)

retriever = DocRetriever(
    embedding_model=embedder,
    chunk_size=1200,
    chunk_overlap=200,
    chunking_strategy="recursive",
    verbose_logging=False
)

retriever.add_source("/data/contracts/Q1_Contracts.zip", source_name="contracts-Q1", tags=["contracts", "q1", "legal"]) 

answers = retriever.retrieve(
    query="termination clause for supplier default",
    k=5,
    tags=["legal"],
)

for i, a in enumerate(answers, 1):
    print(f"{i}. ({a['score']:.4f}) {a['text'][:160]}...")
```


## Vector DB Storage Options
EmbeddingHub provides **3 different ways** to run your Vector Database:

### **1. In-Memory Vector DB (Default)**
```python
# Default behavior - everything stays in memory
retriever = DocRetriever(
    embedding_model=embedder,
    verbose_logging=False
)

# FAISS index and metadata stay in memory
# No persistence - data lost when program exits
# Best for: Development, testing, temporary processing
```

### **2. Local File System Storage**
```python
from storage_interface import FileSystemBackend

# File system backend (default if no backend specified)
retriever = DocRetriever(
    embedding_model=embedder,
    storage_backend=FileSystemBackend(),  # Saves to local disk
    verbose_logging=False
)

# Saves to: storage_path/faiss_index.faiss + storage_path/metadata.pkl
retriever.save_to_disk("my_storage")
retriever.load_from_disk("my_storage")

# Best for: Single-server deployments, development, local persistence
```

### **3. External Database Storage (MongoDB, PostgreSQL, etc.)**
```python
from storage_interface import MongoDBBackend, PostgreSQLBackend

# MongoDB backend
mongo_backend = MongoDBBackend(
    connection_string="mongodb://localhost:27017",
    database="embeddinghub",
    collection="faiss_storage"
)

retriever = DocRetriever(
    embedding_model=embedder,
    storage_backend=mongo_backend,  # Saves to MongoDB
    verbose_logging=False
)

# Same save/load API works with any backend
retriever.save_to_disk("mongo_path")
retriever.load_from_disk("mongo_path")

# Best for: Production, multi-server, cloud deployments, scalability
```

### **Storage Mode Comparison**
| Mode | Performance | Persistence | Scalability | Use Case |
|------|-------------|-------------|-------------|----------|
| **In-Memory** | ⚡ Fastest | ❌ None | ❌ Single server | Development, testing |
| **File System** | 🚀 Fast | ✅ Local | ❌ Single server | Single-server deployment |
| **Database** | 🐌 Slower | ✅ Full | ✅ Multi-server | Production, cloud |

### **Switching Between Modes**
```python
# Start with file system
retriever = DocRetriever(embedding_model=embedder)

# Switch to MongoDB
mongo_backend = MongoDBBackend(...)
retriever.storage_backend = mongo_backend

# Switch back to file system
file_backend = FileSystemBackend()
retriever.storage_backend = file_backend

# All modes use the same API!
retriever.save_to_disk("path")
retriever.load_from_disk("path")
retriever.delete_source("name")
```

## Persistent Storage Backends
EmbeddingHub supports multiple storage backends for FAISS index and metadata persistence:

### **MongoDB Backend**
```python
from storage_interface import MongoDBBackend

mongo_backend = MongoDBBackend(
    connection_string="mongodb://localhost:27017",
    database="embeddinghub",
    collection="faiss_storage"
)

retriever = DocRetriever(
    embedding_model=embedder,
    storage_backend=mongo_backend
)
```

### **PostgreSQL Backend**
```python
from storage_interface import PostgreSQLBackend

postgres_backend = PostgreSQLBackend(
    connection_string="postgresql://user:pass@localhost:5432/embeddinghub",
    table_prefix="faiss_storage"
)

retriever = DocRetriever(
    embedding_model=embedder,
    storage_backend=postgres_backend
)
```

### **Custom Storage Backends**
Extend the `StorageBackend` abstract class to implement any storage system:

```python
from storage_interface import StorageBackend

class S3Backend(StorageBackend):
    def __init__(self, bucket_name: str, aws_credentials: dict):
        self.bucket_name = bucket_name
        self.aws_credentials = aws_credentials
    
    def save_index(self, index, path: str) -> bool:
        # Implement S3 upload logic
        pass
    
    def load_index(self, path: str):
        # Implement S3 download logic
        pass
    
    # ... implement other abstract methods
```

### **Persistence Operations**
```python
# Save FAISS index and metadata
retriever.save_to_disk("storage_path")

# Load from persistent storage
retriever.load_from_disk("storage_path")

# Get storage information
info = retriever.get_storage_info()
print(f"Backend: {info['backend_type']}")
print(f"Sources: {info['sources_count']}")
print(f"Index size: {info['index_size']}")
```

### **Deletion Operations**
```python
# Delete specific source and its chunks
retriever.delete_source("source_name")

# Delete all sources and clear index
retriever.delete_all_sources()

# Delete storage files from backend
retriever.delete_storage_files("storage_path")

# Rebuild index after deletions
retriever.rebuild_index()
```

## Testing and Examples
Comprehensive examples are provided to demonstrate all features:

### **Storage Backend Examples**
```bash
# Test all storage backends
python storage_backend_examples.py
```

### **Deletion Operations Examples**
```bash
# Test deletion capabilities
python deletion_examples.py
```

### **Three Logging Levels**
```bash
# Test logging control
python test_three_logging_levels.py
```

## Tips for Production
- Prefer `verbose_logging=False` in user-facing services
- Use `"Block"` in automation/batch jobs to keep logs clean
- Periodically inspect `embedder.get_cache_stats()` and size limits
- Use persistent storage backends for production deployments
- Implement proper connection pooling for database backends
- Use cloud storage backends (S3, GCS) for scalable deployments
- Consider MongoDB for document-oriented storage needs
- Use PostgreSQL for ACID compliance and complex queries
- **For deletion operations**: Use `rebuild_index()` after deletions to maintain consistency
- **For large datasets**: Consider incremental updates rather than full rebuilds


## License
This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

```
Copyright 2025 EmbeddingHub

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

