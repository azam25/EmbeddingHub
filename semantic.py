from typing import List, Dict
from PyPDF2 import PdfReader
from transformers import AutoTokenizer
from semantic_text_splitter import TextSplitter
import os

class SemanticChunker:
    def __init__(self, tokenizer_name="bert-base-uncased", chunk_size=500, threshold=0.5):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Workaround for tokenizer compatibility
            self.splitter = TextSplitter.from_tiktoken(
                chunk_size=chunk_size,
                chunk_overlap=0,
                tokens_per_chunk=chunk_size,
                threshold=threshold
            )
        except Exception as e:
            raise ValueError(
                f"Semantic chunker init failed: {str(e)}\n"
                "Required: pip install transformers semantic-text-splitter"
            )

    def chunk(self, file_path: str) -> List[Dict]:
        """Process PDF into semantic chunks"""
        chunks = []
        reader = PdfReader(file_path)
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():
                for chunk in self.splitter.chunks(text):
                    chunks.append({
                        "text": chunk,
                        "metadata": {
                            "source": os.path.basename(file_path),
                            "page": page_num,
                            "chunking_strategy": "semantic"
                        }
                    })
        return chunks