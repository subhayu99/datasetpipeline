import pickle
import hashlib
import json
import warnings
import os
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, String, LargeBinary

from langchain_core.embeddings import Embeddings as LCEmbeddings

# --- Configuration ---
class EmbeddingConfig:
    """Configuration class for embedding models."""
    LIGHTWEIGHT_MODEL = "Xenova/all-MiniLM-L6-v2"
    FULL_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    DEFAULT_MODE = "lightweight"  # Can be "lightweight" or "full"


# --- Improved Lightweight Embeddings Class ---
class LightweightEmbeddings(LCEmbeddings):
    """
    A lightweight, fully functional embedding class using ONNX Runtime.

    This class runs a quantized model, providing a good balance of performance 
    and small dependency footprint. It does not require PyTorch or TensorFlow.
    """
    
    def __init__(self, model_id: str = EmbeddingConfig.LIGHTWEIGHT_MODEL):
        self.model_name = model_id
        self._validate_dependencies()
        self._initialize_model()

    def _validate_dependencies(self):
        """Validate that required dependencies are available."""
        try:
            import onnxruntime as ort
            import huggingface_hub
        except ImportError as e:
            raise ImportError(
                "The 'onnxruntime' and 'huggingface-hub' packages are required for LightweightEmbeddings. "
                "Please install them with: 'pip install onnxruntime huggingface-hub'"
            ) from e

    def _initialize_model(self):
        """Initialize the ONNX model and tokenizer."""
        try:
            from huggingface_hub import snapshot_download
            import onnxruntime as ort
            
            # Download model from Hugging Face Hub
            model_path = snapshot_download(repo_id=self.model_name)
            
            # Load the tokenizer and model
            self.tokenizer = self._load_tokenizer(model_path)
            onnx_path = Path(model_path) / "onnx" / "model.onnx"
            
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
                
            self.session = ort.InferenceSession(str(onnx_path))
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LightweightEmbeddings: {str(e)}") from e

    def _load_tokenizer(self, model_path: str):
        """Load tokenizer configuration and vocabulary."""
        tokenizer_config_path = Path(model_path) / "tokenizer.json"
        
        if not tokenizer_config_path.exists():
            raise FileNotFoundError(f"Tokenizer configuration not found at {tokenizer_config_path}")
            
        try:
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # Extract vocabulary and special tokens
            self.vocab = config['model']['vocab']
            self.cls_token_id = self.vocab.get('[CLS]', 101)
            self.sep_token_id = self.vocab.get('[SEP]', 102)
            self.unk_token_id = self.vocab.get('[UNK]', 100)
            self.pad_token_id = self.vocab.get('[PAD]', 0)
            
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Could not load vocab from tokenizer.json. "
                f"The tokenizer format may not be supported: {str(e)}"
            ) from e
        
        return self

    def _simple_tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization with basic preprocessing.
        Note: This is a simplified approach and may not handle all edge cases.
        """
        # Basic preprocessing
        text = text.lower().strip()
        
        # Simple whitespace tokenization
        # In a production system, you might want to use a more sophisticated tokenizer
        tokens = text.split()
        
        return tokens

    def _tokenize(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Tokenizes a list of texts and returns input for the ONNX model."""
        if not texts:
            raise ValueError("Input texts list cannot be empty")
            
        token_ids = []
        max_length = 512  # Standard BERT max length
        
        for text in texts:
            tokens = self._simple_tokenize(text)
            
            # Convert tokens to IDs
            ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
            
            # Add special tokens and truncate
            ids = [self.cls_token_id] + ids[:max_length-2] + [self.sep_token_id]
            token_ids.append(ids)

        # Pad to the length of the longest sequence
        max_len = min(max(len(ids) for ids in token_ids), max_length)
        
        input_ids = []
        attention_mask = []
        
        for ids in token_ids:
            # Pad sequence
            padded_ids = ids + [self.pad_token_id] * (max_len - len(ids))
            mask = [1] * len(ids) + [0] * (max_len - len(ids))
            
            input_ids.append(padded_ids)
            attention_mask.append(mask)
        
        return {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64)
        }

    def _mean_pooling(self, model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Performs mean pooling on the token embeddings."""
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        
        # Sum embeddings
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        
        # Sum mask (avoid division by zero)
        sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)
        
        return sum_embeddings / sum_mask

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of documents."""
        if not texts:
            return []
            
        try:
            tokenized_input = self._tokenize(texts)
            model_output = self.session.run(None, tokenized_input)
            
            # The first output is typically the last hidden state
            last_hidden_state = model_output[0]
            pooled_embeddings = self._mean_pooling(last_hidden_state, tokenized_input['attention_mask'])
            
            # Normalize embeddings
            norms = np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            normalized_embeddings = pooled_embeddings / norms
            
            return normalized_embeddings.tolist()
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        """Generates an embedding for a single query."""
        return self.embed_documents([text])[0]


# --- Default Embeddings Function with Hybrid Approach ---
def GET_DEFAULT_EMBEDDINGS(mode: str = None) -> LCEmbeddings:
    """
    Provides the default embedding model for the application with hybrid approach.
    
    Args:
        mode: Either "lightweight" or "full". If None, uses EmbeddingConfig.DEFAULT_MODE
        
    Returns:
        An embedding model instance
        
    Raises:
        ValueError: If mode is not supported
        ImportError: If required dependencies are not available
    """
    if mode is None:
        mode = os.getenv("EMBEDDING_MODE", EmbeddingConfig.DEFAULT_MODE)
    
    mode = mode.lower()
    
    if mode == "lightweight":
        return LightweightEmbeddings(EmbeddingConfig.LIGHTWEIGHT_MODEL)
    
    elif mode == "full":
        try:
            from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=EmbeddingConfig.FULL_MODEL,
                model_kwargs={"device": "cpu"},
                show_progress=False,
            )
        except ImportError as e:
            warnings.warn(
                "HuggingFace embeddings not available. Falling back to lightweight mode. "
                "Install with: pip install langchain-community transformers torch"
            )
            return LightweightEmbeddings(EmbeddingConfig.LIGHTWEIGHT_MODEL)
    
    else:
        raise ValueError(f"Unknown embedding mode: {mode}. Supported modes: 'lightweight', 'full'")


# --- Database and Caching Logic (Improved) ---
Base = declarative_base()

class EmbeddingRecord(Base):
    __tablename__ = "embeddings"
    text_hash = Column(String, primary_key=True)
    embedding = Column(LargeBinary)


@dataclass
class Embeddings:
    """
    Main embeddings class with caching capabilities.
    
    Args:
        model: The embedding model to use. If None, uses GET_DEFAULT_EMBEDDINGS()
        use_cache: Whether to use database caching
        database_url: SQLAlchemy database URL for caching
        embedding_mode: Mode for default embeddings ("lightweight" or "full")
    """
    model: Optional[LCEmbeddings] = None
    use_cache: bool = True
    database_url: str = "sqlite:///embeddings.db"
    embedding_mode: str = EmbeddingConfig.DEFAULT_MODE

    def __post_init__(self):
        if self.model is None:
            self.model = GET_DEFAULT_EMBEDDINGS(self.embedding_mode)
            
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the database for caching."""
        try:
            self.engine = create_engine(self.database_url, echo=False)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
        except Exception as e:
            warnings.warn(f"Failed to initialize database: {str(e)}. Disabling cache.")
            self.use_cache = False

    def hash_text(self, text: str) -> str:
        """Generate a hash for the text including model information."""
        model_name = getattr(self.model, 'model_name', str(type(self.model)))
        return hashlib.sha256((model_name + text).encode('utf-8')).hexdigest()

    def get_embedding_from_db(self, text_hash: str) -> Optional[List[float]]:
        """Retrieve embedding from database cache."""
        if not self.use_cache:
            return None
            
        session = self.Session()
        try:
            result = session.query(EmbeddingRecord).filter_by(text_hash=text_hash).first()
            return pickle.loads(result.embedding) if result else None
        except Exception as e:
            warnings.warn(f"Failed to retrieve embedding from cache: {str(e)}")
            return None
        finally:
            session.close()

    def _save_embedding_to_db(self, text_hash: str, embedding: List[float]):
        """Save embedding to database cache."""
        if not self.use_cache:
            return
            
        session = self.Session()
        try:
            binary_embedding = pickle.dumps(embedding)
            record = EmbeddingRecord(text_hash=text_hash, embedding=binary_embedding)
            session.merge(record)  # Insert or update
            session.commit()
        except Exception as e:
            warnings.warn(f"Failed to save embedding to cache: {str(e)}")
            session.rollback()
        finally:
            session.close()

    def _embed_documents_with_caching(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with caching support."""
        if not texts:
            return []
            
        text_hashes = {text: self.hash_text(text) for text in texts}
        cached_embeddings = {}
        uncached_texts_map = {}

        # Check cache for existing embeddings
        for text, text_hash in text_hashes.items():
            embedding = self.get_embedding_from_db(text_hash)
            if embedding is not None:
                cached_embeddings[text] = embedding
            else:
                uncached_texts_map[text] = text_hash

        uncached_texts = list(uncached_texts_map.keys())
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                generated_embeddings = self.model.embed_documents(uncached_texts)
                
                # Save new embeddings to cache
                for text, embedding in zip(uncached_texts, generated_embeddings):
                    text_hash = uncached_texts_map[text]
                    self._save_embedding_to_db(text_hash, embedding)
                    cached_embeddings[text] = embedding
                    
            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e

        # Return embeddings in original order
        return [cached_embeddings[text] for text in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        if self.use_cache:
            return self._embed_documents_with_caching(texts)
        else:
            return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed_documents([text])[0]

    def switch_mode(self, mode: str):
        """Switch between embedding modes."""
        self.embedding_mode = mode
        self.model = GET_DEFAULT_EMBEDDINGS(mode)


# --- Usage Examples ---
if __name__ == "__main__":
    # Example 1: Default lightweight mode
    embeddings = Embeddings()
    vectors = embeddings.embed_documents(["Hello world", "How are you?"])
    
    # Example 2: Explicit full mode
    embeddings_full = Embeddings(embedding_mode="full")
    
    # Example 3: Switch modes dynamically
    embeddings.switch_mode("full")
    
    # Example 4: Environment variable control
    os.environ["EMBEDDING_MODE"] = "lightweight"
    embeddings_env = Embeddings()
