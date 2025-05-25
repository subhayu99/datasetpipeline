"""
This module provides functionality for semantic deduplication using configurable embeddings.

The module defines two classes:

- `SemanticDedupConfig`: A Pydantic model that defines the configuration options for semantic deduplication.
- `SemanticDedup`: A class that performs semantic deduplication on a dataset.

Example Usage:

```python
from datasets import Dataset
from semantic import SemanticDedup, SemanticDedupConfig

# Create a dataset
dataset = Dataset.from_pandas(pd.DataFrame({"messages": ["Hello", "World"]}))

# Create a configuration object with lightweight embeddings (default)
config = SemanticDedupConfig(column="messages", threshold=0.8)

# Or use full embeddings for better quality
config_full = SemanticDedupConfig(
    column="messages", 
    threshold=0.8, 
    embedding_mode="full",
    embeddings_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)

# Create an instance of SemanticDedup
deduplicator = SemanticDedup(dataset, config)

# Perform semantic deduplication
deduplicated_dataset = deduplicator.dedup()

# Access the deduplicated dataset
print(deduplicated_dataset.deduplicated)
print(deduplicated_dataset.duplicates)
```

"""

from typing import Any, Literal, Optional
from functools import cached_property
import warnings

import pandas as pd
from langchain_community.vectorstores import FAISS
from datasets import Dataset, DatasetDict
from pydantic import Field, validator

from .base import BaseDedupConfig, BaseDedup
from ..helpers.utils import hash_uuid
from ..helpers.embeddings import Embeddings, GET_DEFAULT_EMBEDDINGS, EmbeddingConfig


class SemanticDedupConfig(BaseDedupConfig):
    """Configuration for semantic deduplication."""
    
    column: str = Field(
        default="messages",
        description="Name of the column to deduplicate. Defaults to 'messages'"
    )
    threshold: float = Field(
        default=0.8,
        description="Minimum threshold to consider two messages similar. Defaults to '0.8'",
        ge=0.0,
        le=1.0
    )
    cache_embeddings: bool = Field(
        default=True,  # Changed default to True for better performance
        description="Whether to cache the embeddings. Defaults to 'true'"
    )
    embedding_mode: Literal['lightweight', 'full'] = Field(
        default="lightweight",
        description="Embedding mode: 'lightweight' for fast ONNX-based embeddings or 'full' for HuggingFace transformers. Defaults to 'lightweight'"
    )
    embeddings_model: Optional[str] = Field(
        default=None,
        description="Name of the embedding model to use. If None, uses default model for the selected mode"
    )
    device: Optional[Literal['mps', 'cuda', 'npu', 'hpu', 'cpu']] = Field(
        default=None,
        description="Device to use for embeddings (only applies to 'full' mode). Defaults to 'cpu'"
    )
    multi_process: bool = Field(
        default=False,
        description="Whether to use multiple processing (only applies to 'full' mode). Use only when dataset is large. Defaults to 'false'"
    )
    show_progress: bool = Field(
        default=True,
        description="Whether to show progress during embedding generation. Defaults to 'true'"
    )
    database_url: str = Field(
        default="sqlite:///semantic_dedup_embeddings.db",
        description="Database URL for caching embeddings. Defaults to 'sqlite:///semantic_dedup_embeddings.db'"
    )

    @validator('threshold')
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Threshold must be between 0.0 and 1.0')
        return v

    @validator('embeddings_model', always=True)
    def set_default_model(cls, v, values):
        if v is None:
            mode = values.get('embedding_mode', 'lightweight')
            if mode == 'lightweight':
                return EmbeddingConfig.LIGHTWEIGHT_MODEL
            else:
                return EmbeddingConfig.FULL_MODEL
        return v


class SemanticDedup(BaseDedup):
    """Semantic deduplication using embeddings with configurable backends."""
    
    def __init__(self, dataset: Dataset, config: SemanticDedupConfig = None):
        if config is None:
            config = SemanticDedupConfig()
        super().__init__(dataset, config)
        self.config: SemanticDedupConfig = config
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embeddings based on configuration."""
        try:
            if self.config.embedding_mode == "lightweight":
                # Use lightweight embeddings
                from ..helpers.embeddings import LightweightEmbeddings
                embedding_model = LightweightEmbeddings(self.config.embeddings_model)
                
            elif self.config.embedding_mode == "full":
                # Use full HuggingFace embeddings
                try:
                    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
                    embedding_model = HuggingFaceEmbeddings(
                        model_name=self.config.embeddings_model,
                        model_kwargs={'device': self.config.device or 'cpu'},
                        multi_process=self.config.multi_process,
                        show_progress=self.config.show_progress,
                    )
                except ImportError:
                    warnings.warn(
                        "HuggingFace embeddings not available. Falling back to lightweight mode. "
                        "Install with: pip install langchain-community transformers torch"
                    )
                    from ..helpers.embeddings import LightweightEmbeddings
                    embedding_model = LightweightEmbeddings(self.config.embeddings_model)
            else:
                raise ValueError(f"Unknown embedding mode: {self.config.embedding_mode}")

            self.embeddings = Embeddings(
                model=embedding_model,
                use_cache=self.config.cache_embeddings,
                database_url=self.config.database_url,
                embedding_mode=self.config.embedding_mode
            )
            
        except Exception as e:
            warnings.warn(f"Failed to initialize embeddings: {str(e)}. Using default lightweight embeddings.")
            self.embeddings = Embeddings(
                use_cache=self.config.cache_embeddings,
                database_url=self.config.database_url,
                embedding_mode="lightweight"
            )

    @cached_property
    def can_be_deduped(self) -> bool:
        """Check if the dataset can be deduplicated."""
        if not isinstance(self.dataset, Dataset) or self.dataset.shape[0] == 0:
            return False
        
        if self.config.column not in self.dataset.column_names:
            return False
            
        try:
            _col = self.dataset[self.config.column]
            if not all(isinstance(x, str) for x in _col):
                return False
        except Exception:
            return False
            
        return True

    def _create_vector_store(self, texts: list[str], embeddings: list[list[float]]) -> FAISS:
        """Create a FAISS vector store from texts and embeddings."""
        text_embeddings = list(zip(texts, embeddings))
        ids = [hash_uuid(x).hex for x in texts]
        
        try:
            vdb = FAISS.from_embeddings(
                text_embeddings, 
                self.embeddings.model, 
                metadatas=[{"id": _id, "text": text} for _id, text in zip(ids, texts)],
                ids=ids,
            )
            return vdb
        except Exception as e:
            raise RuntimeError(f"Failed to create vector store: {str(e)}") from e

    def _find_similar_documents(self, dataset_with_embeddings: Dataset, vdb: FAISS) -> pd.DataFrame:
        """Find similar documents using vector similarity search."""
        embeddings_col = "_embeddings"
        score_col = "score"
        match_col = "match"
        match_id_col = "match_id"

        def add_score_and_match(row: dict[str, Any]) -> dict[str, str | float]:
            try:
                # Get the second most similar document (first is the document itself)
                results = vdb.similarity_search_with_score_by_vector(
                    row[embeddings_col], 
                    k=2
                )
                
                if len(results) < 2:
                    # If only one result, it's the document itself
                    return {
                        match_col: row[self.config.column], 
                        score_col: 0.0,
                        match_id_col: "self"
                    }
                
                # Take the second result (most similar other document)
                doc, score = results[1]
                return {
                    match_col: doc.page_content, 
                    score_col: float(score),
                    match_id_col: doc.metadata.get("id", "unknown")
                }
                
            except Exception as e:
                warnings.warn(f"Error in similarity search: {str(e)}")
                return {
                    match_col: row[self.config.column], 
                    score_col: 0.0,
                    match_id_col: "error"
                }

        # Apply similarity search
        result_dataset = dataset_with_embeddings.map(
            add_score_and_match,
            desc="Finding similar documents" if self.config.show_progress else None
        )
        
        return result_dataset.to_pandas()

    def _normalize_scores_and_deduplicate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Normalize similarity scores and perform deduplication."""
        score_col = "score"
        match_col = "match"
        match_id_col = "match_id"
        embeddings_col = "_embeddings"
        
        # Normalize the scores between 0 and 1
        scores = df[score_col]
        if scores.max() > scores.min():
            df[score_col] = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            df[score_col] = 0.0  # All scores are the same
        
        # Convert similarity threshold to distance threshold
        # Higher threshold means more strict (less similar documents considered duplicates)
        distance_threshold = 1 - self.config.threshold
        
        # Identify duplicates: documents with similarity score above threshold
        duplicate_mask = df[score_col] >= distance_threshold
        
        # Create deduplicated dataframe
        # Keep documents that are not duplicates OR are the first occurrence of duplicates
        unique_texts = set()
        keep_indices = []
        
        for idx, row in df.iterrows():
            text = row[self.config.column]
            if not duplicate_mask.iloc[idx] or text not in unique_texts:
                keep_indices.append(idx)
                unique_texts.add(text)
        
        deduped_df = df.loc[keep_indices].drop(
            columns=[embeddings_col, match_col, score_col, match_id_col]
        ).sort_index()
        
        # Create duplicates dataframe
        duplicate_indices = df.index.difference(deduped_df.index)
        duplicates_df = df.loc[duplicate_indices].drop(
            columns=[embeddings_col, match_col, score_col, match_id_col]
        ).sort_index()
        
        return deduped_df, duplicates_df

    def _dedup(self) -> DatasetDict:
        """Perform semantic deduplication."""
        if not self.can_be_deduped:
            warnings.warn("Dataset cannot be deduplicated. Returning original dataset.")
            return DatasetDict(
                deduplicated=self.dataset,
                duplicates=Dataset.from_pandas(pd.DataFrame())
            )

        try:
            # Extract texts and generate embeddings
            texts = self.dataset[self.config.column]
            
            if self.config.show_progress:
                print(f"Generating embeddings for {len(texts)} documents...")
            
            embeddings = self.embeddings.embed_documents(texts)
            
            # Create vector store
            if self.config.show_progress:
                print("Creating vector store...")
            
            vdb = self._create_vector_store(texts, embeddings)
            
            # Add embeddings to dataset
            dataset_with_embeddings = self.dataset.add_column("_embeddings", embeddings)
            
            # Find similar documents
            if self.config.show_progress:
                print("Finding similar documents...")
            
            df = self._find_similar_documents(dataset_with_embeddings, vdb)
            
            # Clean up memory
            del vdb, dataset_with_embeddings, texts, embeddings
            
            # Normalize scores and deduplicate
            if self.config.show_progress:
                print("Performing deduplication...")
            
            deduped_df, duplicates_df = self._normalize_scores_and_deduplicate(df)
            
            # Create result datasets
            result = DatasetDict(
                deduplicated=Dataset.from_pandas(deduped_df),
                duplicates=Dataset.from_pandas(duplicates_df)
            )
            
            if self.config.show_progress:
                print(f"Deduplication complete. Kept {len(deduped_df)} documents, removed {len(duplicates_df)} duplicates.")
            
            return result
            
        except Exception as e:
            warnings.warn(f"Deduplication failed: {str(e)}. Returning original dataset.")
            return DatasetDict(
                deduplicated=self.dataset,
                duplicates=Dataset.from_pandas(pd.DataFrame())
            )

    def get_embedding_info(self) -> dict[str, Any]:
        """Get information about the current embedding configuration."""
        return {
            "embedding_mode": self.config.embedding_mode,
            "model_name": getattr(self.embeddings.model, 'model_name', 'unknown'),
            "use_cache": self.config.cache_embeddings,
            "database_url": self.config.database_url,
            "threshold": self.config.threshold,
        }

    def switch_embedding_mode(self, mode: Literal['lightweight', 'full']):
        """Switch between embedding modes dynamically."""
        self.config.embedding_mode = mode
        if mode == 'lightweight':
            self.config.embeddings_model = EmbeddingConfig.LIGHTWEIGHT_MODEL
        else:
            self.config.embeddings_model = EmbeddingConfig.FULL_MODEL
        self._initialize_embeddings()
