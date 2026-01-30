"""
Retrieval-Augmented Generation (RAG) Engine
Handles document retrieval from knowledge base using embeddings
"""

import numpy as np
import logging
from typing import List, Dict, Optional
from pathlib import Path
import json

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Using fallback retrieval.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Using simple retrieval.")

from config import RAG_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine.
    
    Uses embeddings and vector similarity to retrieve relevant
    documents from knowledge base for question generation.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize RAG engine.
        
        Args:
            config: Configuration dictionary. Uses RAG_CONFIG if None.
        """
        if config is None:
            config = RAG_CONFIG
        
        self.config = config
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.document_embeddings = None
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        logger.info("RAG Engine initialized")
        logger.info(f"  Embedding model: {config['embedding_model']}")
        logger.info(f"  Top-k retrieval: {config['top_k_retrieval']}")
    
    def _initialize_embedding_model(self):
        """Initialize the sentence embedding model"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(
                    self.config['embedding_model']
                )
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.embedding_model = None
        else:
            logger.warning("Using fallback embedding (random)")
            self.embedding_model = None
    
    def load_knowledge_base(self, kb_path: str):
        """
        Load knowledge base from file.
        
        Args:
            kb_path: Path to knowledge base (JSON or CSV)
        """
        kb_path = Path(kb_path)
        
        if not kb_path.exists():
            logger.warning(f"Knowledge base not found: {kb_path}")
            self._create_default_kb()
            return
        
        logger.info(f"Loading knowledge base from {kb_path}")
        
        if kb_path.suffix == '.json':
            with open(kb_path, 'r') as f:
                self.documents = json.load(f)
        elif kb_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(kb_path)
            self.documents = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {kb_path.suffix}")
        
        logger.info(f"Loaded {len(self.documents)} documents")
        
        # Build index
        self._build_index()
    
    def _create_default_kb(self):
        """Create a default knowledge base for demo purposes"""
        self.documents = [
            {
                'id': 1,
                'topic': 'Calculus',
                'content': 'Calculus is the mathematical study of continuous change. '
                          'It has two major branches: differential calculus and integral calculus.',
                'difficulty': 'Medium',
                'bloom_level': 1
            },
            {
                'id': 2,
                'topic': 'Linear Algebra',
                'content': 'Linear algebra is the branch of mathematics concerning linear equations '
                          'and linear functions, and their representations through matrices.',
                'difficulty': 'Medium',
                'bloom_level': 2
            },
            {
                'id': 3,
                'topic': 'Probability',
                'content': 'Probability theory is the branch of mathematics concerned with '
                          'probability, the analysis of random phenomena.',
                'difficulty': 'Easy',
                'bloom_level': 1
            },
        ]
        logger.info("Created default knowledge base with 3 documents")
        self._build_index()
    
    def _build_index(self):
        """Build vector index for fast retrieval"""
        if not self.documents:
            logger.warning("No documents to index")
            return
        
        # Extract content
        contents = [doc.get('content', '') for doc in self.documents]
        
        # Generate embeddings
        if self.embedding_model is not None:
            logger.info("Generating embeddings...")
            self.document_embeddings = self.embedding_model.encode(
                contents,
                show_progress_bar=True
            )
        else:
            # Fallback: random embeddings
            embedding_dim = 384  # Default dimension
            self.document_embeddings = np.random.randn(
                len(contents), embedding_dim
            ).astype('float32')
        
        # Build FAISS index if available
        if FAISS_AVAILABLE:
            dimension = self.document_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.document_embeddings)
            logger.info(f"Built FAISS index with {len(self.documents)} vectors")
        else:
            logger.info("Using simple cosine similarity for retrieval")
    
    def retrieve(self,
                query: str,
                bloom_level: Optional[int] = None,
                difficulty: Optional[str] = None,
                top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            bloom_level: Filter by Bloom level (1-3)
            difficulty: Filter by difficulty ('Easy', 'Medium', 'Hard')
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        if top_k is None:
            top_k = self.config['top_k_retrieval']
        
        if not self.documents:
            logger.warning("No documents in knowledge base")
            return []
        
        # Generate query embedding
        if self.embedding_model is not None:
            query_embedding = self.embedding_model.encode([query])[0]
        else:
            # Fallback: random
            query_embedding = np.random.randn(
                self.document_embeddings.shape[1]
            ).astype('float32')
        
        # Search
        if FAISS_AVAILABLE and self.index is not None:
            # Use FAISS
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1),
                min(top_k * 2, len(self.documents))  # Get more for filtering
            )
            
            retrieved_docs = []
            for dist, idx in zip(distances[0], indices[0]):
                doc = self.documents[idx].copy()
                doc['score'] = float(1 / (1 + dist))  # Convert distance to similarity
                retrieved_docs.append(doc)
        else:
            # Use simple cosine similarity
            similarities = np.dot(
                self.document_embeddings,
                query_embedding
            ) / (
                np.linalg.norm(self.document_embeddings, axis=1) *
                np.linalg.norm(query_embedding)
            )
            
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]
            
            retrieved_docs = []
            for idx in top_indices:
                doc = self.documents[idx].copy()
                doc['score'] = float(similarities[idx])
                retrieved_docs.append(doc)
        
        # Apply filters
        if bloom_level is not None:
            retrieved_docs = [
                d for d in retrieved_docs
                if d.get('bloom_level') == bloom_level
            ]
        
        if difficulty is not None:
            retrieved_docs = [
                d for d in retrieved_docs
                if d.get('difficulty') == difficulty
            ]
        
        # Return top-k after filtering
        retrieved_docs = retrieved_docs[:top_k]
        
        logger.debug(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}")
        
        return retrieved_docs
    
    def add_document(self, document: Dict):
        """
        Add a new document to the knowledge base.
        
        Args:
            document: Document dictionary with 'content' and metadata
        """
        self.documents.append(document)
        
        # Rebuild index (in production, use incremental indexing)
        self._build_index()
        
        logger.info(f"Added document. Total: {len(self.documents)}")
    
    def save_knowledge_base(self, filepath: str):
        """
        Save knowledge base to file.
        
        Args:
            filepath: Output file path (JSON)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        logger.info(f"Knowledge base saved to {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("RAG ENGINE DEMO")
    print("=" * 80)
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Create default knowledge base
    rag._create_default_kb()
    
    # Test retrieval
    print("\n" + "=" * 80)
    print("TESTING RETRIEVAL")
    print("=" * 80)
    
    queries = [
        "What is calculus?",
        "Explain matrices",
        "Random events probability"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = rag.retrieve(query, top_k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"\n  Result {i} (score: {doc['score']:.3f}):")
            print(f"    Topic: {doc['topic']}")
            print(f"    Content: {doc['content'][:80]}...")
    
    # Test with filters
    print("\n" + "=" * 80)
    print("TESTING WITH FILTERS")
    print("=" * 80)
    
    results = rag.retrieve(
        "mathematics",
        bloom_level=1,
        difficulty='Medium',
        top_k=5
    )
    
    print(f"\nRetrieved {len(results)} documents with filters")
    for doc in results:
        print(f"  - {doc['topic']}: Bloom={doc['bloom_level']}, Difficulty={doc['difficulty']}")
    
    print("\n" + "=" * 80)
    print("RAG Engine Demo Complete!")
    print("=" * 80)
