"""
Build embeddings and FAISS index for SHL assessments
"""

import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingBuilder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding builder
        
        Args:
            model_name: Sentence transformer model name
                - 'all-MiniLM-L6-v2': Fast, 384 dimensions (recommended)
                - 'all-mpnet-base-v2': Better quality, 768 dimensions
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.assessments = []
        self.embeddings = None
        self.index = None
        
    def load_assessments(self, filepath: str = "data/raw_assessments.json") -> List[Dict]:
        """Load assessments from JSON file"""
        logger.info(f"Loading assessments from {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.assessments = json.load(f)
        logger.info(f"Loaded {len(self.assessments)} assessments")
        return self.assessments
    
    def create_rich_text(self, assessment: Dict) -> str:
        """
        Create rich text representation for better semantic understanding
        Combines multiple fields to create comprehensive text
        """
        parts = []
        
        # Assessment name (most important)
        if assessment.get('name'):
            parts.append(f"Assessment: {assessment['name']}")
        
        # Description
        if assessment.get('description'):
            parts.append(f"Description: {assessment['description']}")
        
        # Test type context
        test_type_map = {
            'K': 'Knowledge and Technical Skills Assessment',
            'P': 'Personality and Behavioral Assessment',
            'C': 'Cognitive Abilities Assessment',
            'General': 'General Assessment'
        }
        test_type = assessment.get('test_type', 'General')
        parts.append(f"Type: {test_type_map.get(test_type, 'General Assessment')}")
        
        # Category
        if assessment.get('category'):
            parts.append(f"Category: {assessment['category']}")
        
        # Add detailed description if available
        if assessment.get('detailed_description'):
            parts.append(assessment['detailed_description'])
        
        return ". ".join(parts)
    
    def build_embeddings(self) -> np.ndarray:
        """Generate embeddings for all assessments"""
        logger.info("Generating embeddings...")
        
        # Create rich text representations
        texts = [self.create_rich_text(assessment) for assessment in self.assessments]
        
        # Generate embeddings in batches
        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Cosine similarity
        )
        
        logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def build_faiss_index(self) -> faiss.Index:
        """Build FAISS index for efficient similarity search"""
        logger.info("Building FAISS index...")
        
        if self.embeddings is None:
            raise ValueError("Embeddings not generated. Call build_embeddings() first.")
        
        # Use IndexFlatIP for cosine similarity (since embeddings are normalized)
        # For larger datasets (10k+), consider IndexIVFFlat for faster search
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        return self.index
    
    def save_index(self, index_path: str = "embeddings/faiss_index.bin"):
        """Save FAISS index to disk"""
        if self.index is None:
            raise ValueError("Index not built. Call build_faiss_index() first.")
        
        faiss.write_index(self.index, index_path)
        logger.info(f"FAISS index saved to {index_path}")
    
    def save_metadata(self, metadata_path: str = "embeddings/metadata.pkl"):
        """Save assessment metadata"""
        metadata = {
            'assessments': self.assessments,
            'dimension': self.dimension,
            'model_name': self.model_name
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def save_embeddings(self, embeddings_path: str = "embeddings/embeddings.npy"):
        """Save raw embeddings as numpy array"""
        if self.embeddings is None:
            raise ValueError("Embeddings not generated.")
        
        np.save(embeddings_path, self.embeddings)
        logger.info(f"Embeddings saved to {embeddings_path}")
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Test search function
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of assessment dictionaries with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built.")
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            result = self.assessments[idx].copy()
            result['similarity_score'] = float(score)
            results.append(result)
        
        return results


def main():
    """Build embeddings and FAISS index"""
    
    # Initialize builder
    builder = EmbeddingBuilder(model_name='all-MiniLM-L6-v2')
    
    # Load assessments
    builder.load_assessments("data/raw_assessments.json")
    
    # Build embeddings
    builder.build_embeddings()
    
    # Build FAISS index
    builder.build_faiss_index()
    
    # Save everything
    builder.save_index("embeddings/faiss_index.bin")
    builder.save_metadata("embeddings/metadata.pkl")
    builder.save_embeddings("embeddings/embeddings.npy")
    
    # Test search
    print("\n" + "="*60)
    print("Testing search functionality...")
    print("="*60)
    
    test_query = "Java developer with collaboration skills"
    results = builder.search(test_query, k=5)
    
    print(f"\nQuery: {test_query}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']} (Score: {result['similarity_score']:.3f})")
        print(f"   Type: {result['test_type']} | URL: {result['url']}\n")
    
    print("="*60)
    print("OK Embeddings and index built successfully!")
    print("="*60)


if __name__ == "__main__":
    main()