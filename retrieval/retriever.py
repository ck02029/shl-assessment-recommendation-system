"""
Hybrid Retrieval System for SHL Assessments
Combines semantic search with keyword matching and type balancing
"""

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import logging
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssessmentRetriever:
    def __init__(
        self,
        index_path: str = "embeddings/faiss_index.bin",
        metadata_path: str = "embeddings/metadata.pkl",
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """Initialize retriever with FAISS index and metadata"""
        logger.info("Loading retriever components...")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index with {self.index.ntotal} assessments")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        self.assessments = metadata['assessments']
        
        # Load embedding model
        self.model = SentenceTransformer(model_name)
        
        # Build keyword index
        self._build_keyword_index()
        
        logger.info("Retriever initialized successfully")
    
    def _build_keyword_index(self):
        """Build inverted index for keyword matching"""
        self.keyword_index = {}
        
        for idx, assessment in enumerate(self.assessments):
            # Extract keywords from name and description
            text = f"{assessment['name']} {assessment['description']}".lower()
            words = re.findall(r'\b\w+\b', text)
            
            for word in set(words):
                if len(word) > 3:  # Ignore very short words
                    if word not in self.keyword_index:
                        self.keyword_index[word] = []
                    self.keyword_index[word].append(idx)
    
    def _keyword_search(self, query: str, k: int = 50) -> List[int]:
        """
        Find assessments using keyword matching
        
        Returns:
            List of assessment indices sorted by keyword match count
        """
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        match_counts = Counter()
        
        for word in query_words:
            if word in self.keyword_index:
                for idx in self.keyword_index[word]:
                    match_counts[idx] += 1
        
        # Return top K by match count
        return [idx for idx, _ in match_counts.most_common(k)]
    
    def _semantic_search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        """
        Perform semantic search using FAISS
        
        Returns:
            List of (index, score) tuples
        """
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return as list of (idx, score) tuples
        return list(zip(indices[0], scores[0]))
    
    def _detect_required_types(self, query: str, query_analysis: Optional[Dict] = None) -> Dict[str, float]:
        """
        Detect which test types are needed based on query
        
        Returns:
            Dict with test_type: weight mapping
        """
        query_lower = query.lower()
        weights = {'K': 0.0, 'P': 0.0, 'C': 0.0, 'General': 0.0}
        
        # Use LLM analysis if available
        if query_analysis:
            technical_skills = query_analysis.get('technical_skills', [])
            soft_skills = query_analysis.get('soft_skills', [])
            
            if technical_skills:
                weights['K'] = 0.6
            if soft_skills:
                weights['P'] = 0.4
            if 'cognitive' in query_analysis.get('additional_requirements', []):
                weights['C'] = 0.3
        
        # Keyword-based detection as fallback
        technical_keywords = [
            'java', 'python', 'javascript', 'sql', 'coding', 'programming',
            'technical', 'developer', 'engineer', 'excel', 'data analysis'
        ]
        
        soft_keywords = [
            'collaboration', 'teamwork', 'leadership', 'communication',
            'personality', 'behavior', 'motivation', 'interpersonal'
        ]
        
        cognitive_keywords = [
            'reasoning', 'cognitive', 'numerical', 'verbal', 'logical',
            'analytical', 'problem-solving'
        ]
        
        # Count keyword matches
        technical_count = sum(1 for kw in technical_keywords if kw in query_lower)
        soft_count = sum(1 for kw in soft_keywords if kw in query_lower)
        cognitive_count = sum(1 for kw in cognitive_keywords if kw in query_lower)
        
        total = technical_count + soft_count + cognitive_count
        
        if total > 0:
            weights['K'] = max(weights['K'], technical_count / total)
            weights['P'] = max(weights['P'], soft_count / total)
            weights['C'] = max(weights['C'], cognitive_count / total)
        else:
            # If no specific type detected, use general search
            weights['General'] = 1.0
        
        return weights
    
    def _balance_by_type(
        self,
        candidates: List[Dict],
        required_types: Dict[str, float],
        target_count: int = 10
    ) -> List[Dict]:
        """
        Balance results by test type based on requirements
        """
        if required_types.get('General', 0) == 1.0:
            # No specific type required, return top results
            return candidates[:target_count]
        
        # Separate by type
        by_type = {'K': [], 'P': [], 'C': [], 'General': []}
        for candidate in candidates:
            test_type = candidate.get('test_type', 'General')
            by_type[test_type].append(candidate)
        
        # Calculate target counts per type
        results = []
        remaining = target_count
        
        # Sort types by weight
        sorted_types = sorted(
            [(t, w) for t, w in required_types.items() if w > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        for test_type, weight in sorted_types:
            if remaining <= 0:
                break
            
            count = max(1, int(target_count * weight))
            count = min(count, remaining, len(by_type[test_type]))
            
            results.extend(by_type[test_type][:count])
            remaining -= count
        
        # Fill remaining slots with top candidates
        if remaining > 0:
            added_urls = {r['url'] for r in results}
            for candidate in candidates:
                if candidate['url'] not in added_urls:
                    results.append(candidate)
                    remaining -= 1
                    if remaining <= 0:
                        break
        
        return results[:target_count]
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        query_analysis: Optional[Dict] = None,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """
        Main retrieval function
        
        Args:
            query: Search query
            k: Number of results to return (5-10)
            query_analysis: Optional LLM analysis of query
            use_hybrid: Use hybrid search (semantic + keyword)
        
        Returns:
            List of top K assessments
        """
        logger.info(f"Retrieving assessments for query: {query}")
        
        # Retrieve more candidates than needed for reranking
        retrieval_k = min(50, k * 5)
        
        if use_hybrid:
            # Hybrid search: combine semantic and keyword
            semantic_results = self._semantic_search(query, retrieval_k)
            keyword_indices = self._keyword_search(query, retrieval_k)
            
            # Merge results with scoring
            scores = {}
            for idx, score in semantic_results:
                scores[idx] = score  # Semantic score (0-1)
            
            # Add keyword bonus
            for rank, idx in enumerate(keyword_indices[:20]):
                bonus = (20 - rank) / 20 * 0.2  # Up to 0.2 bonus
                scores[idx] = scores.get(idx, 0) + bonus
            
            # Sort by combined score
            sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            candidates = [
                {**self.assessments[idx], 'retrieval_score': score}
                for idx, score in sorted_indices[:retrieval_k]
            ]
        else:
            # Pure semantic search
            semantic_results = self._semantic_search(query, retrieval_k)
            candidates = [
                {**self.assessments[idx], 'retrieval_score': float(score)}
                for idx, score in semantic_results
            ]
        
        # Detect required test types
        required_types = self._detect_required_types(query, query_analysis)
        logger.info(f"Required type distribution: {required_types}")
        
        # Balance results by type
        balanced_results = self._balance_by_type(candidates, required_types, k)
        
        logger.info(f"Retrieved {len(balanced_results)} assessments")
        return balanced_results
    
    def get_assessment_by_url(self, url: str) -> Optional[Dict]:
        """Get assessment by URL"""
        for assessment in self.assessments:
            if assessment['url'] == url:
                return assessment
        return None


def main():
    """Test retriever"""
    retriever = AssessmentRetriever()
    
    test_queries = [
        "Java developer with collaboration skills",
        "Python programmer with SQL knowledge",
        "Need cognitive and personality tests",
        "Leadership assessment for managers"
    ]
    
    print("\n" + "="*80)
    print("Testing Retrieval System")
    print("="*80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        results = retriever.retrieve(query, k=5)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']}")
            print(f"   Type: {result['test_type']} | Score: {result.get('retrieval_score', 0):.3f}")
            print(f"   URL: {result['url']}\n")
    
    print("="*80)


if __name__ == "__main__":
    main()