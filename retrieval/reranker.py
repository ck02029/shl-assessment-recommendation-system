"""
LLM-based Reranker for Assessment Results
Uses Gemini to score and rerank retrieved assessments
"""

import os
import json
import re
from typing import List, Dict, Optional
import logging
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssessmentReranker:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize reranker with Gemini API
        
        Args:
            api_key: Google Gemini API key (or set GEMINI_API_KEY env variable)
        """
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("No Gemini API key found. Reranker will use fallback scoring.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini reranker initialized")
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10,
        query_analysis: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Rerank candidate assessments using LLM
        
        Args:
            query: Original search query
            candidates: List of candidate assessments
            top_k: Number of results to return
            query_analysis: Optional parsed query information
        
        Returns:
            Reranked list of top K assessments
        """
        if not self.model:
            return self._fallback_rerank(query, candidates, top_k, query_analysis)
        
        try:
            # Batch reranking for efficiency
            if len(candidates) > 20:
                # For large candidate sets, do multi-stage reranking
                # Stage 1: Quick scoring for top 20
                quick_scored = self._quick_score(query, candidates[:30])
                top_candidates = sorted(
                    quick_scored,
                    key=lambda x: x.get('quick_score', 0),
                    reverse=True
                )[:20]
                
                # Stage 2: Detailed reranking for top 20
                return self._detailed_rerank(query, top_candidates, top_k, query_analysis)
            else:
                return self._detailed_rerank(query, candidates, top_k, query_analysis)
                
        except Exception as e:
            logger.warning(f"LLM reranking failed: {str(e)}. Using fallback.")
            return self._fallback_rerank(query, candidates, top_k, query_analysis)
    
    def _quick_score(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Quick scoring for initial filtering"""
        # Create candidate list
        candidate_text = "\n".join([
            f"{i+1}. {c['name']} - {c.get('description', '')[:100]}"
            for i, c in enumerate(candidates)
        ])
        
        prompt = f"""Score these assessments for relevance to this query (1-10 scale).

Query: "{query}"

Assessments:
{candidate_text}

Return ONLY a JSON array of scores: [score1, score2, score3, ...]
Each score should be 1-10. Return ONLY the JSON array, no other text."""

        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            result_text = re.sub(r'```json\s*|\s*```', '', result_text)
            
            scores = json.loads(result_text)
            
            for i, score in enumerate(scores):
                if i < len(candidates):
                    candidates[i]['quick_score'] = float(score)
            
            return candidates
        except:
            # Fallback: use retrieval score
            for c in candidates:
                c['quick_score'] = c.get('retrieval_score', 0.5) * 10
            return candidates
    
    def _detailed_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
        query_analysis: Optional[Dict]
    ) -> List[Dict]:
        """
        Detailed reranking with comprehensive assessment
        """
        # Prepare candidate information
        candidate_details = []
        for i, candidate in enumerate(candidates):
            detail = {
                'index': i,
                'name': candidate['name'],
                'description': candidate.get('description', ''),
                'type': candidate.get('test_type', 'General'),
                'category': candidate.get('category', '')
            }
            candidate_details.append(detail)
        
        # Build prompt with context
        context = ""
        if query_analysis:
            context = f"""
Context from query analysis:
- Technical skills needed: {', '.join(query_analysis.get('technical_skills', []))}
- Soft skills needed: {', '.join(query_analysis.get('soft_skills', []))}
- Seniority: {query_analysis.get('seniority', 'unspecified')}
"""
        
        prompt = f"""You are an expert in talent assessment. Evaluate and rank these assessments for relevance to the hiring query.

Query: "{query}"
{context}

Assessment Types:
- K = Knowledge & Technical Skills
- P = Personality & Behavioral  
- C = Cognitive Abilities

Assessments to evaluate:
{json.dumps(candidate_details, indent=2)}

Instructions:
1. Score each assessment 1-10 for relevance
2. Consider: skill match, test type appropriateness, comprehensiveness
3. If query needs both technical AND behavioral skills, ensure balanced selection
4. Return top {top_k} assessments

Return ONLY a JSON array of indices (0-based) in ranked order:
[best_index, second_best_index, ...]

Return ONLY the JSON array of {top_k} indices, no other text."""

        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            result_text = re.sub(r'```json\s*|\s*```', '', result_text)
            
            ranked_indices = json.loads(result_text)
            
            # Return reranked results
            reranked = []
            for rank, idx in enumerate(ranked_indices[:top_k]):
                if 0 <= idx < len(candidates):
                    result = candidates[idx].copy()
                    result['rerank_position'] = rank + 1
                    result['rerank_score'] = (top_k - rank) / top_k
                    reranked.append(result)
            
            logger.info(f"LLM reranking successful: returned {len(reranked)} results")
            return reranked
            
        except Exception as e:
            logger.warning(f"Detailed reranking failed: {str(e)}")
            return candidates[:top_k]
    
    def _fallback_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
        query_analysis: Optional[Dict]
    ) -> List[Dict]:
        """
        Fallback reranking using heuristics when LLM is unavailable
        """
        query_lower = query.lower()
        
        for candidate in candidates:
            score = candidate.get('retrieval_score', 0.5)
            
            # Boost if name matches query words
            name_lower = candidate['name'].lower()
            query_words = set(query_lower.split())
            name_words = set(name_lower.split())
            word_match = len(query_words & name_words) / max(len(query_words), 1)
            score += word_match * 0.2
            
            # Type matching boost
            if query_analysis:
                assessment_types = query_analysis.get('assessment_types', {})
                test_type = candidate.get('test_type', 'General')
                
                if test_type == 'K' and assessment_types.get('technical'):
                    score += 0.15
                if test_type == 'P' and assessment_types.get('behavioral'):
                    score += 0.15
                if test_type == 'C' and assessment_types.get('cognitive'):
                    score += 0.15
            
            candidate['rerank_score'] = min(score, 1.0)
        
        # Sort by rerank score
        reranked = sorted(
            candidates,
            key=lambda x: x.get('rerank_score', 0),
            reverse=True
        )
        
        return reranked[:top_k]


def main():
    """Test reranker"""
    reranker = AssessmentReranker()
    
    # Sample candidates
    candidates = [
        {
            'name': 'Java Programming Test',
            'description': 'Assess Java programming skills',
            'test_type': 'K',
            'category': 'Technical Skills',
            'url': 'https://example.com/java',
            'retrieval_score': 0.85
        },
        {
            'name': 'Teamwork Assessment',
            'description': 'Evaluate collaboration abilities',
            'test_type': 'P',
            'category': 'Personality',
            'url': 'https://example.com/teamwork',
            'retrieval_score': 0.75
        },
        {
            'name': 'Python Coding Test',
            'description': 'Test Python programming',
            'test_type': 'K',
            'category': 'Technical Skills',
            'url': 'https://example.com/python',
            'retrieval_score': 0.70
        }
    ]
    
    query = "Java developer with collaboration skills"
    
    print("\n" + "="*80)
    print("Testing Reranker")
    print("="*80)
    print(f"\nQuery: {query}\n")
    
    reranked = reranker.rerank(query, candidates, top_k=3)
    
    print("Reranked Results:")
    for i, result in enumerate(reranked, 1):
        print(f"{i}. {result['name']}")
        print(f"   Score: {result.get('rerank_score', 0):.3f}")
        print()
    
    print("="*80)


if __name__ == "__main__":
    main()