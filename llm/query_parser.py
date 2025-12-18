"""
LLM-based Query Parser
Extracts structured information from job queries using Gemini API
"""

import os
import json
import re
from typing import Dict, List, Optional
import logging
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryParser:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize QueryParser with Gemini API
        
        Args:
            api_key: Google Gemini API key (or set GEMINI_API_KEY env variable)
        """
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("No Gemini API key found. Parser will work in limited mode.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini model initialized")
    
    def parse_query(self, query: str) -> Dict:
        """
        Parse job query and extract structured information
        
        Returns:
            Dict with:
                - technical_skills: List of technical skills required
                - soft_skills: List of soft/behavioral skills required
                - seniority: Job seniority level
                - domains: Relevant domains/industries
                - assessment_types: Needed assessment types
        """
        if not self.model:
            return self._fallback_parse(query)
        
        try:
            prompt = f"""Analyze this job description or hiring query and extract structured information.

Query: "{query}"

Extract and return ONLY a JSON object (no other text) with these fields:
{{
  "technical_skills": ["list of specific technical skills like Java, Python, SQL, etc."],
  "soft_skills": ["list of soft skills like collaboration, leadership, communication, etc."],
  "seniority": "junior/mid/senior/unspecified",
  "domains": ["list of relevant domains like software development, data analysis, etc."],
  "assessment_types": {{
    "technical": true/false,
    "behavioral": true/false,
    "cognitive": true/false
  }},
  "key_requirements": ["main requirements from the query"]
}}

Be specific and extract actual skills mentioned. Return ONLY valid JSON, no markdown formatting."""

            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up markdown formatting if present
            result_text = re.sub(r'```json\s*|\s*```', '', result_text)
            result_text = result_text.strip()
            
            # Parse JSON
            parsed = json.loads(result_text)
            
            logger.info(f"Successfully parsed query using LLM")
            return parsed
            
        except Exception as e:
            logger.warning(f"LLM parsing failed: {str(e)}. Using fallback.")
            return self._fallback_parse(query)
    
    def _fallback_parse(self, query: str) -> Dict:
        """
        Fallback parser using keyword matching when LLM is unavailable
        """
        query_lower = query.lower()
        
        # Technical skills keywords
        technical_skills = []
        tech_keywords = {
            'java': 'Java',
            'python': 'Python',
            'javascript': 'JavaScript',
            'sql': 'SQL',
            'c++': 'C++',
            'c#': 'C#',
            'ruby': 'Ruby',
            'php': 'PHP',
            'swift': 'Swift',
            'kotlin': 'Kotlin',
            'react': 'React',
            'angular': 'Angular',
            'vue': 'Vue.js',
            'node': 'Node.js',
            'excel': 'Excel',
            'powerpoint': 'PowerPoint',
            'data analysis': 'Data Analysis',
            'machine learning': 'Machine Learning',
            'ai': 'AI'
        }
        
        for keyword, skill in tech_keywords.items():
            if keyword in query_lower:
                technical_skills.append(skill)
        
        # Soft skills keywords
        soft_skills = []
        soft_keywords = {
            'collaboration': 'Collaboration',
            'communicate': 'Communication',
            'communication': 'Communication',
            'leadership': 'Leadership',
            'teamwork': 'Teamwork',
            'team work': 'Teamwork',
            'problem solving': 'Problem Solving',
            'critical thinking': 'Critical Thinking',
            'creativity': 'Creativity',
            'adaptability': 'Adaptability',
            'interpersonal': 'Interpersonal Skills'
        }
        
        for keyword, skill in soft_keywords.items():
            if keyword in query_lower:
                soft_skills.append(skill)
        
        # Detect seniority
        seniority = 'unspecified'
        if any(word in query_lower for word in ['junior', 'entry', 'entry-level']):
            seniority = 'junior'
        elif any(word in query_lower for word in ['mid-level', 'mid level', 'intermediate']):
            seniority = 'mid'
        elif any(word in query_lower for word in ['senior', 'lead', 'principal', 'architect']):
            seniority = 'senior'
        
        # Detect domains
        domains = []
        domain_keywords = {
            'software development': ['developer', 'programmer', 'engineer', 'coding'],
            'data analysis': ['analyst', 'data', 'analytics'],
            'management': ['manager', 'management', 'director'],
            'sales': ['sales', 'account executive'],
            'marketing': ['marketing', 'brand'],
            'customer service': ['customer service', 'support']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                domains.append(domain)
        
        # Determine assessment types needed
        assessment_types = {
            'technical': len(technical_skills) > 0,
            'behavioral': len(soft_skills) > 0,
            'cognitive': any(word in query_lower for word in ['cognitive', 'reasoning', 'analytical'])
        }
        
        return {
            'technical_skills': technical_skills,
            'soft_skills': soft_skills,
            'seniority': seniority,
            'domains': domains,
            'assessment_types': assessment_types,
            'key_requirements': technical_skills + soft_skills
        }
    
    def should_balance_types(self, parsed_query: Dict) -> bool:
        """
        Determine if results should be balanced between technical and behavioral
        """
        assessment_types = parsed_query.get('assessment_types', {})
        technical = assessment_types.get('technical', False)
        behavioral = assessment_types.get('behavioral', False)
        
        return technical and behavioral


def main():
    """Test query parser"""
    parser = QueryParser()
    
    test_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams.",
        "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.",
        "Need an analyst with cognitive and personality tests",
        "Senior leadership position requiring strategic thinking and team management"
    ]
    
    print("\n" + "="*80)
    print("Testing Query Parser")
    print("="*80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        parsed = parser.parse_query(query)
        
        print(json.dumps(parsed, indent=2))
        print()
    
    print("="*80)


if __name__ == "__main__":
    main()