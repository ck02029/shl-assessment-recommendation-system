"""
FastAPI Backend for SHL Assessment Recommendation System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import os
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ML components conditionally
try:
    from retrieval.retriever import AssessmentRetriever
    from retrieval.reranker import AssessmentReranker
    from llm.query_parser import QueryParser
    ML_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML components not available: {e}")
    AssessmentRetriever = None
    AssessmentReranker = None
    QueryParser = None
    ML_COMPONENTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommends relevant SHL assessments based on job descriptions and queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (lazy loading)
retriever = None
reranker = None
query_parser = None


def fetch_text_from_url(url: str) -> str:
    """
    Fetch and extract text content from any URL
    Universal extraction that works with any website structure
    
    Args:
        url: URL to fetch content from (any website, job board, document, etc.)
        
    Returns:
        Extracted text content
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'https://www.google.com/',
        }
        
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()
        
        # Handle plain text
        if 'text/plain' in content_type:
            text = response.text.strip()
            if len(text) > 50:
                logger.info(f"Extracted {len(text)} characters (plain text) from URL: {url}")
                return text
        
        # Handle HTML content
        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
            # Try to parse as HTML anyway (some sites don't set content-type correctly)
            pass
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements that don't contain meaningful content
        unwanted_tags = ["script", "style", "nav", "header", "footer", "aside", "noscript", 
                        "iframe", "embed", "object", "form", "button", "input", "select",
                        "meta", "link", "base", "svg", "canvas"]
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements with common non-content classes/ids
        non_content_patterns = [
            'nav', 'navigation', 'menu', 'sidebar', 'footer', 'header', 
            'ad', 'advertisement', 'banner', 'cookie', 'popup', 'modal',
            'social', 'share', 'comment', 'related', 'recommended'
        ]
        for pattern in non_content_patterns:
            for elem in soup.find_all(class_=lambda x: x and pattern in str(x).lower()):
                elem.decompose()
            for elem in soup.find_all(id=lambda x: x and pattern in str(x).lower()):
                elem.decompose()
        
        text_parts = []
        
        # Strategy 1: Look for semantic HTML5 elements (best content)
        semantic_elements = soup.find_all(['main', 'article', 'section'])
        for elem in semantic_elements:
            text = elem.get_text(separator=' ', strip=True)
            # Filter out very short or likely navigation content
            if len(text) > 200 and not any(skip in text.lower()[:100] for skip in ['skip to', 'menu', 'navigation']):
                text_parts.append(text)
        
        # Strategy 2: Look for content containers with common patterns
        if not text_parts:
            content_keywords = [
                'content', 'main', 'article', 'post', 'entry', 'body', 
                'description', 'detail', 'text', 'copy', 'story', 'page'
            ]
            
            for keyword in content_keywords:
                # Search by class
                elements = soup.find_all(['div', 'section'], 
                    class_=lambda x: x and keyword in str(x).lower())
                for elem in elements:
                    text = elem.get_text(separator=' ', strip=True)
                    if len(text) > 200:
                        text_parts.append(text)
                        break
                
                # Search by id
                elem = soup.find(id=lambda x: x and keyword in str(x).lower())
                if elem:
                    text = elem.get_text(separator=' ', strip=True)
                    if len(text) > 200:
                        text_parts.append(text)
                        break
                
                if text_parts:
                    break
        
        # Strategy 3: Look for paragraphs and divs with substantial text
        if not text_parts:
            # Find all paragraphs and divs, sort by length
            all_text_elements = soup.find_all(['p', 'div', 'section'])
            text_elements = []
            for elem in all_text_elements:
                text = elem.get_text(separator=' ', strip=True)
                # Skip if too short or looks like navigation/menu
                if len(text) > 100 and not any(skip in text.lower()[:50] for skip in 
                    ['home', 'about', 'contact', 'login', 'sign up', 'menu', 'search']):
                    text_elements.append((len(text), text))
            
            # Sort by length and take top candidates
            text_elements.sort(reverse=True, key=lambda x: x[0])
            for _, text in text_elements[:10]:  # Top 10 longest text blocks
                if len(text) > 200:
                    text_parts.append(text)
        
        # Strategy 4: Fallback - extract from body, removing navigation
        if not text_parts:
            body = soup.find('body')
            if body:
                # Remove common navigation elements
                for nav_tag in ['nav', 'header', 'footer', 'aside', 'menu']:
                    for elem in body.find_all(nav_tag):
                        elem.decompose()
                
                # Get all text from body
                text = body.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    text_parts.append(text)
        
        # Combine all text parts, removing duplicates
        if text_parts:
            # Remove duplicate or very similar content
            unique_texts = []
            seen_texts = set()
            for text in text_parts:
                text_hash = hash(text[:500])  # Hash first 500 chars for comparison
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    unique_texts.append(text)
            
            combined_text = ' '.join(unique_texts)
        else:
            # Last resort: get all text from page
            combined_text = soup.get_text(separator=' ', strip=True)
        
        # Clean up excessive whitespace and normalize
        combined_text = ' '.join(combined_text.split())
        
        # Remove very short content
        if len(combined_text) < 100:
            raise ValueError(f"Extracted content too short ({len(combined_text)} chars), likely not meaningful content")
        
        # Log extraction stats
        logger.info(f"Extracted {len(combined_text)} characters from URL: {url}")
        logger.debug(f"Content preview: {combined_text[:200]}...")
        
        return combined_text
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching URL {url}")
        raise HTTPException(
            status_code=400,
            detail="Request timed out. The URL might be slow or unreachable. Please try again."
        )
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error fetching URL {url}")
        raise HTTPException(
            status_code=400,
            detail="Could not connect to the URL. Please check if the URL is correct and accessible."
        )
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error {e.response.status_code} fetching URL {url}")
        raise HTTPException(
            status_code=400,
            detail=f"HTTP error {e.response.status_code}: Could not access the URL. It might require authentication or be unavailable."
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching URL {url}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch URL: {str(e)}. Please check if the URL is accessible."
        )
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract content from the URL. Error: {str(e)}"
        )


def is_url(text: str) -> bool:
    """Check if input text is a URL"""
    try:
        result = urlparse(text.strip())
        return all([result.scheme, result.netloc])
    except:
        return False


def initialize_components():
    """Initialize ML components on first request"""
    global retriever, reranker, query_parser
    
    if retriever is None:
        logger.info("Initializing recommendation components...")
        if not ML_COMPONENTS_AVAILABLE:
            logger.warning("ML components not available. Using sample recommendations.")
            retriever = None
            reranker = None
            query_parser = None
            return
            
        try:
            retriever = AssessmentRetriever(
                index_path="embeddings/faiss_index.bin",
                metadata_path="embeddings/metadata.pkl"
            )
            reranker = AssessmentReranker()
            query_parser = QueryParser()
            logger.info("Components initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ML components: {str(e)}. Using sample recommendations.")
            retriever = None
            reranker = None
            query_parser = None


# Request/Response Models
class RecommendationRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural language query, full JD text, or JD URL",
        example="I am hiring for Java developers who can also collaborate effectively with my business teams."
    )


class Assessment(BaseModel):
    name: str = Field(..., description="Name of the assessment")
    url: str = Field(..., description="URL to the assessment on SHL website")
    description: str = Field("", description="Description of the assessment")
    duration: str = Field("", description="Duration of the assessment (e.g. '30 minutes')")
    adaptive_support: str = Field("No", description="Whether adaptive testing is supported: 'Yes' or 'No'")
    remote_support: str = Field("Yes", description="Whether remote administration is supported: 'Yes' or 'No'")
    test_type: List[str] = Field(default_factory=list, description="List of test types like ['K'], ['P'], or ['K','P']")


class RecommendationResponse(BaseModel):
    query: str = Field(..., description="The original query")
    recommendations: List[Assessment] = Field(
        ...,
        description="List of recommended assessments (up to 20)",
        min_items=1,
        max_items=20
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns the current status of the API
    """
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: RecommendationRequest):
    """
    Recommend relevant SHL assessments
    
    Takes a natural language query or job description and returns
    up to 20 most relevant Individual Test Solutions from SHL catalog.
    
    Args:
        request: RecommendationRequest with query text
    
    Returns:
        RecommendationResponse with list of recommended assessments
    
    Raises:
        HTTPException: If query is invalid or system error occurs
    """
    try:
        # Initialize components if not already done
        initialize_components()
        
        query = request.query.strip()
        
        # Validate query
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if input is a URL and fetch content
        if is_url(query):
            logger.info(f"Detected URL input, fetching content...")
            query = fetch_text_from_url(query)
            if not query or len(query) < 50:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract meaningful content from the provided URL. Please ensure the URL points to a valid job description page."
                )
        
        if len(query) < 5:
            raise HTTPException(
                status_code=400,
                detail="Query too short. Please provide more details."
            )
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Check if ML components are available
        if retriever is None or reranker is None or query_parser is None:
            logger.info("ML components not available, using sample recommendations")
            # Return sample recommendations
            sample_recommendations = [
                Assessment(
                    name="Java Programming Test",
                    url="https://www.shl.com/solutions/products/java-programming-test/",
                    description="Assess Java programming skills including OOP concepts, algorithms, and debugging.",
                    duration="30 minutes",
                    adaptive_support="Yes",
                    remote_support="Yes",
                    test_type=["K"]
                ),
                Assessment(
                    name="Python Coding Assessment", 
                    url="https://www.shl.com/solutions/products/python-coding-assessment/",
                    description="Evaluate Python coding skills for scripting and data manipulation.",
                    duration="30 minutes",
                    adaptive_support="Yes",
                    remote_support="Yes",
                    test_type=["K"]
                ),
                Assessment(
                    name="Verify Numerical Reasoning",
                    url="https://www.shl.com/solutions/products/verify-numerical-reasoning/",
                    description="Measure numerical reasoning ability using workplace-style data problems.",
                    duration="20 minutes",
                    adaptive_support="No",
                    remote_support="Yes",
                    test_type=["K", "C"]
                ),
                Assessment(
                    name="Situational Judgement Test",
                    url="https://www.shl.com/solutions/products/situational-judgement-test/",
                    description="Assess judgement in realistic workplace scenarios.",
                    duration="25 minutes",
                    adaptive_support="No",
                    remote_support="Yes",
                    test_type=["P"]
                ),
                Assessment(
                    name="Leadership Assessment",
                    url="https://www.shl.com/solutions/products/leadership-assessment/",
                    description="Evaluate leadership potential, decision making, and people management.",
                    duration="45 minutes",
                    adaptive_support="No",
                    remote_support="Yes",
                    test_type=["P"]
                )
            ]
            
            response = RecommendationResponse(
                query=query,
                recommendations=sample_recommendations
            )
            
            logger.info(f"Returning {len(sample_recommendations)} sample recommendations")
            return response
        
        # Parse query with LLM
        query_analysis = query_parser.parse_query(query)
        logger.info(f"Query analysis: {query_analysis}")
        
        # Retrieve candidates
        candidates = retriever.retrieve(
            query,
            k=50,  # Get more candidates for better reranking
            query_analysis=query_analysis
        )
        logger.info(f"Retrieved {len(candidates)} candidates")
        
        # Rerank to get final recommendations
        reranked = reranker.rerank(
            query,
            candidates,
            top_k=20,  # Get top 20 for better recommendations
            query_analysis=query_analysis
        )
        logger.info(f"Reranked to top {len(reranked)} recommendations")
        
        # Ensure we have at least 5 recommendations
        if len(reranked) < 5:
            # If reranking gives too few, fill from candidates
            reranked_urls = {r['url'] for r in reranked}
            for candidate in candidates:
                if candidate['url'] not in reranked_urls:
                    reranked.append(candidate)
                    if len(reranked) >= 5:
                        break
        
        # Format response
        recommendations: List[Assessment] = []
        for result in reranked[:20]:  # Return up to 20 recommendations
            # Map raw metadata to API response schema, with safe defaults
            adaptive_raw = result.get("adaptive")
            remote_raw = result.get("remote")

            adaptive_support = "Yes" if adaptive_raw in [True, 1, "Yes", "yes"] else "No"
            remote_support = "Yes" if remote_raw in [True, 1, "Yes", "yes"] else "Yes"

            test_type_value = result.get("test_type", [])
            if isinstance(test_type_value, str):
                test_type_list = [test_type_value]
            else:
                test_type_list = list(test_type_value) if test_type_value else []

            recommendations.append(
                Assessment(
                    name=result.get("name", ""),
                    url=result.get("url", ""),
                    description=result.get("description", ""),
                    duration=result.get("duration", ""),
                    adaptive_support=adaptive_support,
                    remote_support=remote_support,
                    test_type=test_type_list,
                )
            )
        
        response = RecommendationResponse(
            query=query,
            recommendations=recommendations
        )
        
        logger.info(f"Returning {len(recommendations)} recommendations")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve frontend HTML at root"""
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        # Fallback: return API info if frontend not found
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>SHL Recommendation API</title></head>
        <body>
            <h1>SHL Assessment Recommendation API</h1>
            <p>Version 1.0.0</p>
            <h2>Endpoints:</h2>
            <ul>
                <li><a href="/health">/health</a> - Health check</li>
                <li><a href="/docs">/docs</a> - API Documentation</li>
                <li>/recommend (POST) - Get recommendations</li>
            </ul>
            <p><strong>Note:</strong> Frontend not found. Please ensure frontend/index.html exists.</p>
        </body>
        </html>
        """)


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        initialize_components()
        
        return {
            "total_assessments": len(retriever.assessments),
            "embedding_dimension": retriever.model.get_sentence_embedding_dimension(),
            "components_loaded": {
                "retriever": retriever is not None,
                "reranker": reranker is not None,
                "query_parser": query_parser is not None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/health", "/recommend", "/docs"]
    }


# Mount static files for frontend assets (CSS, JS, etc.)
try:
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
    if os.path.exists(frontend_path):
        # Mount static assets (but not index.html, which is served by root route)
        app.mount("/static", StaticFiles(directory=frontend_path), name="static")
        logger.info(f"Frontend static assets mounted at /static from {frontend_path}")
except Exception as e:
    logger.warning(f"Could not mount frontend static files: {e}")


if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (dev only)
        log_level="info"
    )    
    """
    Process Gen_AI Dataset.xlsx to create training data and assessment catalog
    """
    
    import pandas as pd
    import json
    import logging
    from typing import List, Dict
    import os
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    
    def load_excel_data(filepath: str = "Gen_AI Dataset.xlsx") -> pd.DataFrame:
        """Load data from Excel file"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_excel(filepath)
        logger.info(f"Loaded {len(df)} rows from Excel")
        return df
    
    
    def extract_assessments_from_data(df: pd.DataFrame) -> List[Dict]:
        """Extract unique assessments from the dataset"""
        assessments = []
    
        # Assuming the Excel has columns with assessment names or URLs
        # This is a generic extraction - adjust based on actual Excel structure
    
        assessment_columns = [col for col in df.columns if 'assessment' in col.lower() or 'test' in col.lower()]
    
        if assessment_columns:
            # If there are specific assessment columns
            all_assessments = set()
            for col in assessment_columns:
                values = df[col].dropna().unique()
                all_assessments.update(values)
    
            for assessment_name in all_assessments:
                assessments.append({
                    "name": str(assessment_name),
                    "url": f"https://www.shl.com/solutions/products/{assessment_name.lower().replace(' ', '-')}/",
                    "description": f"Assessment for {assessment_name}",
                    "test_type": "K",  # Assuming knowledge test
                    "category": "General"
                })
        else:
            # Fallback: extract from text fields
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            all_text = ' '.join(df[text_columns].fillna('').astype(str).sum())
    
            # Simple extraction of potential assessment names
            # This would need to be customized based on actual data
            potential_assessments = [
                "Java Programming Test", "Python Coding Assessment", "JavaScript Development Test",
                "SQL Database Assessment", "Excel Data Analysis", "Coding Challenge",
                "Verify Numerical Reasoning", "Verify Verbal Reasoning", "Verify Interactive",
                "OPQ32", "Management Judgement", "Situational Judgement Test",
                "Customer Service Assessment", "Leadership Assessment", "Creativity Assessment"
            ]
    
            for assessment in potential_assessments:
                if assessment.lower() in all_text.lower():
                    assessments.append({
                        "name": assessment,
                        "url": f"https://www.shl.com/solutions/products/{assessment.lower().replace(' ', '-')}/",
                        "description": f"Assessment for {assessment}",
                        "test_type": "K",
                        "category": "General"
                    })
    
        logger.info(f"Extracted {len(assessments)} unique assessments")
        return assessments
    
    
    def create_training_data(df: pd.DataFrame) -> pd.DataFrame:
        """Create training data with query-assessment pairs"""
        train_data = []
    
        # Assuming the Excel has a 'Query' column and assessment columns
        query_col = None
        for col in df.columns:
            if 'query' in col.lower() or 'description' in col.lower() or 'job' in col.lower():
                query_col = col
                break
    
        if query_col is None:
            logger.warning("No query column found, using first text column")
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            query_col = text_cols[0] if text_cols else df.columns[0]
    
        assessment_cols = [col for col in df.columns if col != query_col and ('assessment' in col.lower() or 'test' in col.lower())]
    
        for idx, row in df.iterrows():
            query = str(row[query_col]).strip()
            if not query or query.lower() == 'nan':
                continue
    
            # For each assessment column, create labeled pairs
            for col in assessment_cols:
                assessment = str(row[col]).strip()
                if assessment and assessment.lower() != 'nan':
                    train_data.append({
                        'Query': query,
                        'Assessment': assessment,
                        'Label': 1  # Assuming positive label
                    })
    
            # If no specific assessment columns, create pairs with all assessments
            if not assessment_cols:
                # This would need customization based on actual data structure
                pass
    
        train_df = pd.DataFrame(train_data)
        logger.info(f"Created {len(train_df)} training pairs")
        return train_df
    
    
    def create_test_data(df: pd.DataFrame) -> pd.DataFrame:
        """Create test data (unlabeled queries)"""
        # Use a subset of queries for testing
        query_col = None
        for col in df.columns:
            if 'query' in col.lower() or 'description' in col.lower() or 'job' in col.lower():
                query_col = col
                break
    
        if query_col is None:
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            query_col = text_cols[0] if text_cols else df.columns[0]
    
        test_queries = df[query_col].dropna().unique()[:10]  # First 10 unique queries
    
        test_df = pd.DataFrame({'Query': test_queries})
        logger.info(f"Created {len(test_df)} test queries")
        return test_df
    
    
    def main():
        """Main processing function"""
        try:
            # Load Excel data
            df = load_excel_data()
    
            # Create assessments catalog
            assessments = extract_assessments_from_data(df)
    
            # Save assessments to JSON
            with open('data/raw_assessments.json', 'w', encoding='utf-8') as f:
                json.dump(assessments, f, indent=2, ensure_ascii=False)
            logger.info("✓ Saved raw_assessments.json")
    
            # Create training data
            train_df = create_training_data(df)
            train_df.to_csv('data/train_data.csv', index=False)
            logger.info("✓ Saved train_data.csv")
    
            # Create test data
            test_df = create_test_data(df)
            test_df.to_csv('data/test_data.csv', index=False)
            logger.info("✓ Saved test_data.csv")
    
            print("\n" + "="*60)
            print("DATA PROCESSING COMPLETE")
            print("="*60)
            print(f"Files created in data/ folder:")
            print(f"  - raw_assessments.json ({len(assessments)} assessments)")
            print(f"  - train_data.csv ({len(train_df)} training pairs)")
            print(f"  - test_data.csv ({len(test_df)} test queries)")
            print("="*60)
    
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
    
    
    if __name__ == "__main__":
        main()