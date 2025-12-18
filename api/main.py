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
    Fetch and extract text content from a URL (for JD URLs)
    
    Args:
        url: URL to fetch content from
        
    Returns:
        Extracted text content
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Extract text from main content areas
        main_content = soup.find(['main', 'article', 'div'], class_=lambda x: x and ('content' in x.lower() or 'job' in x.lower() or 'description' in x.lower()))
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback: get all text
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean up excessive whitespace
        text = ' '.join(text.split())
        
        logger.info(f"Extracted {len(text)} characters from URL: {url}")
        return text
        
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch content from URL: {str(e)}"
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
        description="List of recommended assessments (5-10)",
        min_items=1,
        max_items=10
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
    5-10 most relevant Individual Test Solutions from SHL catalog.
    
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
            if not query or len(query) < 10:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract meaningful content from the provided URL"
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
            k=30,  # Get more candidates for reranking
            query_analysis=query_analysis
        )
        logger.info(f"Retrieved {len(candidates)} candidates")
        
        # Rerank to get final recommendations
        reranked = reranker.rerank(
            query,
            candidates,
            top_k=10,
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
        for result in reranked[:10]:  # Max 10 recommendations
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