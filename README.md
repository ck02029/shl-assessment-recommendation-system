# SHL Assessment Recommendation System

A machine learning-powered system that recommends relevant SHL (Saville & Holdsworth Ltd) assessments based on job descriptions and hiring requirements. The system uses semantic search, embeddings, and LLM-based reranking to provide personalized assessment suggestions.

## Features

- **Semantic Search**: Uses Sentence Transformers and FAISS for efficient retrieval of relevant assessments
- **LLM-Powered Analysis**: Integrates Google Gemini for query parsing and result reranking
- **REST API**: FastAPI backend for scalable recommendations
- **Web Interface**: Simple HTML frontend for easy interaction
- **Evaluation Framework**: Comprehensive evaluation metrics for system performance

## Project Structure

### Root Files
- `process_data.py` - Processes Excel data to create training/test datasets and assessment catalog
- `requirements.txt` - Python dependencies
- `README.md` - This file

### api/
- `main.py` - FastAPI server with recommendation endpoints
- `__pycache__/` - Python bytecode cache

### data/
- `raw_assessments.json` - Assessment catalog with metadata
- `train_data.csv` - Training data for model evaluation
- `test_data.csv` - Test queries for evaluation
- `clean_data.py` - Data cleaning utilities
- `create_sample_data.py` - Sample data generation
- `__pycache__/` - Python bytecode cache

### embeddings/
- `build_embeddings.py` - Creates FAISS index and embeddings for assessments
- `embeddings.npy` - Pre-computed assessment embeddings
- `__pycache__/` - Python bytecode cache

### evaluation/
- `evaluate.py` - Main evaluation script with metrics calculation
- `generate_submission.py` - Submission file generation for competitions
- `load_eval_data.py` - Evaluation data loading utilities
- `recall.py` - Recall and ranking metrics
- `__pycache__/` - Python bytecode cache

### frontend/
- `index.html` - Web interface for the recommendation system

### llm/
- `query_parser.py` - LLM-based query analysis and parsing
- `__pycache__/` - Python bytecode cache

### retrieval/
- `retriever.py` - FAISS-based assessment retrieval
- `reranker.py` - LLM-based result reranking
- `__pycache__/` - Python bytecode cache

### scraper/
- `scrape_shl.py` - Web scraper for SHL assessment data
- `__pycache__/` - Python bytecode cache

## Installation

**⚠️ Python Version Requirement**: This project requires **Python 3.10** (not 3.14 or newer) due to PyTorch compatibility on Windows.

1. **Install Python 3.10** if you don't have it:
   - Download from https://www.python.org/downloads/
   - Or use `py -3.10` on Windows if multiple versions are installed

2. **Create a virtual environment** (recommended):
   ```bash
   # Windows
   py -3.10 -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. **Set API Key**: Get a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   ```bash
   # Windows Command Prompt
   set GEMINI_API_KEY=your_api_key_here
   
   # Windows PowerShell
   $env:GEMINI_API_KEY = "your_api_key_here"
   ```

2. **Process Data** (optional - data already exists):
   ```bash
   python process_data.py
   ```

3. **Build Embeddings**:
   ```bash
   python embeddings/build_embeddings.py
   ```

4. **Start the API Server**:
   ```bash
   python api/main.py
   ```
   The server runs on `http://localhost:8000`

5. **Access Frontend**:
   Open `frontend/index.html` in your web browser

## Usage

### Web Interface
1. Open `frontend/index.html` in a browser
2. Enter a job description (e.g., "Java developer with collaboration skills")
3. Click "Search" to get recommendations

### API Usage
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "Python developer with SQL skills"}'
```

### Example Queries
- "Java developer with 5 years experience"
- "Data analyst with Excel proficiency"
- "Team leader with communication skills"
- "Software engineer with cloud experience"

## Evaluation

The system includes comprehensive evaluation metrics:

### Running Evaluation
```bash
python evaluation/evaluate.py
```

This will:
- Load test queries from `data/test_data.csv`
- Generate recommendations for each query
- Calculate metrics like Mean Reciprocal Rank (MRR), Precision@K, Recall@K
- Output performance statistics

### Evaluation Metrics
- **Retrieval Metrics**: Precision, Recall, F1-Score at different K values
- **Ranking Metrics**: Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG)
- **Relevance Assessment**: Based on labeled training data

### Custom Evaluation
Modify `evaluation/evaluate.py` to test with your own queries or datasets.

## Architecture

1. **Query Processing**: Input query is parsed using LLM to extract skills, seniority, domains
2. **Retrieval**: FAISS searches for semantically similar assessments using embeddings
3. **Reranking**: LLM scores and reorders results for better relevance
4. **Response**: Top recommendations returned via API

## Dependencies

- `pandas` - Data processing
- `sentence-transformers` - Text embeddings
- `faiss-cpu` - Vector search
- `fastapi` - Web API framework
- `uvicorn` - ASGI server
- `google-generativeai` - LLM integration

## Troubleshooting

### API Key Issues
- Ensure `GEMINI_API_KEY` is set correctly
- Check API quota and billing status
- Verify key has Generative AI API access

### Embedding Issues
- Ensure `embeddings/embeddings.npy` exists
- Re-run `python embeddings/build_embeddings.py` if missing
- **PyTorch DLL Error**: If you get `OSError: [WinError 1114]`, use Python 3.10 instead of 3.14

### Frontend Issues
- API must be running on port 8000
- Check browser console for CORS or network errors

### Performance Issues
- For large datasets, consider GPU acceleration for embeddings
- Adjust FAISS parameters in `retriever.py` for better speed/accuracy tradeoffs

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure compatibility with existing evaluation metrics

