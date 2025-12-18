# Complete Project Run Instructions

## ⚠️ Important: Python Version Requirement

**This project requires Python 3.10** (not 3.14 or newer)

**Why**: PyTorch and sentence-transformers have compatibility issues with Python 3.14 on Windows, causing DLL initialization errors.

**Check your Python version**:
```bash
py -3.10 --version
```

**If you have Python 3.14 or newer**, you need to:
1. Install Python 3.10 from https://www.python.org/downloads/
2. Use Python 3.10 specifically for this project
3. Create a virtual environment with Python 3.10:
   ```bash
   # Windows
   py -3.10 -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3.10 -m venv venv
   source venv/bin/activate
   ```

---

## 📋 Step-by-Step Execution Guide

### ✅ Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
**Status**: ✅ Already installed

---

### 🔑 Step 1.5: Set Up Gemini API Key (Optional but Recommended)

**Why**: Enables LLM-based query parsing and reranking (better recommendations)

**Get API Key**:
1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Create new API key
4. Copy the key

**Set API Key**:

**Windows PowerShell** (Temporary - for current session):
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Windows CMD** (Temporary - for current session):
```cmd
set GEMINI_API_KEY=your-api-key-here
```

**Windows Permanent Setup**:
1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Click "Advanced" tab → "Environment Variables"
3. Under "User variables", click "New"
4. Variable name: `GEMINI_API_KEY`
5. Variable value: `your-api-key-here`
6. Click OK on all dialogs
7. **Restart terminal/IDE** for changes to take effect

**Linux/Mac** (Add to ~/.bashrc or ~/.zshrc):
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Verify API Key is Set**:
```powershell
# PowerShell
echo $env:GEMINI_API_KEY

# CMD
echo %GEMINI_API_KEY%

# Linux/Mac
echo $GEMINI_API_KEY
```

**Note**: 
- ✅ System works WITHOUT API key (uses keyword-based fallback)
- ✅ System works BETTER WITH API key (uses LLM for smarter recommendations)
- If you see "No Gemini API key found" in logs, it's using fallback mode

---

### ✅ Step 2: Prepare Data

**Current Status**: You have:
- `data/raw_assessments.json` - 54 assessments
- `data/train_data.csv` - Training data  
- `data/test_data.csv` - 10 test queries

**If you need more data**:

**Option A: Create Sample Data** (Quick testing)
```bash
py -3.10 data/create_sample_data.py
```

**Option B: Scrape Real Data** (Recommended - needs 377+)
```bash
py -3.10 scraper/scrape_shl.py
```

---

### ✅ Step 3: Clean Data
```bash
py -3.10 data/clean_data.py --data-dir data
```

**What it does**:
- Removes duplicates
- Validates URLs
- Cleans text fields
- Standardizes test types
- Validates count (warns if < 377)

**Status**: ✅ Already cleaned (54 assessments)

---

### ✅ Step 4: Build Embeddings
```bash
py -3.10 embeddings/build_embeddings.py
```

**What it does**:
- Loads assessments from `data/raw_assessments.json`
- Creates embeddings using SentenceTransformer
- Builds FAISS index for fast search
- Saves to `embeddings/` folder

**Status**: ✅ Already built (files exist)

**Files created**:
- `embeddings/faiss_index.bin`
- `embeddings/metadata.pkl`
- `embeddings/embeddings.npy`

**Note**: If you get PyTorch DLL errors, the embeddings already exist, so you can skip this step.

---

### 🚀 Step 5: Start the API Server

**Method 1: Using Python directly**
```bash
py -3.10 api/main.py
```

**Method 2: Using uvicorn** (Recommended)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**What happens**:
- Server starts on `http://localhost:8000`
- Loads embeddings and ML components
- Ready to accept requests

**Verify it's running**:
- Open browser: `http://localhost:8000/health`
- Should return: `{"status":"healthy"}`

**API Endpoints**:
- `GET /health` - Health check
- `POST /recommend` - Get recommendations
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /stats` - System statistics

---

### 🧪 Step 6: Test the API

**Test 1: Health Check**
```bash
# PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/health"

# Or open in browser
http://localhost:8000/health
```

**Test 2: Get Recommendations** (PowerShell)
```powershell
$body = @{
    query = "Java developer with collaboration skills"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/recommend" `
    -Method POST `
    -Body $body `
    -ContentType "application/json"
```

**Test 3: Use Frontend**
1. Make sure API is running
2. Open `frontend/index.html` in a web browser
3. Enter a query and click "Search"

---

### 📊 Step 7: (Optional) Evaluate Performance
```bash
py -3.10 evaluation/evaluate.py --mode train --k 10
```

**What it does**:
- Evaluates on training data
- Calculates Recall@10 and Precision@10
- Shows retrieval vs final recommendation metrics
- Prints detailed per-query results

---

### 📝 Step 8: Generate Submission CSV
```bash
py -3.10 evaluation/generate_submission.py
```

**What it does**:
- Loads test queries from `data/test_data.csv`
- Generates recommendations for each query
- Creates `submission.csv` with format:
  ```
  Query,Assessment_url
  "query 1",https://www.shl.com/...
  "query 1",https://www.shl.com/...
  ```

**Output**: `submission.csv` in project root

---

## 🎯 Quick Start (All Steps)

**Prerequisites**: Python 3.10 installed and activated

```bash
# 0. Create virtual environment with Python 3.10 (if not already done)
# Windows:
py -3.10 -m venv venv
venv\Scripts\activate

# Linux/Mac:
python3.10 -m venv venv
source venv/bin/activate

# 1. Install dependencies (if needed)
pip install -r requirements.txt

# 2. Create sample data (if needed)
py -3.10 data/create_sample_data.py

# 3. Clean data
py -3.10 data/clean_data.py --data-dir data

# 4. Build embeddings (if needed - skip if files exist)
py -3.10 embeddings/build_embeddings.py

# 5. Start API (in one terminal)
py -3.10 api/main.py

# 6. Test API (in another terminal or browser)
# Open: http://localhost:8000/health
# Or: http://localhost:8000/docs

# 7. Generate submission
py -3.10 evaluation/generate_submission.py
```

---

## 🔧 Common Issues & Solutions

### Issue 1: PyTorch DLL Error
**Error**: `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`

**Root Cause**: Python 3.14 is not compatible with PyTorch on Windows

**Solution**:
1. **Use Python 3.10** (recommended):
   ```bash
   # Install Python 3.10 if needed
   # Then create virtual environment:
   py -3.10 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **If embeddings already exist** (`embeddings/faiss_index.bin`), you can skip building:
   ```bash
   # Skip Step 4 if files already exist
   ```

3. **Alternative**: If you must use current Python, try:
   ```bash
   pip uninstall torch -y
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
   But **Python 3.10 is strongly recommended** for best compatibility.

### Issue 2: API Won't Start
**Error**: Port 8000 already in use

**Solution**:
```bash
# Use different port
uvicorn api.main:app --port 8001
```

### Issue 3: Missing Embeddings
**Error**: `FAISS index not found`

**Solution**:
```bash
py -3.10 embeddings/build_embeddings.py
```

### Issue 4: Missing Data
**Error**: `raw_assessments.json not found`

**Solution**:
```bash
py -3.10 data/create_sample_data.py
```

### Issue 5: API Returns Sample Data
**Message**: "ML components not available, using sample recommendations"

**Solution**:
- Check that embeddings exist
- Check that `embeddings/faiss_index.bin` and `embeddings/metadata.pkl` are present
- Restart API server

---

## 📁 File Structure After Running

```
shl-recommendation-system/
├── data/
│   ├── raw_assessments.json      ✅ Scraped/cleaned assessments
│   ├── train_data.csv            ✅ Training data
│   ├── test_data.csv             ✅ Test queries (10 queries)
│   └── clean_assessments.csv     (Optional) Cleaned CSV
├── embeddings/
│   ├── faiss_index.bin           ✅ FAISS search index
│   ├── metadata.pkl               ✅ Assessment metadata
│   └── embeddings.npy            ✅ Raw embeddings
├── submission.csv                (Generated after Step 8)
├── api/
│   └── main.py                   ✅ API server
└── frontend/
    └── index.html                ✅ Web interface
```

---

## ✅ Verification Checklist

After completing all steps:

- [ ] `data/raw_assessments.json` exists
- [ ] `embeddings/faiss_index.bin` exists
- [ ] `embeddings/metadata.pkl` exists
- [ ] API responds at `http://localhost:8000/health`
- [ ] `/recommend` endpoint works
- [ ] `submission.csv` generated (after Step 8)

---

## 🚀 Running the API

### Start API Server:
```bash
py -3.10 api/main.py
```

**Expected Output**:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test in Browser:
1. Open: `http://localhost:8000/docs`
2. Click on `/recommend` endpoint
3. Click "Try it out"
4. Enter query: `{"query": "Java developer with collaboration skills"}`
5. Click "Execute"
6. See recommendations!

### Test with Frontend:
1. Start API: `py -3.10 api/main.py`
2. Open `frontend/index.html` in browser
3. Enter query and search

---

## 📝 Notes

- **Current Data**: 54 assessments (target is 377+)
- **Test Queries**: 10 queries in `data/test_data.csv`
- **API Port**: Default is 8000
- **LLM Features**: Work without API key (uses fallback), but better with `GEMINI_API_KEY`

---

## 🎉 You're Ready!

Once the API is running, you can:
1. Test queries via API or frontend
2. Generate submission CSV
3. Evaluate performance
4. Deploy to production

