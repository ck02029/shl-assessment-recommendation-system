# 🚀 Deployment Guide

This guide will help you deploy the SHL Recommendation System to production.

## 📋 Prerequisites

1. **Google Gemini API Key** - Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. **GitHub Account** (for code hosting)
3. **Deployment Platform Account** (Render, Railway, or Heroku)

---

## 🔑 Setting Up API Key

### For Local Development:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Windows CMD
set GEMINI_API_KEY=your-api-key-here

# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"
```

### For Production:
Set the `GEMINI_API_KEY` environment variable in your deployment platform's dashboard.

---

## 🌐 Deployment Options

### Option 1: Render (Recommended - Free Tier Available)

**Steps:**

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Create Render Account**
   - Go to https://render.com
   - Sign up with GitHub

3. **Create New Web Service**
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

4. **Configure Service**
   - **Name**: `shl-recommendation-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

5. **Set Environment Variables**
   - Go to "Environment" tab
   - Add: `GEMINI_API_KEY` = `your-api-key-here`
   - Add: `PYTHON_VERSION` = `3.10.11`

6. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

7. **Update Frontend**
   - Open `frontend/index.html`
   - Change API_URL to your Render URL (e.g., `https://shl-recommendation-api.onrender.com`)
   - Or deploy frontend separately (see below)

**Render URL Format**: `https://your-service-name.onrender.com`

---

### Option 2: Railway (Easy Setup)

**Steps:**

1. **Push code to GitHub** (same as above)

2. **Create Railway Account**
   - Go to https://railway.app
   - Sign up with GitHub

3. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

4. **Configure Environment Variables**
   - Go to "Variables" tab
   - Add: `GEMINI_API_KEY` = `your-api-key-here`

5. **Deploy**
   - Railway auto-detects Python and deploys
   - Get your URL from the dashboard

**Railway URL Format**: `https://your-project-name.up.railway.app`

---

### Option 3: Heroku (Classic Platform)

**Steps:**

1. **Install Heroku CLI**
   - Download from https://devcenter.heroku.com/articles/heroku-cli

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku App**
   ```bash
   heroku create shl-recommendation-api
   ```

4. **Set Environment Variables**
   ```bash
   heroku config:set GEMINI_API_KEY=your-api-key-here
   ```

5. **Deploy**
   ```bash
   git push heroku main
   ```

6. **Open App**
   ```bash
   heroku open
   ```

**Heroku URL Format**: `https://your-app-name.herokuapp.com`

---

## 🎨 Frontend Deployment

### Option A: Deploy Frontend with API (Same Domain)

1. **Update API to serve static files**
   - Add this to `api/main.py`:
   ```python
   from fastapi.staticfiles import StaticFiles
   
   app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
   ```

2. **Access**: Your API URL will serve the frontend at root

### Option B: Deploy Frontend Separately (Netlify/Vercel)

1. **Update frontend/index.html**
   - Change `API_URL` to your deployed API URL

2. **Deploy to Netlify**:
   - Go to https://netlify.com
   - Drag and drop `frontend/` folder
   - Or connect GitHub repo

3. **Deploy to Vercel**:
   - Go to https://vercel.com
   - Import GitHub repository
   - Set root directory to `frontend/`

---

## 🔧 Post-Deployment Checklist

- [ ] API is accessible at `/health` endpoint
- [ ] `GEMINI_API_KEY` is set in environment variables
- [ ] Frontend can connect to API (check browser console)
- [ ] Test a recommendation query
- [ ] Check logs for any errors

---

## 🐛 Troubleshooting

### API Not Starting
- Check Python version (must be 3.10)
- Verify all dependencies in `requirements.txt`
- Check build logs for errors

### API Key Not Working
- Verify `GEMINI_API_KEY` is set in environment variables
- Check API key is valid at https://aistudio.google.com/app/apikey
- Check logs for "No Gemini API key found" warnings

### CORS Errors
- API already has CORS enabled for all origins
- If issues persist, check `api/main.py` CORS settings

### Frontend Can't Connect
- Update `API_URL` in `frontend/index.html`
- Check API is running and accessible
- Verify CORS is enabled

---

## 📝 Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | No* | Google Gemini API key for LLM features |
| `PORT` | Auto | Port number (set by platform) |
| `PYTHON_VERSION` | Auto | Python version (3.10.11) |

*System works without API key but with limited functionality

---

## 🔒 Security Notes

1. **Never commit API keys** to Git
2. **Use environment variables** for all secrets
3. **Add `.env` to `.gitignore`** (already done)
4. **Rotate API keys** if exposed
5. **Use HTTPS** in production (most platforms provide this)

---

## 📊 Monitoring

### Check API Health
```bash
curl https://your-api-url.com/health
```

### View Logs
- **Render**: Dashboard → Logs tab
- **Railway**: Dashboard → Deployments → View Logs
- **Heroku**: `heroku logs --tail`

---

## 🎉 You're Live!

Once deployed, your API will be accessible at:
- `https://your-api-url.com/health` - Health check
- `https://your-api-url.com/docs` - API documentation
- `https://your-api-url.com/recommend` - Recommendation endpoint

Update your frontend to use the new API URL and you're ready to go!

