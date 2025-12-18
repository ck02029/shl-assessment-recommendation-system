# ⚡ Quick Deployment Guide

## 🚀 Fastest Way to Deploy (Render - Free)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add deployment configuration"
git push origin main
```

### Step 2: Deploy on Render
1. Go to https://render.com
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Connect your repository
5. Settings:
   - **Name**: `shl-recommendation-api`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
6. Add Environment Variable:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: Your API key from https://aistudio.google.com/app/apikey
7. Click "Create Web Service"
8. Wait 5-10 minutes for deployment

### Step 3: Get Your URL
Your API will be at: `https://shl-recommendation-api.onrender.com`

### Step 4: Update Frontend (Optional)
The frontend will auto-detect the API URL, but you can manually set it in `frontend/index.html`:
```javascript
const API_URL = 'https://your-api-url.onrender.com';
```

## ✅ That's It!

Your API is now live! Test it:
- Health: `https://your-api-url.onrender.com/health`
- Docs: `https://your-api-url.onrender.com/docs`
- Frontend: `https://your-api-url.onrender.com/` (if static files are served)

## 🔑 Important Notes

1. **API Key**: Set `GEMINI_API_KEY` in Render dashboard (Environment tab)
2. **Free Tier**: Render free tier sleeps after 15 min inactivity (first request may be slow)
3. **Upgrade**: For always-on service, upgrade to paid plan

## 📚 Full Guide

See `DEPLOYMENT.md` for detailed instructions on other platforms (Railway, Heroku, etc.)

