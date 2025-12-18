#!/bin/bash
# Startup script for Render deployment
# Render sets PORT automatically, use it directly
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}

