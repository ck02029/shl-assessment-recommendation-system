#!/usr/bin/env python3
"""
Startup script for production deployment
Reads PORT from environment variable (set by Render/Railway/etc.)
"""
import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get("PORT", 8000))
    
    # Run the API
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

