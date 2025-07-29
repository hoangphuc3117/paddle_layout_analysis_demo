#!/bin/bash

# Railway start script for Streamlit app
# This script ensures proper configuration for Railway deployment

# Set environment variables for Railway
export PORT=${PORT:-8501}
export STREAMLIT_SERVER_PORT=$PORT
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"

# Create output directory if it doesn't exist
mkdir -p output

# Start Streamlit
streamlit run web_demo.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
