#!/bin/bash

# Activate virtual environment
source /Users/user/AmazonProject/.venv-2/bin/activate 2>/dev/null || source /Users/user/AmazonProject/.venv/bin/activate 2>/dev/null

# Start the agent in the background
echo "🚀 Starting Dawson Data BI Agent..."
python /Users/user/AmazonProject/dawson_data_agent.py &
AGENT_PID=$!

# Wait for the server to start (2 seconds)
sleep 2

# Open the browser
echo "🌐 Opening browser at http://127.0.0.1:7860"
open http://127.0.0.1:7860

# Keep the script running while agent is active
wait $AGENT_PID
