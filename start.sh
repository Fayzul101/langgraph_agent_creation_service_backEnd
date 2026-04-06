#!/bin/bash

echo "Starting KB Service on port 8000..."
python3 kb_service.py &
KB_SERVICE_PID=$!

echo "Starting Agent API on port 8001..."
python3 agent_api.py &
AGENT_API_PID=$!

echo "Copying nginx config..."
cp nginx/nginx.conf /usr/local/etc/nginx/nginx.conf

echo "Starting Nginx on port 8080..."
nginx

echo ""
echo "All services started!"
echo "KB Service  → http://localhost:8080/kb_service/"
echo "Agent API   → http://localhost:8080/agent_api/"
echo ""
echo "KB Service PID: $KB_SERVICE_PID"
echo "Agent API PID:  $AGENT_API_PID"

# Save PIDs for stop script
echo $KB_SERVICE_PID > .kb_service.pid
echo $AGENT_API_PID > .agent_api.pid