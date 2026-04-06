#!/bin/bash

echo "Stopping Nginx..."
nginx -s stop

echo "Stopping KB Service..."
if [ -f .kb_service.pid ]; then
    kill $(cat .kb_service.pid) && rm .kb_service.pid
fi

echo "Stopping Agent API..."
if [ -f .agent_api.pid ]; then
    kill $(cat .agent_api.pid) && rm .agent_api.pid
fi

echo "All services stopped."