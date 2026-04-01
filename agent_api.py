from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph_agent_2 import run_agent
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv(dotenv_path=".env")

app = FastAPI(title="LangGraph Agent API")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

class ChatRequest(BaseModel):
    query: str
    k: int
    namespace: str
    user_id: str
    thread_id: str
    kb_name: str
    system_prompt: str

class WidgetChatRequest(BaseModel):
    agent_id: int
    message: str
    session_id: str

def get_agent_by_id(agent_id: int):
    # Fetch agent details
    agent_url = f"{SUPABASE_URL}/rest/v1/created_agents"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"
    }
    params = {
        "id": f"eq.{agent_id}",
        "select": "user_id,system_prompt,k_value,knowledge_base,namespace,agent_name,init_mssg,first_mssg,show_preset"
    }
    
    try:
        response = requests.get(agent_url, headers=headers, params=params)
        if response.status_code != 200 or not response.json():
            return None
        
        row = response.json()[0]
        
        # Fetch preset questions
        questions_url = f"{SUPABASE_URL}/rest/v1/preset_questions"
        q_params = {
            "agent_id": f"eq.{agent_id}",
            "select": "question"
        }
        q_response = requests.get(questions_url, headers=headers, params=q_params)
        questions = [r["question"] for r in q_response.json()] if q_response.status_code == 200 else []
        
        return {
            "user_id": str(row["user_id"]),
            "system_prompt": row["system_prompt"],
            "k_value": row["k_value"],
            "knowledge_base": row["knowledge_base"],
            "namespace": row["namespace"],
            "agent_name": row["agent_name"],
            "init_mssg": row["init_mssg"],
            "first_mssg": row["first_mssg"],
            "show_preset": row["show_preset"],
            "preset_questions": questions
        }
    except Exception as e:
        print(f"Error fetching agent from Supabase: {str(e)}")
        return None

@app.get("/")
async def root():
    return {"message": "Agent API Service is running"}

@app.get("/widget/config/{agent_id}")
async def widget_config(agent_id: int):
    agent = get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {
        "agent_name": agent["agent_name"],
        "init_mssg": agent["init_mssg"],
        "first_mssg": agent["first_mssg"],
        "show_preset": agent["show_preset"],
        "preset_questions": agent["preset_questions"]
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Call the run_agent function from langgraph_agent.py
        response = run_agent(
            query=request.query,
            meth_k=request.k,
            meth_namespace=request.namespace,
            meth_uuid=request.user_id,
            thread_id=request.thread_id,
            meth_kb=request.kb_name,
            meth_sys_prompt=request.system_prompt
        )
        
        # Prepare the response to be JSON serializable
        serializable_messages = []
        for msg in response.get("messages", []):
            role = "user" if hasattr(msg, "type") and msg.type == "human" else "bot"
            content = msg.content
            
            # Remove "System_response:" if it exists at the start
            if role == "bot" and content.startswith("System_response:"):
                content = content.replace("System_response:", "").strip()
            
            serializable_messages.append({
                "role": role,
                "content": content
            })
            
        return {
            "status": "success",
            "messages": serializable_messages
        }
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# In agent_api.py
@app.post("/widget/chat")
async def widget_chat(body: WidgetChatRequest):
    try:
        agent = get_agent_by_id(body.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        response = run_agent(
            query=body.message,
            meth_k=agent["k_value"],
            meth_namespace=agent["namespace"],
            meth_uuid=agent["user_id"],
            thread_id=body.session_id,
            meth_kb=agent["knowledge_base"],
            meth_sys_prompt=agent["system_prompt"]
        )
        
        # Extract the bot's response message
        bot_message = ""
        for msg in response.get("messages", []):
            role = "user" if hasattr(msg, "type") and msg.type == "human" else "bot"
            if role == "bot":
                content = msg.content
                if content.startswith("System_response:"):
                    content = content.replace("System_response:", "").strip()
                bot_message = content
                
        return {"response": bot_message}
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in widget/chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)


