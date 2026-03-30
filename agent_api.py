from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph_agent_2 import run_agent, get_db_connection
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any

app = FastAPI(title="LangGraph Agent API")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, system_prompt, k_value, knowledge_base, namespace, agent_name, init_mssg, first_mssg, show_preset FROM created_agents WHERE id = %s",
        (agent_id,)
    )
    row = cur.fetchone()
    
    if not row:
        cur.close()
        conn.close()
        return None
        
    # Fetch preset questions from the separate table
    cur.execute(
        "SELECT question FROM preset_questions WHERE agent_id = %s",
        (agent_id,)
    )
    questions = [r[0] for r in cur.fetchall()]
    
    cur.close()
    conn.close()
    
    return {
        "user_id": str(row[0]),
        "system_prompt": row[1],
        "k_value": row[2],
        "knowledge_base": row[3],
        "namespace": row[4],
        "agent_name": row[5],
        "init_mssg": row[6],
        "first_mssg": row[7],
        "show_preset": row[8],
        "preset_questions": questions
    }

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


