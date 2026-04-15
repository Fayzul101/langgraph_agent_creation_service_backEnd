from fastapi import FastAPI, HTTPException, UploadFile, File, Form, APIRouter
from pydantic import BaseModel
from langgraph_agent import run_agent
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import re
import tempfile
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv(dotenv_path=".env")

app = FastAPI(title="LangGraph Unified API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Configuration ──────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize shared tools
embedding_tool = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

# ─── Agent API Models ───────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    k: int
    namespace: str
    user_id: str
    thread_id: str
    kb_name: str
    system_prompt: str

class WidgetChatRequest(BaseModel):
    agent_id: str
    message: str
    session_id: str

class PromptGenerationRequest(BaseModel):
    description: str

# ─── Agent API Helper Functions ──────────────────────────────────────────────
def get_agent_by_id(agent_id: str):
    agent_url = f"{SUPABASE_URL}/rest/v1/created_agents"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"
    }
    params = {
        "agent_id": f"eq.{agent_id}",
        "select": "user_id,system_prompt,k_value,knowledge_base,namespace,agent_name,init_mssg,first_mssg,show_preset"
    }
    
    try:
        response = requests.get(agent_url, headers=headers, params=params)
        if response.status_code != 200 or not response.json():
            return None
        
        row = response.json()[0]
        
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

# ─── Agent API Endpoints ────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "LangGraph Unified API Service is running"}

@app.get("/widget/config/{agent_id}")
async def widget_config(agent_id: str):
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
        response = run_agent(
            query=request.query,
            meth_k=request.k,
            meth_namespace=request.namespace,
            meth_uuid=request.user_id,
            thread_id=request.thread_id,
            meth_kb=request.kb_name,
            meth_sys_prompt=request.system_prompt
        )
        
        serializable_messages = []
        for msg in response.get("messages", []):
            role = "user" if hasattr(msg, "type") and msg.type == "human" else "bot"
            content = msg.content
            
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

@app.post("/generate-system-prompt")
async def generate_system_prompt(request: PromptGenerationRequest):
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=OPENAI_API_KEY,
        )
        
        prompt_template = ChatPromptTemplate.from_template(
            "You are an expert at creating system prompts for AI agents. "
            "Based on the following description, generate a comprehensive and effective system prompt for an agent: "
            "{description}\n\n"
            "Return ONLY the system prompt text, without any additional formatting or explanations."
        )
        
        chain = prompt_template | llm
        response = chain.invoke({"description": request.description})
        
        return {"system_prompt": response.content}
    except Exception as e:
        print(f"Error in generate-system-prompt endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("")
async def process_kb(
    files: List[UploadFile] = File(...),
    kb_name: str = Form(...),
    knowledge_name: str = Form(...),
    user_id: str = Form(...)
):
    temp_files = []
    try:
        for file in files:
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                temp_files.append(tmp.name)
            
            documents = []
            if suffix.lower() == ".pdf":
                loader = PyPDFLoader(file_path=tmp.name)
                documents = loader.load()
            elif suffix.lower() in [".txt", ".md", ".csv"]:
                loader = TextLoader(file_path=tmp.name, encoding="utf-8")
                documents = loader.load()
            else:
                print(f"Skipping unsupported file type: {suffix}")
                continue

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)
            
            all_chunks = []
            for doc in documents:
                text = doc.page_content
                text = re.sub(r" {2,}", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"[ \t]+\n", "\n", text)
                chunks = text_splitter.split_text(text.strip())
                all_chunks.extend(chunks)

            url = f"{SUPABASE_URL}/rest/v1/knowledge_base"
            headers = {
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            }
            
            payload = []
            for chunk_text in all_chunks:
                vector = embedding_tool.embed_query(chunk_text)
                payload.append({
                    "content": chunk_text,
                    "name_metadata": knowledge_name,
                    "knowledge_base_name": kb_name,
                    "vector_content": vector,
                    "user_id": user_id
                })
            
            if payload:
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code not in [200, 201, 204]:
                    print(f"Supabase Insert Error: {response.status_code} - {response.text}")
                    raise Exception(f"Failed to insert chunks into Supabase: {response.text}")

        return {"message": f"{len(files)} files processed and uploaded successfully!"}

    except Exception as e:
        print(f"Error in KB process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

@app.post("/kb")
async def process_kb(
    files: List[UploadFile] = File(...),
    kb_name: str = Form(...),
    knowledge_name: str = Form(...),
    user_id: str = Form(...)
):
    temp_files = []
    try:
        # Loop through each uploaded file
        for file in files:
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                temp_files.append(tmp.name)
            
            # Load Documents per file
            documents = []
            if suffix.lower() == ".pdf":
                loader = PyPDFLoader(file_path=tmp.name)
                documents = loader.load()
            elif suffix.lower() in [".txt", ".md", ".csv"]:
                loader = TextLoader(file_path=tmp.name, encoding="utf-8")
                documents = loader.load()
            else:
                # Add warning but continue with other files
                print(f"Skipping unsupported file type: {suffix}")
                continue

            # Clean and Chunk Text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)
            
            all_chunks = []
            for doc in documents:
                text = doc.page_content
                text = re.sub(r" {2,}", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"[ \t]+\n", "\n", text)
                chunks = text_splitter.split_text(text.strip())
                all_chunks.extend(chunks)

            # Insert into Supabase for this file
            url = f"{SUPABASE_URL}/rest/v1/knowledge_base"
            headers = {
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            }
            
            payload = []
            for chunk_text in all_chunks:
                vector = embedding_tool.embed_query(chunk_text)
                payload.append({
                    "content": chunk_text,
                    "name_metadata": knowledge_name,
                    "knowledge_base_name": kb_name,
                    "vector_content": vector,
                    "user_id": user_id
                })
            
            if payload:
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code not in [200, 201, 204]:
                    print(f"Supabase Insert Error: {response.status_code} - {response.text}")
                    raise Exception(f"Failed to insert chunks into Supabase: {response.text}")

        return {"message": f"{len(files)} files processed and uploaded successfully!"}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
