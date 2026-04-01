from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import tempfile
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv(dotenv_path=".env")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Supabase Configuration ─────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("Warning: Supabase credentials are missing!")

# ─── Initialize Embeddings ───────────────────────────────────────────────────
embedding_tool = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

@app.get("/")
async def root():
    return {"message": "Knowledge Base Service is running"}

from typing import List

@app.post("/process-kb")
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
