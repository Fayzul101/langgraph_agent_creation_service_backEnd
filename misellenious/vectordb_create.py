import sys
import argparse
import os
import re
import hashlib
import psycopg
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv(dotenv_path=".env")

# ─── Argument Parsing ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Process document for Vector DB")
parser.add_argument("--file_path", required=True, help="Path to the file to process")
parser.add_argument("--kb_name", required=True, help="Knowledge Base Name")
parser.add_argument("--knowledge_name", required=True, help="Name of the Knowledge")
parser.add_argument("--user_id", required=True, help="User ID")
args = parser.parse_args()

KNOWLEDGE_BASE_NAME = args.kb_name
NAME_METADATA = args.knowledge_name
USER_ID = args.user_id
FILE_PATH = args.file_path

# ─── PostgreSQL Configuration ───────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5434")
DB_NAME = os.getenv("DB_NAME", "langgraph_users")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ─── Load Documents ─────────────────────────────────────────────────────────
all_documents = []
filename = os.path.basename(FILE_PATH)

try:
    if filename.lower().endswith(".pdf"):
        print(f"Loading PDF: {filename}")
        loader = PyPDFLoader(file_path=FILE_PATH)
        documents = loader.load()
    elif filename.lower().endswith((".txt", ".md", ".csv")):
        print(f"Loading text file: {filename}")
        loader = TextLoader(file_path=FILE_PATH, encoding="utf-8")
        documents = loader.load()
    else:
        print(f"Unsupported file type: {filename}")
        sys.exit(1)

    all_documents.extend(documents)
except Exception as e:
    print(f"Error loading {filename}: {str(e)}")
    sys.exit(1)

# ─── Clean Text ─────────────────────────────────────────────────────────────
for doc in all_documents:
    text = doc.page_content
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    doc.page_content = text.strip()

# ─── Chunk Documents ────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=60
)
chunks = text_splitter.split_documents(all_documents)

# ─── Initialize Embeddings ───────────────────────────────────────────────────
embedding_tool = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

# ─── Insert into Database ───────────────────────────────────────────────────
try:
    with psycopg.connect(CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                # Generate vector embedding for the chunk
                vector = embedding_tool.embed_query(chunk.page_content)
                
                # Insert chunk data into the knowledge_base table
                cur.execute(
                    """
                    INSERT INTO knowledge_base (content, name_metadata, knowledge_base_name, vector_content, user_id)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (chunk.page_content, NAME_METADATA, KNOWLEDGE_BASE_NAME, vector, USER_ID)
                )
        conn.commit()
    print("✓ Knowledge chunks and vectors uploaded successfully!")
except Exception as e:
    print(f"Database error: {str(e)}")
    sys.exit(1)