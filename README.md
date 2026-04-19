# LangGraph Console 🤖

A full-stack, self-hosted platform for building, managing, and chatting with AI agents powered by **LangGraph**, **LangChain**, and **OpenAI**. It provides a clean management console to upload documents into knowledge bases, configure agents with custom personas, and have multi-turn, Retrieval-Augmented Generation (RAG) conversations.

---

## ✨ Features

| Feature | Description |
|---|---|
| **User Authentication** | Secure signup/login with `bcrypt`-hashed passwords stored in PostgreSQL |
| **Knowledge Base Management** | Upload multiple PDF, TXT, MD, or CSV files; they are chunked, embedded, and stored as vectors |
| **Agent Configuration** | Create agents with a name, system prompt, knowledge base, namespace, and retrieval `K` value |
| **RAG Chat Interface** | Multi-turn chat with context-aware responses via LangGraph workflows |
| **Session Management** | Each chat session generates a fresh UUID; sessions can be reset without page reload |
| **Agent Settings** | Live-edit agent parameters from within the chat page via a settings modal |
| **Dashboard Overview** | Real-time stats showing total knowledge bases and active agents |
| **Database Inspector** | Browse raw database chunks from any knowledge base |
| **Dark Mode** | Full light/dark theme support across all pages |
| **Responsive Design** | Collapsible sidebar and mobile-friendly layout |

---

## 🏗️ Architecture

The system consists of the Next.js Frontend and a unified Python backend:

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (User)                       │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP
                        ▼
┌─────────────────────────────────────────────────────────┐
│          Next.js App (Port 3000)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  /dashboard  │  │  /api/kb/*   │  │  /api/agents │  │
│  │  /agent/[id] │  │  /api/auth/* │  │  /api/dash.. │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────┬──────────────────────────────┘
                          │
              ┌───────────▼────────────────────────┐
              │        langgraph-api.py            │
              │          (Port 8002)               │
              │   • PDF Parse   • LangGraph        │
              │   • Chunk Text  • RAG Retrieval    │
              │   • Embed+Save  • Chat Response    │
              └───────────┬────────────────────────┘
                      │
                      └──────────────────────────┐
                                 ▼
              ┌──────────────────────────────────┐
              │   PostgreSQL Database            │
              │   • user                         │
              │   • created_agents               │
              │   • knowledge_base (+ pgvector)  │
              └──────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Frontend
| Technology | Version | Purpose |
|---|---|---|
| [Next.js](https://nextjs.org/) | 16.1.6 | App framework (App Router) |
| [React](https://react.dev/) | 19 | UI Library |
| [Tailwind CSS](https://tailwindcss.com/) | 4 | Styling |
| [Lucide React](https://lucide.dev/) | 0.575 | Icons |
| [node-postgres (pg)](https://node-postgres.com/) | 8 | Direct DB access for Next.js API routes |
| [bcrypt](https://github.com/kelektiv/node.bcrypt.js) | 6 | Password hashing |

### Backend (Python)
| Technology | Purpose |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | REST API framework for both Python services |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful agent workflow management |
| [LangChain](https://www.langchain.com/) | LLM orchestration, prompt templates, document loaders |
| [OpenAI GPT-4o](https://platform.openai.com/) | LLM for chat responses |
| [OpenAI `text-embedding-3-small`](https://platform.openai.com/) | Vector embeddings |
| [PyPDF](https://pypi.org/project/pypdf/) | PDF document loading |
| [psycopg / psycopg2](https://www.psycopg.org/) | PostgreSQL driver |

### Database
| Technology | Purpose |
|---|---|
| [PostgreSQL](https://www.postgresql.org/) | Primary relational database |
| [pgvector](https://github.com/pgvector/pgvector) | Vector similarity search extension |

---

## 📁 Project Structure

```text
langgraph_console_backend/
│
├── langgraph-api.py                     # Main FastAPI unified backend service (port 8002)
├── langgraph_agent.py                   # Core LangGraph workflow definition
├── requirements.txt                     # Python dependencies
├── .env                                 # Environment variables for PostgreSQL, Supabase, OpenAI
├── misellenious/                        # Misc scripts and older variants
└── not needed/                          # Legacy service files (agent_api.py, kb_service.py)
```

---

## 🗄️ Database Schema

### `user`
| Column | Type | Description |
|---|---|---|
| `id` | `UUID` | Primary key |
| `email` | `TEXT` | Unique user email |
| `password` | `TEXT` | bcrypt-hashed password |
| `date_of_creation` | `TIMESTAMP` | Account creation date |

### `created_agents`
| Column | Type | Description |
|---|---|---|
| `id` | `SERIAL` | Primary key |
| `user_id` | `UUID` | Foreign key → `user.id` |
| `agent_name` | `TEXT` | Display name of the agent |
| `system_prompt` | `TEXT` | Custom persona/instructions |
| `k_value` | `INTEGER` | Number of chunks to retrieve |
| `knowledge_base` | `TEXT` | Associated knowledge base name |
| `namespace` | `TEXT` | Namespace/sub-collection within KB |
| `session_id` | `UUID` | Current conversation thread ID |

### `knowledge_base`
| Column | Type | Description |
|---|---|---|
| `id` | `SERIAL` | Primary key |
| `user_id` | `UUID` | Foreign key → `user.id` |
| `content` | `TEXT` | Text chunk content |
| `name_metadata` | `TEXT` | Namespace/document name |
| `knowledge_base_name` | `TEXT` | KB group name |
| `vector_content` | `vector(1536)` | OpenAI embedding (pgvector) |

> **Note:** The `knowledge_base` table requires the `pgvector` PostgreSQL extension using `<=>` (cosine distance) for similarity search.

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+
- **Python** 3.10+
- **PostgreSQL** 14+ with the `pgvector` extension installed

### 1. Clone the Repository

```bash
git clone <repository-url>
cd langgraph_console
```

### 2. Install Frontend Dependencies

```bash
npm install
```

### 3. Install Python Dependencies

It is strongly recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure the Database

Connect to your PostgreSQL instance and run the following SQL to set up the required schema:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Users table
CREATE TABLE IF NOT EXISTS "user" (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    date_of_creation TIMESTAMP DEFAULT NOW()
);

-- Agents table
CREATE TABLE IF NOT EXISTS created_agents (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES "user"(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    system_prompt TEXT,
    k_value INTEGER DEFAULT 4,
    knowledge_base TEXT,
    namespace TEXT,
    session_id UUID
);

-- Knowledge base vector table
CREATE TABLE IF NOT EXISTS knowledge_base (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES "user"(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    name_metadata TEXT,
    knowledge_base_name TEXT,
    vector_content vector(1536)
);
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project backend directory:

```env
OPENAI_API_KEY=sk-...your-openai-key...

# Supabase Credentials
NEXT_PUBLIC_SUPABASE_URL=https://<your-project>.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1...

# Database URL
DB_URL=postgresql://postgres:your_db_password@localhost:5432/postgres
```

---

## 🏃 Running the Application

If you are running the backend in isolation, you only need to start the main FastAPI application. Navigate to the backend directory and run:

### Unified Backend Service
```bash
# Activate your virtual environment first
source .venv/bin/activate

# The FastAPI app is exposed on port 8002
python langgraph-api.py
```
> API Docs: **http://localhost:8002/docs**

*(Note: The Next.js frontend application, if applicable to your deployment, should be started separately in its respective repository using `npm run dev`.)*

---

## 🔌 API Reference

### Next.js API Routes

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/auth/login` | Authenticate a user; verifies bcrypt password |
| `POST` | `/api/auth/signup` | Create a new user account |
| `POST` | `/api/auth/logout` | Clear the user session |
| `GET` | `/api/dashboard/stats?userId=` | Get count of KBs and agents for a user |
| `GET` | `/api/agents?userId=` | List all agents for a user |
| `POST` | `/api/agents` | Create a new agent |
| `PATCH` | `/api/agents` | Update agent fields (e.g., `session_id`) |
| `POST` | `/api/kb/upload` | Proxy file upload to the KB service |
| `GET` | `/api/kb/chunks?userId=&kbName=` | Retrieve stored text chunks |
| `GET` | `/api/kb/names?userId=` | List unique knowledge base names |
| `GET` | `/api/kb/namespaces?kbName=` | List namespaces within a knowledge base |
| `GET` | `/api/kb/unique-names` | Get globally unique KB names |
| `DELETE` | `/api/kb/delete` | Delete a knowledge base |

### Python Backend Service (Port 8002)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/process_kb` | Upload, parse, chunk, embed, and store documents |
| `POST` | `/chat` | Send a message to the LangGraph agent |
| `POST` | `/widget/chat` | Isolated chat interface for the embeddable chat widget |
| `GET` | `/widget/config` | Retrieves configuration details to initialize the widget |
| `POST` | `/generate_system_prompt` | Dynamically produce a system prompt based on description |

**`POST /process_kb` — Form Data:**
```text
files:          List[UploadFile]   # One or more PDF/TXT/MD/CSV files
kb_name:        str                # Knowledge base group name
knowledge_name: str                # Namespace/document identifier
user_id:        str                # User UUID
```

**`POST /chat` — JSON Body:**
```json
{
  "query": "What is the refund policy?",
  "k": 4,
  "namespace": "my-document",
  "user_id": "884a8be4-...",
  "thread_id": "some-uuid",
  "kb_name": "company-docs",
  "system_prompt": "You are a helpful support agent."
}
```

---

## 🧠 How the LangGraph Agent Works

The agent workflow in `langgraph_agent.py` follows a two-node graph:

```
  [START]
     │
     ▼
┌──────────┐     Embeds query → queries pgvector with cosine similarity
│ retriever│─────────────────────────────────────────────────────────►
└──────────┘     Returns top-K matching document chunks as context
     │
     ▼
┌──────────┐     Feeds system_prompt + context + user query to GPT-4o
│ chatbot  │─────────────────────────────────────────────────────────►
└──────────┘     Returns the final answer
     │
     ▼
  [END]
```

- **State** is persisted in-memory using `MemorySaver` keyed by `thread_id`, enabling multi-turn conversations.
- If no `namespace` is configured for an agent, the retriever is skipped and the agent responds using only its system prompt.
- The model used for chat is **GPT-4o** (`temperature=0.1`).
- Embeddings use **`text-embedding-3-small`** (1536 dimensions).

---

## 🧩 Key Components

| Component | File | Description |
|---|---|---|
| `AgentList` | `components/agents/AgentList.tsx` | Filterable list of agents; click to open chat |
| `CreateAgentModal` | `components/agents/CreateAgentModal.tsx` | Form to create a new agent |
| `AgentSettingsModal` | `components/agents/AgentSettingsModal.tsx` | Edit agent config from within the chat page |
| `AddKnowledgeBaseForm` | `components/knowledge-base/AddKnowledgeBaseForm.tsx` | Multi-file upload form with drag-and-drop support |
| `DashboardLayout` | `app/dashboard/layout.tsx` | Collapsible sidebar navigation |
| `AgentChatPage` | `app/dashboard/agent/[id]/page.tsx` | Full chat interface with session reset |

---

## 📄 License

This project is licensed under the MIT License.
# langgraph_console
