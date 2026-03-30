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

The system is split into three independently running processes:

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
│  └──────────────┘  └──────┬───────┘  └──────┬───────┘  │
└─────────────────────────  │  ─────────────  │  ────────┘
                            │                 │
              ┌─────────────▼──┐    ┌─────────▼──────────┐
              │  kb_service.py │    │   agent_api.py      │
              │  (Port 8000)   │    │   (Port 8001)       │
              │  FastAPI       │    │   FastAPI           │
              │  • PDF Parse   │    │   • LangGraph       │
              │  • Chunk Text  │    │   • RAG Retrieval   │
              │  • Embed + Save│    │   • Chat Response   │
              └───────┬────────┘    └─────────┬──────────┘
                      │                       │
                      └──────────┬────────────┘
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

```
langgraph_console/
│
├── src/
│   ├── app/
│   │   ├── page.tsx                     # Login page
│   │   ├── signup/                      # User signup page
│   │   ├── dashboard/
│   │   │   ├── layout.tsx               # Dashboard shell (sidebar + header)
│   │   │   ├── page.tsx                 # Dashboard overview (stats)
│   │   │   ├── agent/[id]/page.tsx      # Agent chat interface
│   │   │   ├── create-agent/            # Agent list & creation page
│   │   │   ├── knowledge-base/          # Knowledge base upload page
│   │   │   └── database/               # Database chunk inspector
│   │   └── api/
│   │       ├── auth/
│   │       │   ├── login/route.ts       # POST /api/auth/login
│   │       │   ├── logout/route.ts      # POST /api/auth/logout
│   │       │   └── signup/route.ts      # POST /api/auth/signup
│   │       ├── dashboard/
│   │       │   └── stats/route.ts       # GET /api/dashboard/stats
│   │       ├── agents/route.ts          # GET/POST/PATCH /api/agents
│   │       └── kb/
│   │           ├── upload/route.ts      # POST /api/kb/upload
│   │           ├── chunks/route.ts      # GET /api/kb/chunks
│   │           ├── names/route.ts       # GET /api/kb/names
│   │           ├── namespaces/route.ts  # GET /api/kb/namespaces
│   │           ├── unique-names/route.ts# GET /api/kb/unique-names
│   │           └── delete/route.ts      # DELETE /api/kb/delete
│   ├── components/
│   │   ├── agents/
│   │   │   ├── AgentList.tsx            # List of agents with filter
│   │   │   ├── CreateAgentModal.tsx     # Modal form to create an agent
│   │   │   ├── AgentSettingsModal.tsx   # Modal to edit agent settings
│   │   │   └── AgentSettingsSidebar.tsx # Sidebar variant of settings
│   │   ├── knowledge-base/
│   │   │   └── AddKnowledgeBaseForm.tsx # PDF upload form
│   │   └── ui/
│   │       ├── Button.tsx               # Reusable button component
│   │       ├── Input.tsx                # Reusable input component
│   │       └── Label.tsx                # Reusable label component
│   └── lib/
│       ├── db.ts                        # PostgreSQL query helper
│       └── utils.ts                     # Utility functions (cn, etc.)
│
├── agent_api.py                         # FastAPI: Agent chat service (port 8001)
├── kb_service.py                        # FastAPI: Document processing service (port 8000)
├── langgraph_agent.py                   # Core LangGraph workflow definition
├── requirements.txt                     # Python dependencies
├── package.json                         # Node.js dependencies
├── .env                                 # Environment variables for Python services
└── .env.local                           # Environment variables for Next.js
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

### Root `.env` (used by Python services)

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...your-openai-key...

DB_HOST=localhost
DB_PORT=5432
DB_NAME=langgraph_users
DB_USER=postgres
DB_PASSWORD=your_db_password
```

### `.env.local` (used by Next.js)

Create a `.env.local` file in the project root:

```env
DATABASE_URL=postgresql://postgres:your_db_password@localhost:5432/langgraph_users
```

---

## 🏃 Running the Application

You need to start **three** separate processes. Open three terminal tabs/windows:

### Terminal 1 — Next.js Frontend
```bash
npm run dev
```
> Accessible at: **http://localhost:3000**

### Terminal 2 — Knowledge Base Service
```bash
# Activate your virtual environment first
source .venv/bin/activate

uvicorn kb_service:app --host 0.0.0.0 --port 8000 --reload
```
> API Docs: **http://localhost:8000/docs**

### Terminal 3 — Agent Chat API
```bash
# Activate your virtual environment first
source .venv/bin/activate

uvicorn agent_api:app --host 0.0.0.0 --port 8001 --reload
```
> API Docs: **http://localhost:8001/docs**

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

### Python Microservices

#### KB Service (Port 8000)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/process-kb` | Upload, parse, chunk, embed, and store documents |

**`POST /process-kb` — Form Data:**
```
files:          List[UploadFile]   # One or more PDF/TXT/MD/CSV files
kb_name:        str                # Knowledge base group name
knowledge_name: str                # Namespace/document identifier
user_id:        str                # User UUID
```

#### Agent API (Port 8001)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/chat` | Send a message to the LangGraph agent |

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
