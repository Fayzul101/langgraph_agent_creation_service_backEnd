system_prompt = """Youre a helpful support guy."""
glob_k = 0
glob_namespace = ""
glob_user_id = "884a8be4-3876-4a67-b9ad-bf898ebceda4"
glob_kb = ""
# Default values kept for manual testing only


from dotenv import load_dotenv
import os
from langgraph.graph.message import BaseMessage, add_messages
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from typing_extensions import TypedDict, Dict, Annotated
from langchain_core.prompts import ChatPromptTemplate
import psycopg2
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

load_dotenv(dotenv_path=".env")
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),   
)

embedding_tool = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    sys_prompt: str
    context: str
    response: str
    glob_namespace: str
    glob_k: int
    glob_user_id: str
    glob_kb_name: str
# def init_state(state: State) -> State:
#     if 'sys_prompt' not in state:
#         if system_prompt == "":
#             return {'sys_prompt': "",}
#         else:
#             return {'sys_prompt': system_prompt}
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5434"),
        dbname=os.getenv("DB_NAME", "langgraph_users"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )
def retrieve_from_postgres(state: State) -> State:
    """
    1. Embeds the query
    2. Searches vector_content column using cosine similarity
    3. Filters by name_metadata = namespace and user_id
    """
    if state["glob_namespace"] != "":
        k = state["glob_k"]
        namespace = state["glob_namespace"]
        user_id = state["glob_user_id"]
        knowlege_base_name = state["glob_kb_name"]
        query = state["messages"][-1].content
        query_embedding = embedding_tool.embed_query(query)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        sql = """
            SELECT
                content,                                          
                name_metadata,
                user_id,
                1 - (vector_content <=> %s::vector) AS similarity
            FROM knowledge_base                                  
            WHERE name_metadata = %s
            AND knowledge_base_name = %s
            AND user_id = %s
            ORDER BY vector_content <=> %s::vector
            LIMIT %s;
        """

        docs = []
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (embedding_str, namespace, knowlege_base_name, user_id, embedding_str, k))
                rows = cur.fetchall()
                for row in rows:
                    text, ns, uid, similarity = row
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"name_metadata": ns, "user_id": str(uid), "similarity": round(similarity, 4)},
                        )
                    )
        finally:
            conn.close()

        return {"context": docs}
    else:
        return {"context": "No context here"}

def chat_agent(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
    """
    System_prompt:
    {system_prompt}

    Context:
    {context}

    User_query:
    {query}
    """
    )
    agent_chain = prompt | llm
    response = agent_chain.invoke({
        "system_prompt": state['sys_prompt'],
        "context": state['context'],
        "query": state['messages']
    })
    return {"messages": [response]}

workflow = StateGraph(State)

# workflow.add_node("initialization", init_state)
workflow.add_node("chatbot", chat_agent)
workflow.add_node("retriever", retrieve_from_postgres)

workflow.set_entry_point("retriever")
# workflow.add_edge("initialization", "retriever")
workflow.add_edge("retriever", "chatbot")
workflow.add_edge("chatbot", END)

app = workflow.compile(checkpointer=memory)

def run_agent(query: str, meth_k: int, meth_namespace: str, meth_uuid: str, thread_id: str, meth_kb: str, meth_sys_prompt: str) -> Dict[str, str]:
    results = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "glob_k": meth_k,
            "glob_user_id": meth_uuid,
            "glob_namespace": meth_namespace,
            "glob_kb_name": meth_kb,
            "sys_prompt": meth_sys_prompt,
        },
        {"configurable": {"thread_id": thread_id}},
    )
    print(f"DEBUG: run_agent results messages count: {len(results.get('messages', []))}")
    return {"messages": results["messages"]}
query = "Nice"
result = run_agent(query, glob_k, glob_namespace, glob_user_id, "thread_1", glob_kb, system_prompt)
print(f"Response: {result['messages'][-1].content}")
print("\n")