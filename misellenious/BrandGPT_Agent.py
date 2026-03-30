system_prompt = """
You are safai's assistnat. Your task is to help users with their queries regarding safai's cleaning services and help them calculate the cost of the services they want to avail. You have access to "retrieve_from_postgres" tool which allows you to retrieve relevant information from the knowledge base. 
After fetching data:
1. Use it on your own way to respond to user queries.
2. Ask follow up related follow up question.
3. Help to caculate the cost of the services based on what the user asked for.

If user say they "need a service" then give them the price at which we are giving that service. After that ask follow up question to get more details about the service they want to avail so that you can calculate the cost for them.
"""
glob_k = 3
glob_namespace = "full_data"
glob_user_id = "884a8be4-3876-4a67-b9ad-bf898ebceda4"
glob_kb = "Safai"
# Default values kept for manual testing only


from dotenv import load_dotenv
import os
from langgraph.graph.message import BaseMessage, add_messages
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from typing_extensions import TypedDict, Dict, Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import psycopg2
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig

load_dotenv(dotenv_path=".env")

memory = MemorySaver()

embedding_tool = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
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

@tool
def retrieve_from_postgres(
    query: str,
    config: RunnableConfig  # ✅ LangGraph injects this automatically
) -> str:
    """
    Use this tool by sending only the query. The tool will:
    1. Embeds the query
    2. Searches vector_content column using cosine similarity
    3. Filters by name_metadata = namespace and user_id
    """
    configurable = config.get("configurable", {})
    k = configurable.get("glob_k", 5)
    namespace = configurable.get("glob_namespace", "")
    user_id = configurable.get("glob_user_id", "")
    knowlege_base_name = configurable.get("glob_kb_name", "")

    if not namespace:
        return "Rag Data: No rag data"

    print(f"Tool Called\nk_Value: {k}\nNamespace: {namespace}\nUser ID: {user_id}\nKnowledge Base Name: {knowlege_base_name}\nQuery: {query}")

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

    return f"Rag Data: {docs}"


llm_tools = [retrieve_from_postgres]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),   
).bind_tools(tools=llm_tools)

def chat_agent(state: State, config: RunnableConfig) -> State:
    configurable = config.get("configurable", {})
    sys_prompt = configurable.get("glob_sys_prompt", "You are a helpful assistant.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    agent_chain = prompt | llm
    response = agent_chain.invoke(state)
    return {"messages": [response]}

workflow = StateGraph(State)

tool_node = ToolNode(llm_tools)

# workflow.add_node("initialization", init_state)
workflow.add_node("chatbot", chat_agent)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("chatbot")
# workflow.add_edge("initialization", "retriever")
# workflow.add_edge("retriever", "chatbot")
workflow.add_conditional_edges(
    "chatbot",
    tools_condition,
)
workflow.add_edge("tools", "chatbot")

app = workflow.compile(checkpointer=memory)

def run_agent(query: str, meth_k: int, meth_namespace: str, meth_uuid: str, thread_id: str, meth_kb: str, meth_sys_prompt: str) -> Dict[str, str]:
    results = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
        },
        config={
            "configurable": {
                "thread_id": thread_id,
                "glob_k": meth_k,
                "glob_user_id": meth_uuid,
                "glob_namespace": meth_namespace,
                "glob_kb_name": meth_kb,
            }
        }
    )
    return {"messages": results["messages"]}

