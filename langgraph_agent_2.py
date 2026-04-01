system_prompt = """Youre a helpful support guy."""
glob_k = 0
glob_namespace = ""
glob_user_id = "884a8be4-3876-4a67-b9ad-bf898ebceda4"
glob_kb = ""
# Default values kept for manual testing only


from dotenv import load_dotenv
import os
import requests
from langgraph.graph.message import BaseMessage, add_messages
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from typing_extensions import TypedDict, Dict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

load_dotenv(dotenv_path=".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

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
def retrieve_from_postgres(state: State) -> State:
    """
    1. Embeds the query
    2. Searches vector_content column using match_documents RPC in Supabase
    3. Filters by namespace and user_id
    """
    if state["glob_namespace"] != "":
        chain_prompt = ChatPromptTemplate.from_template("""
        You are a query maker bot. Your task is to rewrite the user's query using user's 'present_query' and 'query_history' into a perfect context enriched concise query using which the rag system can retrieve the most relevant data from the knowledge base. Below is the 'present_query' and 'query_history' of the user. If the 'present_query' has context to it, then no need to enrich the question, keep the question as it is.
                                                            
        present_query:
        {query}
                                                            
        query_history:
        {query_history}
                                                        
        Just give me the output query without any intro or outro. Also the query has to be translated in English if it is in Bangla or any other language.""")
        question_enricher = chain_prompt | llm
        
        # Ensure we have a string for the query
        last_message = state['messages'][-1]
        msg_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        query = question_enricher.invoke({"query": msg_content, "query_history": state['messages']}).content
        k = int(state["glob_k"])
        namespace = state["glob_namespace"]
        user_id = state["glob_user_id"]
        knowlege_base_name = state["glob_kb_name"]
        query_embedding = embedding_tool.embed_query(query)

        # Call Supabase match_documents RPC
        url = f"{SUPABASE_URL}/rest/v1/rpc/match_documents"
        headers = {
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query_embedding": query_embedding,
            "match_threshold": 0.5,
            "match_count": k,
            "filter_namespace": namespace,
            "filter_kb_name": knowlege_base_name,
            "filter_user_id": user_id
        }
        
        docs = []
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                rows = response.json()
                for row in rows:
                    docs.append(
                        Document(
                            page_content=row["content"],
                            metadata={
                                "name_metadata": row["name_metadata"], 
                                "user_id": str(row["user_id"]), 
                                "similarity": round(row["similarity"], 4)
                            },
                        )
                    )
            else:
                print(f"Supabase Vector Search Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error in vector search: {str(e)}")

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