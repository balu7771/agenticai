# pip install langchain-google-genai requests
import gradio as gr
import os
import requests
from typing import TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# State
class State(TypedDict):
    query: str
    vector: str
    serp: str
    llm: str

# Setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local("c:/code/agentic_ai/4_langgraph/product_embeddings_faiss", embeddings, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Nodes
def vector_search(state: State) -> State:
    results = vectordb.similarity_search(state["query"], k=2)
    state["vector"] = "\n".join([f"• {doc.metadata['title']}" for doc in results])
    return state

def serp_search(state: State) -> State:
    url = "https://serpapi.com/search"
    params = {"q": f"{state['query']} reviews", "api_key": os.getenv("SERPAPI_KEY"), "num": 2}
    data = requests.get(url, params=params).json()
    results = [f"• {r['title']}: {r['snippet']}" for r in data.get("organic_results", [])[:2]]
    state["serp"] = "\n".join(results)
    return state

def llm_analyze(state: State) -> State:
    prompt = f"Analyze: Vector DB: {state['vector']} | Web: {state['serp']} | Query: {state['query']}"
    state["llm"] = llm.invoke(prompt).content
    return state

# Graph
graph = StateGraph(State)
graph.add_node("vector_node", vector_search)
graph.add_node("serp_node", serp_search)
graph.add_node("llm_node", llm_analyze)

graph.set_entry_point("vector_node")
graph.add_edge("vector_node", "serp_node")
graph.add_edge("serp_node", "llm_node")
graph.add_edge("llm_node", END)

runnable = graph.compile()

# Interface
def search(query, chat_history):
    result = runnable.invoke({"query": query})
    answer = f"**Vector:** {result['vector']}\n\n**Web:** {result['serp']}\n\n**AI:** {result['llm']}"
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history

demo = gr.ChatInterface(
    fn=search,
    title="Product Search",
    type="messages"
)


if __name__ == "__main__":
    demo.launch()