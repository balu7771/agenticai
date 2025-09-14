import gradio as gr
import pandas as pd
import sqlite3
from datetime import datetime
from typing import TypedDict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

# State
class ProductState(TypedDict):
    query: str
    results: str
    memory_hit: bool  # Added to indicate memory usage

# Setup SQLite for memory with timestamp
conn = sqlite3.connect('memory.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS memory (
    query TEXT PRIMARY KEY,
    results TEXT,
    timestamp TEXT
)
""")
conn.commit()

def get_memory(query: str) -> Tuple[str, bool]:
    cursor.execute("SELECT results, timestamp FROM memory WHERE query = ?", (query,))
    row = cursor.fetchone()
    if row:
        results, timestamp = row
        return f"(Loaded answer from memory, last updated: {timestamp})\n{results}", True
    return None, False

def save_memory(query: str, results: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT OR REPLACE INTO memory (query, results, timestamp)
    VALUES (?, ?, ?)
    """, (query, results, timestamp))
    conn.commit()

# Load embeddings once
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local("c:/code/agentic_ai/4_langgraph/product_embeddings_faiss", embeddings, allow_dangerous_deserialization=True)
df = pd.read_pickle("c:/code/agentic_ai/4_langgraph/product_data.pkl")

# Node 1: Search with memory check
def search_products(state: ProductState) -> ProductState:
    cached_result, memory_hit = get_memory(state["query"])
    if memory_hit:
        state["results"] = cached_result
        state["memory_hit"] = True
        return state

    # If not in memory, search vector DB
    results = vectordb.similarity_search(state["query"], k=3)
    titles = [doc.metadata["title"] for doc in results]
    result_text = "\n".join([f"• {title}" for title in titles])
    state["results"] = result_text
    state["memory_hit"] = False

    # Save to memory with timestamp
    save_memory(state["query"], result_text)

    return state

# Node 2: Format
def format_response(state: ProductState) -> ProductState:
    if state["results"]:
        prefix = "Found products:" if not state.get("memory_hit") else ""
        state["results"] = f"{prefix}\n{state['results']}" if prefix else state["results"]
    else:
        state["results"] = "No products found"
    return state

# Build graph
graph = StateGraph(ProductState)
graph.add_node("search", search_products)
graph.add_node("format", format_response)
graph.set_entry_point("search")
graph.add_edge("search", "format")
graph.add_edge("format", END)
runnable = graph.compile()

# Search function
def search(query):
    result = runnable.invoke({"query": query})
    return result["results"]

# Gradio interface
def chat_fn(message, history):
    return search(message)

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Product Search with Memory and Timestamp",
    examples=["wireless headphones", "laptop", "camera"]
)

if __name__ == "__main__":
    demo.launch()
