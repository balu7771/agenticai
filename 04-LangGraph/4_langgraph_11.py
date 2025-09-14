import gradio as gr
import pandas as pd
import sqlite3
import time
from datetime import datetime
from typing import TypedDict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

# State
class ProductState(TypedDict):
    query: str
    results: str
    memory_hit: bool
    explanation: str

# Setup SQLite for memory and logs
conn = sqlite3.connect('memory.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS memory (
    query TEXT PRIMARY KEY,
    results TEXT,
    timestamp TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    timestamp TEXT,
    query TEXT,
    memory_hit INTEGER,
    latency_ms INTEGER,
    error TEXT
)
""")
conn.commit()

def log_event(query: str, memory_hit: bool, latency_ms: int, error: str = None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO logs (timestamp, query, memory_hit, latency_ms, error)
    VALUES (?, ?, ?, ?, ?)
    """, (timestamp, query, int(memory_hit), latency_ms, error))
    conn.commit()

def get_memory(query: str) -> Tuple[str, bool, str]:
    cursor.execute("SELECT results, timestamp FROM memory WHERE query = ?", (query,))
    row = cursor.fetchone()
    if row:
        results, timestamp = row
        explanation = f"Loaded from memory; last updated at {timestamp}."
        return f"(Loaded answer from memory, last updated: {timestamp})\n{results}", True, explanation
    return None, False, ""

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

# Node 1: Search with memory check, explainability, and observability
def search_products(state: ProductState) -> ProductState:
    start_time = time.time()
    error_message = None
    try:
        cached_result, memory_hit, explanation = get_memory(state["query"])
        if memory_hit:
            state["results"] = cached_result
            state["memory_hit"] = True
            state["explanation"] = explanation
        else:
            results = vectordb.similarity_search_with_score(state["query"], k=3)
            if not results:
                state["results"] = "No products found"
                state["memory_hit"] = False
                state["explanation"] = "No matches found."
            else:
                titles = []
                explain_details = []
                for doc, score in results:
                    title = doc.metadata["title"]
                    titles.append(f"• {title} (score: {score:.4f})")
                    explain_details.append(f"Matched because the title includes '{title}' with a similarity score of {score:.4f}")
                result_text = "\n".join(titles)
                explanation = "\n".join(explain_details)
                state["results"] = result_text
                state["memory_hit"] = False
                state["explanation"] = explanation
                save_memory(state["query"], result_text)
    except Exception as e:
        error_message = str(e)
        state["results"] = "An error occurred during the search."
        state["memory_hit"] = False
        state["explanation"] = f"Error details: {error_message}"

    latency_ms = int((time.time() - start_time) * 1000)
    log_event(state["query"], state["memory_hit"], latency_ms, error_message)
    return state

# Node 2: Format response to include explanation
def format_response(state: ProductState) -> ProductState:
    if state["results"]:
        prefix = "Found products:" if not state.get("memory_hit") else ""
        explanation = state.get("explanation", "")
        state["results"] = f"{prefix}\n{state['results']}\n\nExplanation:\n{explanation}" if prefix else f"{state['results']}\n\nExplanation:\n{explanation}"
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

# Admin view for logs
def view_logs():
    cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 20")
    rows = cursor.fetchall()
    return "\n".join([f"[{row[0]}] Query: {row[1]}, Memory hit: {bool(row[2])}, Latency: {row[3]} ms, Error: {row[4]}" for row in rows])

# Gradio interface
def chat_fn(message, history):
    response = search(message)
    history = history or []
    history.append([message, response])
    return history


demo = gr.Blocks()

with demo:
    gr.Markdown("# Product Search with Memory, Explainability, Observability & Monitoring")

    with gr.Tab("Search"):
        chatbot = gr.Chatbot(label="Agent Output")
        query = gr.Textbox(label="Ask something", placeholder="e.g. wireless headphones")
        query.submit(chat_fn, [query, chatbot], [chatbot])
    
    with gr.Tab("Admin Logs"):
        gr.Markdown("### Recent Search Logs")
        logs_output = gr.Textbox(label="Logs", lines=20, interactive=False)
        refresh_button = gr.Button("Refresh Logs")
        refresh_button.click(fn=view_logs, inputs=[], outputs=[logs_output])

if __name__ == "__main__":
    demo.launch()
