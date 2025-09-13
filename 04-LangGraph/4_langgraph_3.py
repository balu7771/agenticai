import gradio as gr
import pandas as pd
from typing import TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

# State
class ProductState(TypedDict):
    query: str
    results: str

# Load embeddings once
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local("c:/code/agentic_ai/4_langgraph/product_embeddings_faiss", embeddings, allow_dangerous_deserialization=True)
df = pd.read_pickle("c:/code/agentic_ai/4_langgraph/product_data.pkl")

# Node 1: Search
def search_products(state: ProductState) -> ProductState:
    results = vectordb.similarity_search(state["query"], k=3)
    titles = [doc.metadata["title"] for doc in results]
    state["results"] = "\n".join([f"• {title}" for title in titles])
    return state

# Node 2: Format
def format_response(state: ProductState) -> ProductState:
    if state["results"]:
        state["results"] = f"Found products:\n{state['results']}"
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
    title="Product Search",
    examples=["wireless headphones", "laptop", "camera"]
)

if __name__ == "__main__":
    demo.launch()