import gradio as gr
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

# Simulated memory for HITL (can be replaced with database storage)
memory = {}

# Node 1: Search
def search_products(state: ProductState) -> ProductState:
    if state["query"] in memory:
        state["results"] = memory[state["query"]]
    else:
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

# HITL approve function
def approve(query, results):
    memory[query] = results
    return "Results saved!", memory

# HITL edit function
def edit(query, edited_results):
    memory[query] = edited_results
    return "Edited results saved!", memory

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Product Search with HITL Support")

    chatbot = gr.Chatbot()
    query_input = gr.Textbox(label="Enter your query")
    search_btn = gr.Button("Search")
    
    result_box = gr.Textbox(label="Search Results")
    approve_btn = gr.Button("Approve")
    edit_btn = gr.Button("Edit")
    edit_box = gr.Textbox(label="Edit Results")
    
    status_box = gr.Textbox(label="Status")  # Added for feedback messages
    memory_display = gr.Textbox(label="Memory", lines=10)

    # Perform search
    def run_search(query):
        results = search(query)
        return results, results

    search_btn.click(run_search, inputs=query_input, outputs=[result_box, edit_box])

    # Approve results
    approve_btn.click(approve, inputs=[query_input, result_box], outputs=[status_box, memory_display])

    # Edit and save results
    edit_btn.click(edit, inputs=[query_input, edit_box], outputs=[status_box, memory_display])

if __name__ == "__main__":
    demo.launch()

