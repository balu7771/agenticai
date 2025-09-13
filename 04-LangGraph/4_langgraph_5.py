import gradio as gr
import os
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
    type: str
    result: str

# Setup
vectordb = FAISS.load_local("c:/code/agentic_ai/4_langgraph/product_embeddings_faiss", 
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), 
    allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Nodes
def classify(state: State) -> State:
    if "vs" in state["query"] or "compare" in state["query"]:
        state["type"] = "compare"
    elif "best" in state["query"] or "recommend" in state["query"]:
        state["type"] = "recommend"
    else:
        state["type"] = "specific"
    return state

def handle_specific(state: State) -> State:
    products = vectordb.similarity_search(state["query"], k=2)
    titles = [doc.metadata["title"] for doc in products]
    state["result"] = llm.invoke(f"Analyze this product: {titles[0]}").content
    return state

def handle_compare(state: State) -> State:
    products = vectordb.similarity_search(state["query"], k=4)
    titles = [doc.metadata["title"] for doc in products]
    state["result"] = llm.invoke(f"Compare these products: {titles}").content
    return state

def handle_recommend(state: State) -> State:
    products = vectordb.similarity_search(state["query"], k=3)
    titles = [doc.metadata["title"] for doc in products]
    state["result"] = llm.invoke(f"Recommend from these: {titles}").content
    return state

# Conditional routing
def route(state: State) -> str:
    return state["type"]

# Graph
graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("specific", handle_specific)
graph.add_node("compare", handle_compare)
graph.add_node("recommend", handle_recommend)

graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route, {
    "specific": "specific",
    "compare": "compare", 
    "recommend": "recommend"
})
graph.add_edge("specific", END)
graph.add_edge("compare", END)
graph.add_edge("recommend", END)

runnable = graph.compile()

# Interface
# Interface
def search(query, chat_history):
    result = runnable.invoke({"query": query})
    answer = f"**Type**: {result['type']}\n\n{result['result']}"
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history

demo = gr.ChatInterface(
    fn=search,
    title="Conditional LangGraph",
    examples=["iPhone 15", "iPhone vs Samsung", "best laptop"],
    type="messages"
)


if __name__ == "__main__":
    demo.launch()