# pip install langgraph langchain-community langchain-huggingface langchain-core langchain-text-splitters faiss-cpu sentence-transformers
import sqlite3
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

# Shared State
class ProductState(TypedDict):
    user_query: str
    matched_titles: list
    product_details: str

# Step 1: Initialize SQLite DB and populate with products
def initialize_sqlite():
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            price REAL
        )
    """)
    
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        # Only insert if table is empty
        # Sample data
        products = [
            ("Canon EOS 1500D", "DSLR camera with 24.1 MP sensor", 450.00),
            ("Sony Alpha a6400", "Mirrorless camera with fast autofocus", 900.00),
            ("Nikon D3500", "Beginner DSLR camera, 24.2 MP", 500.00),
            ("GoPro Hero 11", "Action camera with 5.3K video", 400.00)
        ]  
        cursor.executemany("INSERT INTO products (title, description, price) VALUES (?, ?, ?)", products)
        conn.commit()
    conn.close()

# Step 2: Build vector DB from product descriptions
def build_vector_db():
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    cursor.execute("SELECT title, description FROM products")
    rows = cursor.fetchall()
    conn.close()

    documents = [Document(page_content=desc, metadata={"title": title}) for title, desc in rows]
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb

initialize_sqlite()
vectordb = build_vector_db()

# Node 1: Semantic similarity search
def query_vector_db(state: ProductState) -> ProductState:
    print(f"Searching for products similar to: '{state['user_query']}'")
    results = vectordb.similarity_search(state["user_query"], k=2)
    titles = [doc.metadata["title"] for doc in results]
    state["matched_titles"] = titles
    return state

# Node 2: Retrieve product details from SQLite
def get_product_details(state: ProductState) -> ProductState:
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    placeholders = ','.join('?' for _ in state["matched_titles"])
    query = f"SELECT title, description, price FROM products WHERE title IN ({placeholders})"
    cursor.execute(query, state["matched_titles"])
    rows = cursor.fetchall()
    conn.close()

    details = "\n".join([f"{title}: {desc} - ${price}" for title, desc, price in rows])
    state["product_details"] = details
    return state

# Node 3: Format response
def respond_to_user(state: ProductState) -> ProductState:
    print("Recommended products:")
    print(state["product_details"])
    return state

# LangGraph setup
graph = StateGraph(ProductState)
graph.add_node("query_vector_db", query_vector_db)
graph.add_node("get_product_details", get_product_details)
graph.add_node("respond_to_user", respond_to_user)

graph.set_entry_point("query_vector_db")
graph.add_edge("query_vector_db", "get_product_details")
graph.add_edge("get_product_details", "respond_to_user")
graph.add_edge("respond_to_user", END)

runnable = graph.compile()

# Run
if __name__ == "__main__":
    user_input = {
        "user_query": "I want a good camera"
    }
    result = runnable.invoke(user_input)
    print("\nFinal Output:")
    print(result["product_details"])
