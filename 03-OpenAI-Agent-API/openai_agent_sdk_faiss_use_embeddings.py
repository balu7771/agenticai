import os
from dotenv import load_dotenv
import json
import numpy as np
import gradio as gr
import asyncio
import faiss  # FAISS for similarity search
import pickle

from openai import OpenAI
from agents import Agent, Runner, FunctionTool

# --- Load environment ---
load_dotenv(override=True)
client = OpenAI()

# --- Load precomputed embeddings and texts ---
EMBEDDINGS_FILE = "c:/code/agentic_ai/1_foundations/chunk_embeddings.pkl"
INDEX_FILE = "c:/code/agentic_ai/1_foundations/faiss_index.bin"

with open(EMBEDDINGS_FILE, "rb") as f:
    data = pickle.load(f)
texts = data["texts"]
chunk_embeddings = data["embeddings"]
print(f"Loaded {len(texts)} chunks from {EMBEDDINGS_FILE}")

# --- Load FAISS index ---
index = faiss.read_index(INDEX_FILE)
print(f"Loaded FAISS index from {INDEX_FILE}")

# --- Tool callback function (RAG with generation) ---
async def rag_invoker(tool_context, params):
    args = json.loads(params) if isinstance(params, str) else params
    user_query = args.get("topic", "")

    # Embed the query
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=user_query
    )
    query_embedding = response.data[0].embedding
    query_vector = np.array([query_embedding]).astype("float32")

    # Retrieve top 1 nearest neighbor
    distances, indices = index.search(query_vector, k=1)
    best_idx = indices[0][0]
    best_text = texts[best_idx]

    if best_text:
        # --- Generation step using LLM ---
        prompt = f"Answer the user's question based on the following document content. Be concise and clear.\n\nDocument: {best_text}\n\nQuestion: {user_query}\nAnswer:"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    else:
        return "I'm sorry, I couldn't find information related to that query."

# --- Tool schema ---
rag_schema = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "description": "The user's question or topic."}
    },
    "required": ["topic"]
}

# --- Register FunctionTool ---
rag_tool = FunctionTool(
    name="get_textfile_answer",
    description="Answers questions by retrieving relevant information from the text files and generating a concise answer.",
    params_json_schema=rag_schema,
    on_invoke_tool=rag_invoker
)

# --- Define Agent ---
rag_agent = Agent(
    name="Document RAG Bot",
    instructions="You are a helpful assistant that answers questions using the indexed text documents.",
    tools=[rag_tool]
)

# --- Chat function for Gradio ---
async def chat_with_rag(message, chat_history):
    session = await Runner.run(rag_agent, message)
    chat_history = chat_history or []
    chat_history.append((message, session.final_output))
    return chat_history, chat_history

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Charles Darwin Bot (RAG + FAISS Index)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about Charles Darwin ...")
    clear = gr.Button("Clear")

    async def respond(user_message, chat_history):
        return await chat_with_rag(user_message, chat_history)

    msg.submit(respond, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()
