from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import sqlite3
import numpy as np
import gradio as gr
import asyncio

from agents import Agent, Runner, FunctionTool

# --- Load environment ---
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# --- Database connection ---
DB_PATH = "faqs.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# --- Load FAQs from SQLite ---
def load_faqs():
    cursor.execute("SELECT topic, answer FROM faqs")
    return dict(cursor.fetchall())

knowledge_base = load_faqs()

# --- Precompute embeddings for each topic ---
embeddings_index = {}

def compute_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

print("Computing embeddings for knowledge base...")
for topic_key, answer in knowledge_base.items():
    embedding = compute_embedding(answer)
    embeddings_index[topic_key] = embedding
print("Embeddings ready!")

# --- Function to compute cosine similarity ---
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# --- Tool callback function ---
async def faq_invoker(tool_context, params):
    args = json.loads(params) if isinstance(params, str) else params
    user_query = args.get("topic", "")

    query_embedding = compute_embedding(user_query)

    best_topic = None
    best_score = -1

    for topic_key, embedding in embeddings_index.items():
        score = cosine_similarity(query_embedding, embedding)
        if score > best_score:
            best_score = score
            best_topic = topic_key

    if best_topic:
        return knowledge_base[best_topic]
    else:
        return "I'm sorry, but I couldn't find specific information about that topic."

# --- Tool schema ---
faq_schema = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": "The topic or question asked by the customer."
        }
    },
    "required": ["topic"]
}

# --- Register FunctionTool ---
faq_tool = FunctionTool(
    name="get_faq_answer",
    description="Provides answers to frequently asked customer support questions using semantic similarity from a database.",
    params_json_schema=faq_schema,
    on_invoke_tool=faq_invoker
)

# --- Define Agent ---
faq_agent = Agent(
    name="Customer Support Bot",
    instructions="You are a helpful customer support assistant. "
                 "Answer questions based on the knowledge base from the database or use the provided FAQ tool with semantic understanding.",
    tools=[faq_tool]
)

# --- Chat function for Gradio ---
async def chat_with_support(message, chat_history):
    response = await Runner.run(faq_agent, message)
    chat_history = chat_history or []
    chat_history.append((message, response.final_output))
    return chat_history, chat_history

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Customer Support Bot")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about our services or policies...")
    clear = gr.Button("Clear")

    async def respond(user_message, chat_history):
        return await chat_with_support(user_message, chat_history)

    msg.submit(respond, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()
