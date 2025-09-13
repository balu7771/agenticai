import os
import re
import requests
import gradio as gr
from typing import List
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic

load_dotenv()

# ---- Setup ----
vectordb = FAISS.load_local(
    "c:/code/agentic_ai/4_langgraph/product_embeddings_faiss",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True,
)

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    streaming=True,
)

# ---- Tools ----
def tool_search(query: str) -> str:
    results = vectordb.similarity_search(query, k=2)
    return ", ".join([doc.metadata["title"] for doc in results]) or "No results."

def tool_serp(query: str) -> str:
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": os.getenv("SERPAPI_KEY"), "num": 2}
    try:
        data = requests.get(url, params=params).json()
        results = [f"• {r['title']}: {r['snippet']}" for r in data.get("organic_results", [])[:2]]
        return "\n".join(results) if results else "No SERP results."
    except Exception as e:
        return f"SerpAPI error: {e}"

# --- New Tool: ntfy ---
def tool_ntfy(message: str) -> str:
    """Sends a notification to the specified ntfy topic."""
    ntfy_topic = os.getenv("NTFY_TOPIC")
    ntfy_url = f"https://ntfy.sh/{ntfy_topic}"
    try:
        response = requests.post(ntfy_url, data=message, timeout=5)
        response.raise_for_status()
        return "Notification sent successfully."
    except requests.exceptions.RequestException as e:
        return f"Failed to send notification: {e}"

# ---- Agent Loop ----
def react_agent(query: str):
    state = {"query": query, "history": [f"User: {query}"], "final": ""}

    while True:
        prompt = f"""
You are a ReAct-style agent. 
You MUST always follow this exact output format:

Thought: (one short sentence of reasoning)
Action: (exactly one of the following)
- Search[some query]
- SerpSearch[some query]
- Ntfy[some message]
- Finalize[some final answer]

You should use the Ntfy tool to send a notification whenever the user asks about the 'latest iPhone' or related queries like 'new iPhone', 'iPhone 16', etc. The message for the notification should be concise and directly related to the user's query, for example, "User inquired about latest iPhone." After sending the notification, continue to answer the user's question.

Do NOT output anything else. 
Do NOT answer directly unless using Finalize[].

Example:
Thought: I should look up reviews for the product.
Action: SerpSearch[best headphones reviews]

Conversation so far:
{chr(10).join(state['history'])}

User question: {state['query']}
Now continue.
"""

        response = ""
        for chunk in llm.stream(prompt):
            if chunk.content:
                response += chunk.content
        response = response.strip()
        state["history"].append(response)

        # Parse Action
        action_match = re.search(r"Action\s*:\s*(\w+)\s*\[(.*)\]", response)
        if not action_match:
            break  # Claude didn't follow format → stop

        action, arg = action_match.group(1).lower(), action_match.group(2).strip()

        if action == "search":
            obs = tool_search(arg)
            state["history"].append(f"Observation: {obs}")
        elif action == "serpsearch":
            obs = tool_serp(arg)
            state["history"].append(f"Observation: {obs}")
        # --- Handle Ntfy Action ---
        elif action == "ntfy":
            obs = tool_ntfy(arg)
            state["history"].append(f"Observation: {obs}")
        # --- Handle Finalize Action ---
        elif action == "finalize":
            state["final"] = arg
            yield state["final"], "\n".join(state["history"])
            break

        yield None, "\n".join(state["history"])


# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("#ReAct Agent with Claude + VectorDB + SerpAPI + ntfy (Streaming)")

    chatbot = gr.Chatbot(label="Agent Trace")
    query = gr.Textbox(label="Ask something", placeholder="e.g. latest iphone news")

    def respond(user_input, chat_history):
        chat_history.append(("User: " + user_input, ""))
        for final, trace in react_agent(user_input):
            if final:  # final answer
                chat_history[-1] = (
                    chat_history[-1][0],
                    f"**Final Answer:** {final}\n\n---\n**Trace:**\n{trace}"
                )
                yield chat_history
            else:  # intermediate trace
                chat_history[-1] = (
                    chat_history[-1][0],
                    f"Working...\n\n**Trace so far:**\n{trace}"
                )
                yield chat_history

    query.submit(respond, [query, chatbot], [chatbot])

if __name__ == "__main__":
    demo.launch()