from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import numpy as np

from agents import Agent, Runner, FunctionTool

# --- Load environment ---
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# --- Knowledge base ---
knowledge_base = {
    "shipping_time": "Our standard shipping time is 3-5 business days.",
    "return_policy": "You can return any product within 30 days of delivery.",
    "warranty": "All products come with a one-year warranty covering manufacturing defects.",
    "payment_methods": "We accept credit cards, debit cards, and PayPal.",
    "customer_support": "You can reach our support team 24/7 via email or chat."
}

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

# Just for example, printing some embedding results
sample_sentence = "Our standard shipping time is 3-5 business days."
embedding = compute_embedding(sample_sentence)
print(f"Printing embedding for {sample_sentence} ...")
print(embedding)
print(len(embedding))  # Should print 1536


# --- Function to compute cosine similarity ---
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# --- Tool callback function ---
async def faq_invoker(tool_context, params):
    args = json.loads(params) if isinstance(params, str) else params
    user_query = args.get("topic", "")

    # Get embedding of user query
    query_embedding = compute_embedding(user_query)

    # Find the topic with highest similarity
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
    description="Provides answers to frequently asked customer support questions using semantic similarity.",
    params_json_schema=faq_schema,
    on_invoke_tool=faq_invoker
)

# --- Define Agent ---
faq_agent = Agent(
    name="Customer Support Bot",
    instructions="You are a helpful customer support assistant. "
                 "Answer questions based on the knowledge base or use the provided FAQ tool with semantic understanding.",
    tools=[faq_tool]
)

# --- Chat function ---
async def chat_with_support(message):
    session = await Runner.run(faq_agent, message)
    return session.final_output

# --- Main loop for testing ---
if __name__ == "__main__":
    import asyncio

    async def main():
        print("Customer Support Bot is running. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting.")
                break
            response = await chat_with_support(user_input)
            print("Bot:", response)

    asyncio.run(main())
