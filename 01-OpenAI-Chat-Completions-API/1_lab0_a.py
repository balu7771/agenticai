from dotenv import load_dotenv
import os
from openai import OpenAI

# --- Load environment ---
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# --- Conversation history ---
messages = []

# --- Ask the LLM a riddle ---
messages = [{"role": "user", "content": "Tell me a riddle about animals."}]
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=messages
)
riddle = response.choices[0].message.content
print("\nRiddle from OpenAI:", riddle)