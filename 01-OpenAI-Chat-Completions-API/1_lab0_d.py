from dotenv import load_dotenv 
import os
from openai import OpenAI

# --- Load environment (optional, not required for Ollama) ---
load_dotenv(override=True)

# --- Point OpenAI SDK to Ollama server ---
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# --- Conversation history ---
messages = []

# --- Part 1: Ask for a riddle ---
messages.append({"role": "user", "content": "Tell me a riddle about animals."})
response = client.chat.completions.create(
    model="llama3.2:latest",
    messages=messages
)
riddle = response.choices[0].message.content
messages.append({"role": "assistant", "content": riddle})

print("\nRiddle from Llama:", riddle)

# --- Part 2: Ask the model to solve its own riddle ---
messages.append({"role": "user", "content": f"Here is the riddle: {riddle}. What is the answer?"})
response = client.chat.completions.create(
    model="llama3.2:latest",
    messages=messages
)
answer = response.choices[0].message.content
messages.append({"role": "assistant", "content": answer})

print("\nAnswer from Llama:", answer)

# --- Part 3: Ask the model to turn it into a story ---
messages.append({
    "role": "user", 
    "content": f"Now take the riddle and its answer and turn them into a fun short story for kids."
})
response = client.chat.completions.create(
    model="llama3.2:latest",
    messages=messages
)
story = response.choices[0].message.content
messages.append({"role": "assistant", "content": story})

print("\nStory from Llama:\n", story)

# --- Final: Print full conversation history ---
print("\n=== Full Conversation History ===")
for m in messages:
    print(m)
