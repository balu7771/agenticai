from dotenv import load_dotenv 
import os
from openai import OpenAI

# --- Load environment ---
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# --- Conversation history ---
messages = []

# Step 1: Ask for a riddle
messages.append({"role": "user", "content": "Tell me a riddle about animals."})
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
riddle = response.choices[0].message.content
messages.append({"role": "assistant", "content": riddle})

print("\nRiddle from OpenAI:", riddle)

# Step 2: Ask the model to solve its own riddle
messages.append({"role": "user", "content": f"Here is the riddle: {riddle}. What is the answer?"})
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
answer = response.choices[0].message.content
messages.append({"role": "assistant", "content": answer})

print("\nAnswer from OpenAI:", answer)

# Step 3: Print the full message history
print("\n=== Full Conversation History ===")
for m in messages:
    print(m)

# --- Bonus: Ask the LLM to turn the riddle into a short story ---
messages.append({"role": "user", "content": f"Please turn this riddle into a 3-sentence story: {riddle}"})
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
story = response.choices[0].message.content
print("\nStory from OpenAI:", story)

# Final message history
print("\n=== Full Conversation History ===")
for m in messages:
    print(m)
