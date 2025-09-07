from openai import OpenAI
from dotenv import load_dotenv
from agents import Agent, Runner

# --- Load environment ---
import os
load_dotenv(override=True)
client = OpenAI()

# --- Define variables ---
instruction = "You are a helpful customer support analyst."
message = "Analyze this customer feedback and suggest improvements to the product: 'The packaging is great, but the delivery was delayed by two days.'"

# --- Create agent ---
agent = Agent(
    name="Assistant",
    instructions=instruction
)

# --- Run agent synchronously ---
result = Runner.run_sync(agent, message)

# --- Print output ---
print(result.final_output)
