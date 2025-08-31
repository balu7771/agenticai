##
# Start of our Agentic AI journey
##

from dotenv import load_dotenv

load_dotenv(override=True)

import os
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set - please check")
    

# pip install openai
from openai import OpenAI
openai = OpenAI()

messages = [{"role": "user", "content": "What is 2+2?"}]

# openai.chat.completions is the OpenAI SDK method for generating a response from a chat model
#  such as gpt-4.1-nano, so it means: ask a chat model to generate a completion response for
#  the input conversation we pass in
# Returns a ChatCompletion object, which is JSON
response = openai.chat.completions.create(
    model="gpt-4.1-nano",
    messages=messages
)

print(response)
print(response.choices[0].message.content)


# Now let us ask a question
question = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question}]

response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=messages
)

# Save the question returned by the LLM into a variable called question
question = response.choices[0].message.content

print(f"OpenAI Question: {question}")  

# now form a new message list
messages = [{"role": "user", "content": question}]

# Now ask the LLM to answer the question
response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=messages
)

answer = response.choices[0].message.content
print(f"Ollama Answer: {answer} ")

# Now do this using an open source LLM
# Visit https://ollama.com/ and download and install it
# Then open a terminal and run the command ollama pull llama3.2 then ollama list

ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
model_name = "llama3.2:latest"

question = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question}]

response = ollama.chat.completions.create(
    model=model_name, 
    messages=messages
)
question = response.choices[0].message.content

print(f"Ollama Question: {question}")

# now form a new message list
messages = [{"role": "user", "content": question}]

# Now ask the LLM to answer the question
response = ollama.chat.completions.create(
    model=model_name, 
    messages=messages
)

answer = response.choices[0].message.content
print(f"Ollama Answer: {answer}")
