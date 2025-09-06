# Analyze an image
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    instructions="Talk like a drunk person.",
    input="Are semicolons optional in JavaScript?",
)

print(response.output_text)