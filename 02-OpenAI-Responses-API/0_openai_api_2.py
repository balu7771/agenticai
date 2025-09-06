# Analyze an image
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

from openai import OpenAI
client = OpenAI()

file = client.files.create(
    file=open("C:\\code\\agentic_ai\\1_foundations\\animals.pdf", "rb"),
    purpose="user_data"
)

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": file.id,
                },
                {
                    "type": "input_text",
                    "text": "Which animals are these?",
                },
            ]
        }
    ]
)

print(response.output_text)