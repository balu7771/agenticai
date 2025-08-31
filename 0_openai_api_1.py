# Basic example to ask a model to generate a short story

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="Write a short story of two lines about a robot who wanted to learn painting."
)

print(response)

# From OpenAI documentation: Some of our official SDKs include an output_text property 
#   on model responses for convenience, which aggregates all text outputs from the model 
#   into a single string. This may be useful as a shortcut to 
#   access text output from the model.
print(response.output_text)