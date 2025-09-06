from openai import OpenAI
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()

# 1. Define tool for forex rate lookup
tools = [
    {
        "type": "function",
        "name": "get_forex_rate",
        "description": "Get current forex conversion rate between two currencies.",
        "parameters": {
            "type": "object",
            "properties": {
                "base_currency": {
                    "type": "string",
                    "description": "The base currency code, e.g., USD"
                },
                "target_currency": {
                    "type": "string",
                    "description": "The target currency code, e.g., INR"
                },
            },
            "required": ["base_currency", "target_currency"],
        },
    },
]

# 2. Function to actually fetch forex rate
def get_forex_rate(base_currency: str, target_currency: str):
    # Example: using exchangerate.host (free)
    # Visit https://exchangerate.host/ and get a Free API key
    # Add it to .env file as EXCHANGE_RATE_API_KEY=""
    exchange_api_key = os.getenv("EXCHANGE_RATE_API_KEY")
    url = f"https://v6.exchangerate-api.com/v6/{exchange_api_key}/pair/{base_currency}/{target_currency}"
    resp = requests.get(url)
    data = resp.json()
    if "conversion_rate" in data:
        return f"1 {base_currency} = {data['conversion_rate']}"
    return "Error fetching forex rate"

# 3. Prompt the model with tools defined
response = client.responses.create(
    model="gpt-4o-mini",
    tools=tools,
    instructions="Respond only with the forex rate provided by the tool.",
    input="What is the current USD to INR rate?",
)

# Add model output (which may include tool calls) to input list
input_list = response.output.copy()  # start with model output

for item in response.output:
    # Convert Pydantic object to dict
    if not isinstance(item, dict):
        item_dict = item.model_dump()
    else:
        item_dict = item

    # Check for function call
    if item_dict['type'] == "function_call" and item_dict['name'] == "get_forex_rate":
        args = json.loads(item_dict['arguments'])
        forex = get_forex_rate(args["base_currency"], args["target_currency"])

        # Append the tool call output back to input list
        input_list.append({
            "type": "function_call_output",
            "call_id": item_dict['call_id'],
            "output": json.dumps({"forex_rate": forex})
        })


print("Final input:")
print(input_list)

# 5. Final model response using tool output
response = client.responses.create(
    model="gpt-4o-mini",
    instructions="Respond only with the forex rate provided by the tool.",
    tools=tools,
    input=input_list,
)

print("Final output:")
print(response.model_dump_json(indent=2))
print("\n" + response.output_text)
