import os
import json
import requests
from typing import Annotated
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv(Path(__file__).with_name(".env"))

EXCHANGE_API_KEY = os.getenv("apikey")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> dict:
    """This function fetches the currency conversion factor between a given base currency and a target currency"""
    url = f'https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """given a currency conversion rate this function calculates the target currency value from a given base currency value"""
    return base_currency_value * conversion_rate

tools = [get_conversion_factor, convert]

llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-2.4T-A95B",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.1
)

llm = ChatHuggingFace(llm=llm_endpoint)
llm_with_tools = llm.bind_tools(tools)

# Using explicit HumanMessage object instead of a raw string
query = HumanMessage(content="whats the conversation factor between usd and pkr and convert 20 usd into pkr?")
messages = [query]

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)


conversion_rate = None
for tool_call in ai_message.tool_calls:
    # execute the 1st tool and get the value of conversion rate
    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        # fetch this conversion rate
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
        # append this tool message to messages list
        messages.append(tool_message1)
    # execute the 2nd tool using the conversion rate from tool 1
    if tool_call['name'] == 'convert':
        # fetch the current arg
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

final_response = llm_with_tools.invoke(messages)
print("\nFinal Answer:\n", final_response.content)