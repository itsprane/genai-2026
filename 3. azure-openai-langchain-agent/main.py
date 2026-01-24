import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

model = ChatOpenAI(
    model=os.getenv("AZURE_MODEL_NAME"),
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_ENDPOINT")
)

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]

agent = create_agent(
    model, 
    tools=tools, 
    system_prompt="You are a helpful assistant."
)

inputs = {"messages": [{"role": "user", "content": "How many letters are in the word 'Pranesh'?"}]}
response = agent.invoke(inputs)

final_message = response["messages"][-1]
print(f"\nFinal Answer: {final_message.content}")
