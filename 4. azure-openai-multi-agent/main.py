import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

# Initialize the model
model = ChatOpenAI(
    model=os.getenv("AZURE_MODEL_NAME"),
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_ENDPOINT")
)

# --- Researcher Agent ---
@tool
def get_word_info(word: str) -> str:
    """Returns technical information about a word (length and vowels)."""
    length = len(word)
    vowels = sum(1 for char in word.lower() if char in "aeiou")
    return f"The word '{word}' has {length} letters and {vowels} vowels."

researcher_sys_msg = SystemMessage(content="You are a Researcher. Use your tool to find technical facts about words.")

researcher = create_agent(
    model,
    tools=[get_word_info],
    system_prompt=researcher_sys_msg
)

# --- Writer Agent ---
writer_sys_msg = SystemMessage(content="You are a Professional Poet. You take technical facts from the assistant and turn them into beautiful poetry.")

writer = create_agent(
    model,
    system_prompt=writer_sys_msg
)

# --- Orchestration using HumanMessage, AIMessage, and SystemMessage ---
word_to_check = "Pranesh"
print(f"Checking word: {word_to_check}\n")

# 1. Researcher finds the facts
research_query = f"Get info for the word '{word_to_check}'"
research_response = researcher.invoke({"messages": [HumanMessage(content=research_query)]})
facts = research_response["messages"][-1].content
print(f"Researcher Output: {facts}\n")

# 2. Writer receives facts as an AIMessage (representing the Researcher's work)
# This simulates the Writer seeing what the Researcher found as its conversation history
writer_history = [
    AIMessage(content=facts), # Here we pass the facts as an AI message
    HumanMessage(content="Write a beautiful poem based on these findings.")
]

writer_response = writer.invoke({"messages": writer_history})
poem = writer_response["messages"][-1].content
print(f"Writer Output:\n{poem}")
