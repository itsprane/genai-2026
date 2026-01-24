import os
import json
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
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

def stream_agent(agent, inputs, agent_name):
    print(f"--- {agent_name} Streaming Start ---")
    final_content = ""
    for chunk in agent.stream(inputs, stream_mode="updates"):
        for node, data in chunk.items():
            print(f"\n[Node: {node}]")
            # If the node has messages, let's print the latest one
            if "messages" in data:
                msg = data["messages"][-1]
                if msg.content:
                    print(f"Content: {msg.content}")
                    final_content = msg.content
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"Tool Calls: {msg.tool_calls}")
    print(f"\n--- {agent_name} Streaming End ---\n")
    return final_content

# --- Orchestration with Streaming ---
word_to_check = "Pranesh"
print(f"Checking word: {word_to_check}\n")

# 1. Researcher finds the facts (Streaming)
research_inputs = {"messages": [HumanMessage(content=f"Get info for the word '{word_to_check}'")]}
facts = stream_agent(researcher, research_inputs, "Researcher")

# 2. Writer turns facts into poetry (Streaming)
writer_inputs = {
    "messages": [
        AIMessage(content=facts),
        HumanMessage(content="Write a beautiful poem based on these findings.")
    ]
}
poem = stream_agent(writer, writer_inputs, "Writer")

print(f"Final Outcome:\n{poem}")
