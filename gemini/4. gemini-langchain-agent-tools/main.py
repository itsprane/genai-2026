import os
import warnings
# Suppress Pydantic and LangChain deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool

# Load environment variables
load_dotenv("../.env")

# Ensure API key is mapped (LangChain requirement)
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    temperature=0.1,
    max_tokens=1000,
)

# 1. Define a custom tool
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word by counting its characters."""
    return len(word)

tools = [get_word_length]

# 2. Create the agent using the modern factory pattern
agent = create_agent(
    model, 
    tools=tools, 
    system_prompt="You are a helpful AI assistant that uses tools when needed."
)

print("Agent: Hello! I can use tools now. What word should I check for you?")
query = "What is the length of the word 'Antigravity'?"
print(f"User: {query}")

try:
    # 3. Execute the agent graph
    response = agent.invoke({"messages": [("user", query)]})

    # 4. Extract and print the final interaction
    # The graph will have multiple messages if tools were called
    for msg in response["messages"]:
        if msg.type == "ai" and msg.content:
            # Check if it's text content or a tool call
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and 'text' in part:
                        print(f"Agent: {part['text']}")
            else:
                print(f"Agent: {msg.content}")
        elif msg.type == "tool":
            print(f"Tool Result: {msg.content}")

except Exception as e:
    print(f"\n[!] An error occurred: {e}")
