import os
import warnings
# Suppress Pydantic and LangChain deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# Load environment variables
load_dotenv("../.env")

# Ensure API key is mapped (LangChain requirement)
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    temperature=0.1,
    max_tokens=1000,
)

tools = []

agent = create_agent(
    model, 
    tools=tools, 
    system_prompt="You are a helpful AI assistant."
)

print("Agent: Hello! I am an agent. How can I help you today?")
query = "Explain how AI works in a few words"
print(f"User: {query}")

try:
    response = agent.invoke({"messages": [("user", query)]})
    last_message = response["messages"][-1]
    content = last_message.content

    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and 'text' in part:
                print(f"Agent: {part['text']}")
    else:
        print(f"Agent: {content}")
except Exception as e:
    print(f"\n[!] An error occurred: {e}")
