import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../", ".env"), override=True)

# 1. Initialize Langfuse Callback Handler
# Langfuse automatically picks up LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
# and LANGFUSE_HOST from the environment variables (loaded by dotenv above).
# If these are missing, it will silently fail or warn depending on version.
langfuse_client = Langfuse()
langfuse_handler = CallbackHandler()

# 2. Define State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 3. Initialize Groq LLM
llm = ChatGroq(
    model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# 4. Define Node
# By accepting 'config' (of type RunnableConfig), LangGraph knows to pass 
# the execution context into this node. This context contains our Langfuse callbacks.
def chatbot(state: State, config: RunnableConfig):
    # Pass the config to the LLM invocation so Langfuse tracks the actual generation
    response = llm.invoke(state["messages"], config=config)
    return {"messages": [response]}

def run_langfuse_demo():
    print("Building LangGraph Pipeline with Langfuse Tracing...")

    # 5. Build Graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    app = graph_builder.compile()
    print("Pipeline built successfully!\n")
    print("NOTE: Make sure your Langfuse keys (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST) are set in the .env file.")
    print("Welcome to the LangGraph + Langfuse Chatbot!")
    print("Type 'quit' or 'exit' to exit.\n")

    # Interactive Loop
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input.strip():
                continue

            # 6. Execution with Tracing
            # We pass our Langfuse handler inside the `config` dictionary. 
            # LangGraph will track the entire execution as a single trace!
            config = {"callbacks": [langfuse_handler]}
            
            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="updates",
                config=config
            )
            
            for event in events:
                for node_name, state_update in event.items():
                    latest_message = state_update["messages"][-1]
                    if latest_message.type == "ai":
                        print(f"Groq Bot: {latest_message.content}")
                        
        except EOFError:
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    # Flush events to Langfuse before exiting to ensure no data is lost
    print("\nFlushing traces to Langfuse...")
    try:
        langfuse_client.flush()
    except Exception as e:
        print(f"Error flushing properly (you might not have keys setup): {e}")
    print("Goodbye!")

if __name__ == "__main__":
    run_langfuse_demo()
