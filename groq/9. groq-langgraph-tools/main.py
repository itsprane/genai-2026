import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables from the root .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "../../", ".env"), override=True)

# 1. Define the State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Define Tools
@tool
def get_weather(location: str):
    """Call to get the current weather for a specific location."""
    # In a real app, you would call a weather API here.
    # We are returning mock data for demonstration purposes.
    return f"The weather in {location} is currently 72°F and sunny."

@tool
def calculate_sum(a: int, b: int):
    """Call to calculate the sum of two numbers."""
    return str(a + b)

tools = [get_weather, calculate_sum]
tool_node = ToolNode(tools)

# 3. Initialize the Groq model and bind tools
llm = ChatGroq(
    model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
# We must bind the tools to the LLM so it knows what it can call
llm_with_tools = llm.bind_tools(tools)

# 4. Define the Chatbot Node
def chatbot(state: State):
    """
    This node simply invokes the ChatGroq model with the current messages.
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def run_langgraph_tools_demo():
    print("Building LangGraph Pipeline with Tools...")

    # 5. Build the graph
    graph_builder = StateGraph(State)

    # 6. Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    
    # 7. Add edges
    # START always goes to chatbot
    graph_builder.add_edge(START, "chatbot")
    
    # The chatbot node goes to the 'tools' node IF the LLM decided to call a tool.
    # Otherwise, it goes to END. We use the prebuilt `tools_condition` edge.
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    
    # After a tool is executed, we must return to the chatbot so the LLM 
    # can look at the tool's output and decide what to say next.
    graph_builder.add_edge("tools", "chatbot")

    # 8. Compile the graph
    app = graph_builder.compile()
    print("Pipeline built successfully!\n")

    # 9. Interactive Chat Loop
    print("Welcome to the LangGraph Groq Chatbot with Tools!")
    print("Try asking for the weather or doing some math.")
    print("Type 'quit' or 'exit' to exit.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue

            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="updates"
            )
            
            for event in events:
                # `event` is a dict mapping from node_name -> state update
                # e.g. {"chatbot": {"messages": [...]}} or {"tools": {"messages": [...]}}
                for node_name, state_update in event.items():
                    # The nodes specifically return the NEW messages to append
                    # so we just grab the last one since we know it's the new one.
                    latest_message = state_update["messages"][-1]
                    
                    # Check for Tool Calls (The LLM deciding to use a tool)
                    if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                        for tc in latest_message.tool_calls:
                            print(f"-> [Agent is calling tool '{tc['name']}' with args {tc['args']}]")
                    
                    # Check for Tool Messages (The output of the tool execution)
                    elif isinstance(latest_message, ToolMessage):
                        print(f"<- [Tool returned: {latest_message.content}]")
                        
                    # Check for regular AI responses (No tool calls)
                    elif latest_message.type == "ai" and latest_message.content:
                        print(f"Groq Bot: {latest_message.content}")
                    
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_langgraph_tools_demo()
