import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../", ".env"), override=True)

# -------------------------------------------------------------------------------------
# 1. Define a Custom State
# Instead of just "messages", we can add arbitrary fields to track data throughout the graph.
# -------------------------------------------------------------------------------------
class CustomerServiceState(TypedDict):
    # The conversation history (uses the 'add_messages' reducer to append inherently)
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Custom state fields that nodes can read and modify
    user_name: str
    issue_type: str        # e.g., "technical", "billing", "unknown"
    is_resolved: bool
    sentiment: str         # e.g., "positive", "negative", "neutral"


# 2. Initialize the Groq model
llm = ChatGroq(
    model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


# -------------------------------------------------------------------------------------
# 3. Define Nodes that Read and Modify the State
# -------------------------------------------------------------------------------------

def sentiment_analysis_node(state: CustomerServiceState):
    """Analyzes the sentiment of the latest user message and updates the state."""
    latest_msg = state["messages"][-1].content
    
    # Simple prompt to the LLM to classify sentiment
    prompt = f"Analyze the sentiment of this text and reply with only ONE word ('positive', 'negative', or 'neutral'): {latest_msg}"
    response = llm.invoke([HumanMessage(content=prompt)])
    sentiment = response.content.strip().lower()
    
    # We return ONLY the fields we want to update in the state
    return {"sentiment": sentiment}


def triage_node(state: CustomerServiceState):
    """Determines the issue type and extracts the user's name if provided."""
    latest_msg = state["messages"][-1].content
    
    prompt = f"""
    Analyze the following customer message: "{latest_msg}"
    
    Task 1: If the user provides their name, extract it. Otherwise, return the current name '{state.get('user_name', 'Unknown')}'.
    Task 2: Classify the issue as 'technical', 'billing', or 'unknown'.
    
    Reply in exact format:
    NAME: <name>
    ISSUE: <issue_type>
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    lines = response.content.strip().split('\n')
    
    name = state.get("user_name", "Unknown")
    issue = "unknown"
    
    for line in lines:
        if line.startswith("NAME:"):
            name = line.split("NAME:")[1].strip()
        elif line.startswith("ISSUE:"):
            issue = line.split("ISSUE:")[1].strip().lower()
            
    # Update the state with the extracted info
    return {"user_name": name, "issue_type": issue}


def response_node(state: CustomerServiceState):
    """Generates the final response based on the fully populated state."""
    name = state.get("user_name", "Unknown")
    issue = state.get("issue_type", "unknown")
    sentiment = state.get("sentiment", "neutral")
    
    system_prompt = f"""
    You are a helpful customer service AI.
    - The customer's name is: {name}
    - They are experiencing a '{issue}' issue.
    - Their current sentiment is: {sentiment}
    
    If their sentiment is negative, be extra apologetic.
    If you don't know their name, politely ask for it.
    If the issue is unknown, ask for more details.
    """
    
    # Combine the system prompt with the actual conversation history
    messages_to_send = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = llm.invoke(messages_to_send)
    
    # By returning under "messages", add_messages reducer appends it to the history
    return {"messages": [response]}


def should_continue(state: CustomerServiceState):
    """A conditional edge based on state."""
    # We could route differently based on issue_type, but here we just go to the response node.
    return "respond"


# -------------------------------------------------------------------------------------
# 4. Build the Graph
# -------------------------------------------------------------------------------------
def run_state_demo():
    print("Building LangGraph Pipeline with Custom State...")
    
    graph_builder = StateGraph(CustomerServiceState)

    # Add our 3 nodes
    graph_builder.add_node("analyze_sentiment", sentiment_analysis_node)
    graph_builder.add_node("triage", triage_node)
    graph_builder.add_node("respond", response_node)

    # The flow: 
    # START -> analyze_sentiment -> triage -> respond -> END
    graph_builder.add_edge(START, "analyze_sentiment")
    graph_builder.add_edge("analyze_sentiment", "triage")
    graph_builder.add_edge("triage", "respond")
    graph_builder.add_edge("respond", END)

    app = graph_builder.compile()
    print("Pipeline built successfully!\n")

    # -------------------------------------------------------------------------------------
    # 5. Interactive Chat Loop
    # -------------------------------------------------------------------------------------
    print("Welcome to the State-Aware Customer Service Bot!")
    print("Type 'quit' or 'exit' to exit.\n")
    
    # Initialize the custom state variables empty
    current_state = {
        "messages": [],
        "user_name": "Unknown",
        "issue_type": "unknown",
        "is_resolved": False,
        "sentiment": "neutral"
    }

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input.strip():
                continue

            # Update our tracked messages with the human input
            current_state["messages"].append(HumanMessage(content=user_input))

            # Stream updates from the graph execution
            events = app.stream(current_state, stream_mode="updates")
            
            for event in events:
                for node_name, state_update in event.items():
                    # Update our local current_state dict with whatever the node returned
                    # so we persist the state across turns in the while loop.
                    if "user_name" in state_update:
                        current_state["user_name"] = state_update["user_name"]
                    if "issue_type" in state_update:
                        current_state["issue_type"] = state_update["issue_type"]
                    if "sentiment" in state_update:
                        current_state["sentiment"] = state_update["sentiment"]
                        print(f"   [Debug] Sentiment node labeled this as: {state_update['sentiment']}")
                    
                    if "messages" in state_update:
                        latest_msg = state_update["messages"][-1]
                        current_state["messages"].append(latest_msg)
                        if latest_msg.type == "ai":
                            print(f"\nGroq Bot: {latest_msg.content}")
                            
            print(f"   [Debug] Current State Memory: Name={current_state['user_name']}, Issue={current_state['issue_type']}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_state_demo()
