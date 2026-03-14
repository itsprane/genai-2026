import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment variables from the root .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "../../", ".env"), override=True)

# 1. Define the State
# The state is mainly a list of messages. The `add_messages` reducer ensures 
# that new messages are appended to the list rather than overwriting it.
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Initialize the Groq model
llm = ChatGroq(
    model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# 3. Define the node function
def chatbot(state: State):
    """
    This node simply invokes the ChatGroq model with the current messages.
    """
    # Invoke the model with the sequence of messages in the state
    response = llm.invoke(state["messages"])
    
    # Return the new message. The `add_messages` reducer will append this 
    # to the existing state["messages"] array.
    return {"messages": [response]}

def run_langgraph_demo():
    print("Building simple LangGraph Pipeline with ChatGroq...")

    # 4. Build the graph
    graph_builder = StateGraph(State)

    # 5. Add nodes
    graph_builder.add_node("chatbot", chatbot)
    
    # 6. Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # 7. Compile the graph
    app = graph_builder.compile()

    print("Pipeline built successfully!\n")

    # 8. Use the compiled graph
    print("Welcome to the LangGraph Groq Chatbot!")
    print("Type 'quit' or 'exit' to exit.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue

            # Stream the graph updates
            # stream_mode="values" will yield the full state after each node execution
            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="values"
            )
            
            for event in events:
                # Get the most recent message from the state
                latest_message = event["messages"][-1]
                
                # Check if it's the AI's response (so we don't just echo the human input)
                if latest_message.type == "ai":
                    print(f"Groq Bot: {latest_message.content}")
                    
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_langgraph_demo()
