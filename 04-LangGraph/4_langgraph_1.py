from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# --- Initialize the LLM ---
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

# --- Define the state structure ---
class GraphState(TypedDict):
    user_name: str
    greeting: str
    messages: list

# --- Define Node 1: Ask for user's name ---
def ask_name_node(state: GraphState) -> GraphState:
    """Node to ask for user's name"""
    # In a real implementation, you might get input from user here
    # For demo purposes, we'll simulate user input
    user_input = input("Bot: What is your name? ")
    
    return {
        **state,
        "user_name": user_input,
        "messages": state.get("messages", []) + [f"User provided name: {user_input}"]
    }

# --- Define Node 2: Greet the user ---
def greet_user_node(state: GraphState) -> GraphState:
    """Node to greet the user"""
    user_name = state.get("user_name", "there")
    greeting = f"Hello {user_name}! Nice to meet you."
    
    return {
        **state,
        "greeting": greeting,
        "messages": state.get("messages", []) + [f"Bot: {greeting}"]
    }

# --- Create the graph ---
def create_greeting_graph():
    # Initialize the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("ask_name", ask_name_node)
    workflow.add_node("greet_user", greet_user_node)
    
    # Add edges
    workflow.add_edge(START, "ask_name")
    workflow.add_edge("ask_name", "greet_user")
    workflow.add_edge("greet_user", END)
    
    # Compile the graph
    return workflow.compile()

# --- Main execution ---
if __name__ == "__main__":
    # Create the graph
    app = create_greeting_graph()
    
    # Initialize state
    initial_state = GraphState(
        user_name="",
        greeting="",
        messages=[]
    )
    
    # Run the graph
    print("Starting conversation...")
    final_state = app.invoke(initial_state)
    
    # Output the results
    print("\nConversation completed!")
    print("Final greeting:", final_state["greeting"])
    print("\nFull conversation:")
    for message in final_state["messages"]:
        print("-", message)