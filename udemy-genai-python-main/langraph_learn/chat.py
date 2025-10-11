from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import os

# Load environment variables from .env file (like API keys)
load_dotenv()

# Initialize the AI chat model with OpenRouter
# OpenRouter lets us use different AI models through one API
llm = ChatOpenAI(
    model="openai/gpt-4o",  # Using GPT-4 through OpenRouter
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),  # Get API key from .env file
    openai_api_base="https://openrouter.ai/api/v1",  # OpenRouter's endpoint
    default_headers={
        "HTTP-Referer": "http://localhost:3000",  # Optional: your app URL
        "X-Title": "LangGraph Learn"  # Optional: your app name
    }
)

# Define the structure of data that flows through our graph
# Think of this like a container that holds all messages in the conversation
class State(TypedDict):
    # messages is a list that automatically adds new messages to existing ones
    messages: Annotated[list, add_messages]

# First node: sends user's message to AI and gets response
def chatbot(state: State):
    # Get all messages from state and send them to AI
    response = llm.invoke(state.get("messages"))
    # Return the AI's response wrapped in a dictionary
    return { "messages": [response] }

# Second node: a simple example node that adds a custom message
def samplenode(state: State):
    print("\n\nInside samplenode node", state)
    # Add a custom message to the conversation
    return { "messages": ["Sample Message Appended"] }

# Create a graph builder - this helps us connect different steps (nodes)
graph_builder = StateGraph(State)

# Add our two nodes to the graph
# Think of nodes as stops along a journey
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("samplenode", samplenode)

# Connect the nodes in order (like drawing arrows between steps)
# START → chatbot → samplenode → END
graph_builder.add_edge(START, "chatbot")        # First: go from START to chatbot
graph_builder.add_edge("chatbot", "samplenode") # Then: go from chatbot to samplenode
graph_builder.add_edge("samplenode", END)       # Finally: go from samplenode to END

# Build the final graph (make it ready to use)
graph = graph_builder.compile()

# Run the graph with an initial message
# The message flows through: START → chatbot → samplenode → END
updated_state = graph.invoke(State({"messages": ["What is my name?"]}))
print("\n\nupdated_state", updated_state)

# HOW IT WORKS STEP BY STEP:
# =============================
# 
# Flow: (START) → chatbot → samplenode → (END)
# 
# Step 1: We start with this state
#   state = { "messages": ["What is my name?"] }
# 
# Step 2: The "chatbot" node runs
#   - Takes the message "What is my name?"
#   - Sends it to AI (GPT)
#   - AI responds with something like "I don't know your name"
#   - Now state becomes: { "messages": ["What is my name?", "I don't know your name"] }
# 
# Step 3: The "samplenode" node runs
#   - Adds a custom message to the list
#   - Now state becomes: { "messages": ["What is my name?", "I don't know your name", "Sample Message Appended"] }
# 
# Step 4: We reach END
#   - The final state with all 3 messages is returned
#   - This is what gets printed as "updated_state"