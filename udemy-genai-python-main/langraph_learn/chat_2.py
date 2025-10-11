from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Optional, Literal
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
import os

# Load environment variables from .env file (like API keys)
load_dotenv()

# Initialize OpenAI client with OpenRouter configuration
# This lets us use different AI models through OpenRouter's API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # Point to OpenRouter instead of OpenAI
    api_key=os.getenv("OPENROUTER_API_KEY"),  # Get API key from .env file
    default_headers={
        "HTTP-Referer": "http://localhost:3000",  # Optional: your app URL
        "X-Title": "LangGraph Learn - Conditional Flow"  # Optional: your app name
    }
)

# Define the structure of data that flows through our graph
# This is like a container holding the user's question, AI's answer, and quality check
class State(TypedDict):
    user_query: str  # The question the user asks
    llm_output: Optional[str]  # The AI's response (optional, starts as None)
    is_good: Optional[bool]  # Whether the response is good quality (optional)

# First node: Get AI response using first model
def chatbot(state: State):
    print("\n\n\nChatBot Node", state)
    response = client.chat.completions.create(
        model="openai/gpt-4o",  # Using GPT-4 through OpenRouter
        messages=[
            { "role": "user", "content": state.get("user_query") }
        ]
    )

    # Save the AI's response in our state
    state["llm_output"] = response.choices[0].message.content
    return state

# Decision node: Decides which path to take next
# This is the "brain" that chooses where to go - either to another AI or end
def evaluate_response(state: State) -> Literal["chatbot_gemini", "endnode"]:
    # Literal means: this function can ONLY return one of these two exact strings
    # It's like a traffic signal that can only show "go left" or "go right"
    print("\n\n\nevaluate_response Node", state)
    if False:  # Currently set to False, so it will ALWAYS go to chatbot_gemini
        return "endnode"  # If True, skip to end
    
    return "chatbot_gemini"  # Go to the second AI for another response

# Second node: Get AI response using a different model (alternative AI)
def chatbot_gemini(state: State):
    print("\n\n\nchatbot_gemini Node", state)
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",  # Using a different GPT model through OpenRouter
        messages=[
            { "role": "user", "content": state.get("user_query") }
        ]
    )

    # Update the state with the new response (replaces the old one)
    state["llm_output"] = response.choices[0].message.content
    return state

# Final node: Just a placeholder before ending
def endnode(state: State):
    print("\n\n\nendnode Node", state)
    return state

# Create the graph builder - this is where we design our flow
graph_builder = StateGraph(State)

# Add all our nodes (steps) to the graph
graph_builder.add_node("chatbot", chatbot)  # First AI
graph_builder.add_node("chatbot_gemini", chatbot_gemini)  # Second AI (alternative model)
graph_builder.add_node("endnode", endnode)  # Final step before ending

# Connect the nodes with edges (arrows showing the flow)
# START → chatbot (always go here first)
graph_builder.add_edge(START, "chatbot")

# After chatbot, use conditional logic to decide where to go next
# The evaluate_response function decides: go to "chatbot_gemini" or "endnode"
graph_builder.add_conditional_edges("chatbot", evaluate_response)

# If we went to chatbot_gemini, then go to endnode
graph_builder.add_edge("chatbot_gemini", "endnode")

# From endnode, we're done (go to END)
graph_builder.add_edge("endnode", END)

# Build the final graph (make it ready to use)
graph = graph_builder.compile()

# Get user input from the terminal
print("\n" + "="*50)
print("Welcome to LangGraph Chatbot!")
print("="*50 + "\n")
user_question = input("Ask me anything: ")

# Run the graph with the user's question
# Flow: START → chatbot → evaluate_response → chatbot_gemini → endnode → END
print("\n" + "-"*50)
print("Processing your question through the AI flow...")
print("-"*50 + "\n")

updated_state = graph.invoke(State({"user_query": user_question}))

print("\n" + "="*50)
print("FINAL RESULT:")
print("="*50)
print(f"\nYour Question: {updated_state['user_query']}")
print(f"\nAI Response: {updated_state['llm_output']}")
print("\n" + "="*50)

# HOW IT WORKS STEP BY STEP:
# =============================
# 
# Flow: (START) → chatbot → [evaluate_response decides] → chatbot_gemini → endnode → (END)
# 
# Step 1: We start with this state
#   state = { "user_query": "Hey, What is 2 + 2?", "llm_output": None, "is_good": None }
# 
# Step 2: The "chatbot" node runs
#   - Sends question to first AI (GPT-4)
#   - Gets response like "2 + 2 equals 4"
#   - state = { "user_query": "Hey, What is 2 + 2?", "llm_output": "2 + 2 equals 4", "is_good": None }
# 
# Step 3: The "evaluate_response" function runs (DECISION POINT!)
#   - Currently always returns "chatbot_gemini" (because if False: is always False)
#   - If you change False to True, it would skip to "endnode" instead
#   - This is like a fork in the road - code decides which path to take
# 
# Step 4: The "chatbot_gemini" node runs (because evaluate_response chose this path)
#   - Sends SAME question to second AI (GPT-4 Mini)
#   - Gets a different response (might be similar or slightly different)
#   - REPLACES the old response in state
#   - state = { "user_query": "Hey, What is 2 + 2?", "llm_output": "The answer is 4!", "is_good": None }
# 
# Step 5: The "endnode" node runs
#   - Just prints the state, doesn't change anything
#   - This is a placeholder before finishing
# 
# Step 6: We reach END
#   - The final state is returned and printed
#
# KEY CONCEPT: "Conditional Edges"
# ================================
# Unlike chat.py which always follows the same path, this graph can take DIFFERENT paths
# based on logic (the evaluate_response function). This is useful for:
# - Checking if an answer is good enough, or needs to be regenerated
# - Routing to different AI models based on the question type
# - Adding quality control or validation steps