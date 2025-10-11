from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
import os

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "LangGraph Checkpoint"
    }
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return { "messages": [response] }


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def compile_graph_with_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

DB_URI = "mongodb://admin:admin%40123@localhost:27017"
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    graph_with_checkpointer = compile_graph_with_checkpointer(checkpointer=checkpointer)

    config = {
            "configurable": {
                "thread_id": "abhay" # user_id if i change this my chatbot will forget previous conversation and will not remember previous conversation
            }
        }


    for chunk in graph_with_checkpointer.stream(
        State({"messages": ["What is my name?"]}),
        config,
        stream_mode="values"
        ):
            chunk["messages"][-1].pretty_print()
    

# (START) -> chatbot -> (END)

# state = { messages: ["Hey there"] }
# node runs: chatbot(state: ["Hey There"]) -> ["Hi, This is a message from ChatBot Node"]

# Checkpointer (abhay) = Hey, My name is Abhay Singh