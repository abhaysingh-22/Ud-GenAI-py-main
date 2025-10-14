from dotenv import load_dotenv
from mem0 import Memory
import os
import json

from openai import OpenAI

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Memory Agent"
    }
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "huggingface",
        "config": { 
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    },
    "llm": {
        "provider": "openai",
        "config": { 
            "api_key": OPENROUTER_API_KEY, 
            "model": "openai/gpt-4o-mini",
            "openai_base_url": "https://openrouter.ai/api/v1"
        }
    },
    # "graph_store":{
    #     "provider": "neo4j",
    #     "config": {
    #         "url": "neo4j://localhost:7687",
    #         "username": "neo4j",
    #         "password": "password123"
    #     }
    # },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 384  # Set dimension for HuggingFace model
        }
    }
}

# see the code is not working because i didn't have setup neo4j DB for storing data of graphs
# note that it can setup locally or you can use neo4j cloud as well best option is to setup on the cloud
# i have not even setup my id on neo4j cloud so it is not working for me

mem_client = Memory.from_config(config)


while True:


    user_query = input("> ")

# this help to retrieve relevant memories
    search_memory = mem_client.search(query=user_query, user_id="abhaysingh",)

    memories = [
        f"ID: {mem.get("id")}\nMemory: {mem.get("memory")}" 
        for mem in search_memory.get("results")
    ]

    print("Found Memories", memories)

    SYSTEM_PROMPT = f"""
        Here is the context about the user:
        {json.dumps(memories)}
    """

# this is where we are asking the model to remember the previous conversation
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_query }
        ]
    )

    ai_response = response.choices[0].message.content

    print("AI:", ai_response)

    mem_client.add(
        user_id="abhaysingh",
        messages=[
            { "role": "user", "content": user_query },
            { "role": "assistant", "content": ai_response }
        ]
    )

    print("Memory has been saved...")