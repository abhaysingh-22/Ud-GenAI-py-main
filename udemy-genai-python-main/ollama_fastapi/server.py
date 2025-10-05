from fastapi import FastAPI, Body
from ollama import Client

app = FastAPI()
client = Client(
    host="http://localhost:11434",
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/contact-us")
def read_root():
    return {"email": "abhaysingh.dev@gmail.com"}

@app.post("/chat")
def chat(
        message: str = Body(..., description="The Message")
):
    response = client.chat(model="gemma:2b", messages=[
        { "role": "user", "content":message  }
    ])

    return { "response": response.message.content }

# see this code is running on http://localhost:8000/docs and for that you need to setup ollama-docker in the pc basically you have to run one of the model locally in your pc