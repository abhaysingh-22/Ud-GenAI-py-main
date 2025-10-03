# Zero Shot Prompting
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key="AIzaSyCyuvn83UO2cPY7T6aH74qpiwI8oA6uBSc",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# Zero Shot Prompting: Directly giving the inst to the model
SYSTEM_PROMPT = "You should only and only ans the coding related questions. Do not ans anything else. Your name is Alexa. If user asks something other than coding, just say sorry."

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        { "role": "system", "content": SYSTEM_PROMPT },
        { "role": "user", "content": "Hey, Can you write a python code to translate the word hello to Hindi"},
        { "role": "user", "content": "Also, can you tell me the capital of India?"}
    ]
)

print(response.choices[0].message.content)
# 1. Zero-shot Prompting: The model is given a direct question or task without prior examples.