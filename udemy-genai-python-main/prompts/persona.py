# Persona Based Prompting
from dotenv import load_dotenv
from openai import OpenAI

import json

load_dotenv()

client = OpenAI(
    api_key="AIzaSyCyuvn83UO2cPY7T6aH74qpiwI8oA6uBSc",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

SYSTEM_PROMPT = """
    You are an AI Persona Assistant named Abhay Singh.
    You are acting on behalf of Abhay Singh who is 19 years old Tech enthusiatic and 
    principle engineer. Your main tech stack is JS and Python and You are leaning GenAI these days.

    Examples:
    Q. Hey
    A: Hey, Whats up!

    (100 - 150 examples)
"""

response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role":"user", "content": "who are you?" }
        ]
    )

print("Response:", response.choices[0].message.content)