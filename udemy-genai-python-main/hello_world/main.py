from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()    # Load environment variables from a .env file means it help to read the .env file and set the environment variables

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

response = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "https://your-site.com",  # Optional
        "X-Title": "Your App Name",  # Optional
    },
    model="openai/gpt-4o",
    messages=[
        { "role": "user", "content": "Hey, I am Piyush Garg! Nice to meet you! tell me the places in pune where i can visit?" }
    ]
)

print(response.choices[0].message.content)