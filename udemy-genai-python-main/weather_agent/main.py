from openai import OpenAI
from dotenv import load_dotenv
import requests
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    
    return "Something went wrong" 


# currently we cannot here directly ask the model the weather status of any city becuase it does not have access to internet so we need to create an agent.


def main():
    user_query = input("> ")
    
    response = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "Weather Agent",
        },
        model="openai/gpt-4o",
        messages=[
            { "role": "user", "content": user_query } 
        ]
    )

    print(f"🤖: {response.choices[0].message.content}")

main()