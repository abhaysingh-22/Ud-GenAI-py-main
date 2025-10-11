from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

response = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "https://your-site.com",
        "X-Title": "Image Caption Generator",
    },
    model="openai/gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "Generate a caption for this image in about 50 words" },
                { "type": "image_url", "image_url": {"url": "https://images.pexels.com/photos/879109/pexels-photo-879109.jpeg"} }
            ]
         }
    ]
)

print("Response:", response.choices[0].message.content)