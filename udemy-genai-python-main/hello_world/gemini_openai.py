from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# redirecting to the Gemini API endpoint using gemini openai compatable api this means we are using the openai library to access the gemini model

client = OpenAI(
    api_key="AIzaSyCyuvn83UO2cPY7T6aH74qpiwI8oA6uBSc",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        { "role": "system", "content": "You are an expert in Maths and only and only ans maths realted questions. That if the query is not related to maths. Just say sorry and do not ans that." },
        { "role": "user", "content": "Hey, can you help me solve the a + b whole square"}
    ]
)

print(response.choices[0].message.content)