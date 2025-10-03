from google import genai

client = genai.Client(
    api_key="AIzaSyCyuvn83UO2cPY7T6aH74qpiwI8oA6uBSc"
)

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="tell me the places in pune which i can visit?"
)

print(response.text)