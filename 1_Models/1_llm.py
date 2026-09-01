from langchain_google_genai import GoogleGenerativeAI # for base models
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

result  =  model.invoke("What is the capital of India?")
print(result)
