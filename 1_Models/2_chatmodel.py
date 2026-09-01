from langchain_google_genai import ChatGoogleGenerativeAI # for instruct-models
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None, # it is used to set the maximum number of tokens to generate in the response. If set to None, it will use the model's default limit
    timeout=None,
    max_retries=2,
    # other params...
)

result  =  model.invoke("What is the capital of India?")
print(result)

print("\n" + result.content)