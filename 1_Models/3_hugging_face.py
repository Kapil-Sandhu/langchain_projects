from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Flash-0731",
    task = "text-generation"
)

model = ChatHuggingFace(
    llm=LLM,
    role = "user"
)

result = model.invoke("Hey who are You?")
print(result)
