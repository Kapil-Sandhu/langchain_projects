from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

template1 = PromptTemplate(
    template = "Write a detailed report on {topic} \n",
    input_variables= ["topic"]
)

template2 = PromptTemplate(
    template = "Write a summary in 4-5 lines of this {text} \n",
    input_variables= ["text"]
)

parser = StrOutputParser() # this output us a string and this is very useful when we are working with chains
# because chat output contains a lot of meta data which we can't feed to the next chat so it give use just the content of model output

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : "LLMs"})

print(result)