from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
#from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen3-14B",
    task = "text-generation"
)

model = ChatHuggingFace(
    llm=LLM,
    role = "user"
)
parser = JsonOutputParser() # Structured output parser is not used now in langchain so we are going to use the pydaticoutputparser()

review = "I'm Alex Carter and I just turned 28. Upgrading to the iPhone 17 Pro has been completely fantastic, easily earning 5 stars. The battery life effortlessly powers through two full days of intense use, and the A20 chip runs every demanding application without a single hiccup. The low-light photography is unmatched, producing crisp and vibrant shots every time."

template = PromptTemplate(
    template = "Give me the name, age, rating and short summary of the review  - \n {review} \n {format_instruction}",
    input_variables= ["review"],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

chain = template | model
chain_result = chain.invoke({'review': review}).content
print(chain_result)