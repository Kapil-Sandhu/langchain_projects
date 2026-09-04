from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field
from typing import Optional, Literal
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
class Review(BaseModel): 
    customer_name : Optional[str] = None
    rating : int = Field(ge=1, le=5, description= "Customer Rating b/w 1 to 5")
    age : Optional[int] = None
    sentiment : Literal["Positive", "Netural", "Negative"] = Field(description="Return the sentiment of review")
    summary : str = Field(description="give short summary of review and focuse on just review don't add user details")
    pros  : Optional[list[str]] = Field(default=None, description="Write down all pros in the list")
    cons :  Optional[list[str]] = Field(default=None, description="Write down all cons in the list")
    tags : Optional [list [ Literal["Battery", "Display", "Performance", "Camera"] ] ]= Field(default=None, description="Select a tag based on review from these 4 types - battery, performance, display, camera")

parser = PydanticOutputParser(pydantic_object= Review)

template = PromptTemplate(
    template = "Give me the name, age, rating and short summary of the review  - \n {review} \n {format_instruction}",
    input_variables= ["review"],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

    
review = "I'm Alex Carter and I just turned 28. Upgrading to the iPhone 17 Pro has been completely fantastic, easily earning 5 stars. The battery life effortlessly powers through two full days of intense use, and the A20 chip runs every demanding application without a single hiccup. The low-light photography is unmatched, producing crisp and vibrant shots every time."

print(template.invoke({'review' : review}))


#chain = template | model | parser
#chain_result = chain.invoke(review)
#print(chain_result)