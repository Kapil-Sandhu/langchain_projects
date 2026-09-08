from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import Literal
from pydantic import BaseModel, Field
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    max_tokens = None,
    temperature = 0.8,
    timeout = None
)

parser = StrOutputParser()

class review_sentiment(BaseModel):
    sentiment : Literal ["Positive", "Negative"] = Field(description="Classify the sentiment of the review")

parser2 = PydanticOutputParser(pydantic_object= review_sentiment)

prompt = PromptTemplate(
    template="Classify the sentiment of the following review into Positive or Negative \n {review} \n {format_instruction}",
    input_variables=['review'],
    partial_variables={'format_instruction' : parser2.get_format_instructions()}
) # the output of this prompt can be anything if AI hallusinate and as the condition depends on the sentiment type so it must be strucutred ouput


user_review = "This is a terrible Smartphone and total waste of money"

classification_chain = prompt| model | parser2

#result = classification_chain.invoke( user_review).sentiment
#print(result)

prompt2 = PromptTemplate(
    template="Give user an appropiate responce for this positive review \n {review}",
    input_variables=['review'],
)
prompt3 = PromptTemplate(
    template="Give user an appropiate responce for this negaitve review \n {review}",
    input_variables=['review'],
)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "Positive", prompt2 | model | parser),
    (lambda x:x.sentiment == "Negative", prompt3 | model | parser),
    RunnableLambda( lambda x : "cound not find the sentiment" ),
    #'review': lambda x: x['review'] # we can not pass the review to the conditional chains in this way
)

#chain = classification_chain | branch_chain # we can not pass the review to the conditional chains in this way
chain = (
    {
        "sentiment": classification_chain,
        "review": lambda x: x["review"]
    }
    | branch_chain
)
result = chain.invoke(user_review)
print(result)