from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv
from customers_review import review_database

load_dotenv()

# TypeDict does not have any data validation so that's why if a llm hallusionate and did not follow the schema then,
# then we can get errors and that's why we need other techniques like pydyantic

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.2,
    max_tokens = None,
    timeout = None
)

# review schema
class Review(TypedDict): # llm will automatically extract the information form the review based on the tag name like sentiment
    # but sometimes we wanted to tell llm what excetly we wanted so we can use Anotated schema
    #short_summary : str # without Annotated it will give you full review as it is or you have to change name to short summary
    # the llm is not able to follow the schema of the output specially for the tags
    customer_name : str
    rating : int
    age : int
    # sentiment : str
    sentiment : Annotated[ Literal["Positive", "Netural", "Negative"], "Return the sentiment of review"]
    
    summary : Annotated[str, "give short summary of review and focuse on just review don't add user details"]
    #tags : Annotated[Optional[list[str]] , "Select a tag based on review from these 4 types - battery, performance, display, camera"]
    
    # Optional tell the llm that this is optional if you found then ok otherwise no problem which makes the schema flexible
    #pros  : Annotated[Optional[list[str]] , "Write down all pros in the list"]
    #cons : Annotated[Optional[list[str]] , "Write down all cons in the list"]
    
    # We know that the llm is not following the instruction well for the tags schema so we have a Litral tag for this to force llm to follow schema
    tags : Annotated[Optional [list [Literal["Battery", "Display", "Performanc", "Camera"] ] ]  , "Select a tag based on review from these 4 types - battery, performance, display, camera"]
    
    
structured_model = model.with_structured_output(Review)
    
user_review = " My name is Chloe Bennett, I just celebrated my 26th birthday with this phone, and I rate it 5 stars! The edge-to-edge screen is absolutely stunning for watching movies and scrolling through media during my downtime. Gaming on the new A20 chip is perfectly smooth, and I haven't noticed any thermal throttling even after hours of playing heavy graphics games with my friends."
user_structured_review = structured_model.invoke(user_review)
print(user_structured_review)

#user_review_dataset = []

#for i in range (0,10):
#    review = structured_model.invoke(review_database[i])
#    user_review_dataset.append(review)
#   print("\n", review)
    
#print("\n",user_review_dataset)