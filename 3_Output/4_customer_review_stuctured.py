from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv
from customer_dataset_2 import reviews


load_dotenv()


model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.9,
    max_tokens = None,
    timeout = None
)

# review schema
class Review(BaseModel): 
    customer_name : Optional[str] = None
    rating : int = Field(ge=1, le=5, description= "Customer Rating b/w 1 to 5")
    age : Optional[int] = None
    sentiment : Literal["Positive", "Netural", "Negative"] = Field(description="Return the sentiment of review")
    summary : str = Field(description="give short summary of review and focuse on just review don't add user details")
    pros  : Optional[list[str]] = Field(default=None, description="Write down all pros in the list")
    cons :  Optional[list[str]] = Field(default=None, description="Write down all cons in the list")
    tags : Optional [list [ Literal["Battery", "Display", "Performanc", "Camera"] ] ]= Field(default=None, description="Select a tag based on review from these 4 types - battery, performance, display, camera")
    
    
structured_model = model.with_structured_output(Review)
    
#user_review = "I'm Alex Carter and I just turned 28. Upgrading to the iPhone 17 Pro has been completely fantastic, easily earning 5 stars. The battery life effortlessly powers through two full days of intense use, and the A20 chip runs every demanding application without a single hiccup. The low-light photography is unmatched, producing crisp and vibrant shots every time."
#user_structured_review = structured_model.invoke(user_review)
#print("\n", user_structured_review)

user_review_dataset = []

for i in range (0,10):
    review = structured_model.invoke(reviews[i])
    user_review_dataset.append(review)
    print("\n", review)
    
print("\n",user_review_dataset)