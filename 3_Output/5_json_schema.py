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
json_schema = {
    "title" : "Customer_Review",
    "type" : "object",
    "properties" : {
        "customer_name" : {
            "type" : ["string", "null"],
            "description" : "Return the Name of the Customer"
        },
        "rating" : {
            "type" : "integer",
            "description" : "Return a number between 1 to 5"
        },
        "age" : {
            "type" :  ["integer", "null"],
            "description" : "Return the age of the customer"
        },
        "sentiment" : {
            "type" : "string",
            "enum" : ["Positive", "Negative", "Neutral"],
            "description" : "Return the sentiment of the review"
        },
        "summary" : {
            "type" : "string",
            "description" : "Return a short summary of the review"
        },
        "pros" : {
            "type" : ["array", "null"],
            "items" : {
                "type" : "string"
                },
            "description" : "Return the Pros of the Item"
        },
        "cons" : {
            "type" : ["array", "null"],
            "items" : {
                "type" : "string"
                },
            "description" : "Return the Cons of the Item"
        },
        "tags" : {
            "type" : ["array", "null"],
            "items" : {
                "type" : "string",
                "enum" : ["Battery", "Display", "Performanc", "Camera"],
                },
            
            "description" : "Return the Pros of the Item"
        },
    },
    "required" : ["rating", "sentiment", "summary"]
}
    
structured_model = model.with_structured_output(json_schema)
    
user_review = "I'm Alex Carter and I just turned 28. Upgrading to the iPhone 17 Pro has been completely fantastic, easily earning 5 stars. The battery life effortlessly powers through two full days of intense use, and the A20 chip runs every demanding application without a single hiccup. The low-light photography is unmatched, producing crisp and vibrant shots every time."
user_structured_review = structured_model.invoke(user_review)
print("\n", user_structured_review)

#user_review_dataset = []

#for i in range (0,10):
#    review = structured_model.invoke(reviews[i])
#    user_review_dataset.append(review)
#    print("\n", review)
    
#print("\n",user_review_dataset)