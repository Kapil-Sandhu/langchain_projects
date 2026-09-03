from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

print("This lecture is all about the different types of messages")

llm_model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.9,
    max_tokens = None,
    timeout = None
)

# Human-Message = is the messages which the user send to the chatbot
# AI-message =  is the messages which AI send as a responce to the user quary
# System-message = is the messages which set the nature, role, tone of the Chat bot, 
# and it's like the Notebook setting which we did when creating any notebook or agents
# now lets improve the Chatbot

message = [
    SystemMessage(content="You are the world best AI Engineer who has the in deepth knowledge about the different AI tools and methods"),
    HumanMessage(content="Can you please explain me the Langchain in easy language")
]

ai_responce = llm_model.invoke(message).content
print("\n", ai_responce)

message.append(AIMessage( content = ai_responce) )

print("\n", message)