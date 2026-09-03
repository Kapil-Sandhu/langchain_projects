from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    max_tokens = None,
    temperature = 0.9,
)
# as we can see that the model does not have any context about the previous chat, so it is useless untill we provide this a chat history
chat_history = [
    SystemMessage(content="You are a helpful assistent"),
]

print("\nWelcome to the Kapil's API ChatBot")
while True:
    user_input = input("\nYou - ")
    chat_history.append(HumanMessage( content= user_input) )
    if user_input == "exit":
        break
    else:
        result = model.invoke(chat_history).content
        chat_history.append(AIMessage(content=result))
        print("AI - ", result)
        
print("Thank you for using a API based Chatbot")
print(chat_history)