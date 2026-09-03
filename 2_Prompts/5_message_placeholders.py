from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import ast

load_dotenv()

# create the model
model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.9,
    max_tokens = None,
    timeout = None
)

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{quary}')
])


# load chat history
chat_history = []
with open('2_Prompts/chat_history.txt') as f: # give relative path position starting form where is you vene not your current run file
    for line in f:
        # chatPrompt templete need a tuple but direct f.readlines() give us a string so we have to use ast
        parsed_tuple = ast.literal_eval(line.strip())
        chat_history.append(parsed_tuple)
    
print(chat_history)


# create prompt
prompt = chat_template.invoke({
    'chat_history' : chat_history,
    'quary' : 'Hey, yesterday i request for a refund and i did not get it till now can you check please'
})
print(prompt)

result = model.invoke(prompt).content
print("\n",result)