from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.9,
    max_tokens = None,
    timeout = None
)

chat_templete = ChatPromptTemplate.from_messages(
    #messages= [
    #    SystemMessage(content="You are a helpful {domain} expert"),
    #    HumanMessage(content="Explain in the {topic} in  4-5 lines please")
    #],
    # this message format is not working because we are not able to use invoke for this templeate so we can not use System message and others here
    #input_variables = ['domain','topic'],
    #validate_template = True
    [
        ('system', 'You are a helpful {domain} expert'),
        ('user', 'Explain in the {topic} in  4-5 lines please')
    ]   
)

prompt = chat_templete.invoke({
    'domain' : 'LLMs and AI',
    'topic' : 'Langchain'
})
print(prompt)

result = model.invoke(prompt).content
print("\n",result)
