from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt_template1 = PromptTemplate(
    template = "Explain the {topic} in depth with leman terms so that even a child can understand it.",
    input_variables= ['topic'],
    validate_template= True
)

prompt_template2 = PromptTemplate(
    template = "Read this text and summarize this in 4-5 lines \n {report}",
    input_variables= ['report'],
    validate_template= True
)

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    max_tokens = None,
    temperature = 0.8,
    timeout = None
)

parser = StrOutputParser()

chain = prompt_template1 | model | parser | prompt_template2 | model | parser # this is a sequancial chain

chain.get_graph().print_ascii() # to print the chain flowchart to the terminal

user_topic = input("Please Enter the Topic Name which you wanted to understand - \n")
result = chain.invoke({"topic" : user_topic})

print(result)