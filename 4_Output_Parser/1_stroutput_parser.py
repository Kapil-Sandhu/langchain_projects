import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


os.environ["HF_HOME"] = "D:/Ai Learning/LangChain/LangChain_Codebase/Local_llm/"
model_id = "Qwen/Qwen2.5-0.5B-Instruct" 

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto", 
    torch_dtype="auto",  
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.7
)

LLM = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=LLM)

print("Model loaded successfully! Generating response...")


template1 = PromptTemplate(
    template = "Write a detailed report on {topic} \n",
    input_variables= ["topic"]
)

template2 = PromptTemplate(
    template = "Write a summary in 4-5 lines of this {text} \n",
    input_variables= ["text"]
)

prompt1 = template1.invoke({'topic' : "Deep Learning"})
result1 = chat_model.invoke(prompt1).content
print(result1)

prompt2 = template2.invoke({'text' : result1})
result2 = chat_model.invoke(prompt2).content

print(result2)