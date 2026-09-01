import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# location for the Hugging Face models downloads
os.environ["HF_HOME"] = "D:/Ai Learning/LangChain/LangChain_Codebase/Local_llm/"

# Define the exact model ID
model_id = "Qwen/Qwen2.5-0.5B-Instruct" 

print("Loading tokenizer and model...")

# Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the Model directly onto the GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto", 
    torch_dtype="auto"  
)

# 3. Create a Transformers Pipeline for Text Generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    #max_new_tokens=1000,
    temperature=0.7, # controls the final logits softmax fuction agressiveness
    # if the temperature is 0 then same output every time you run the same prompt
    # and for higher temperature add some rendomness to the output
)

# 4. Connect the Pipeline to LangChain
LLM = HuggingFacePipeline(pipeline=pipe)

# Wrap it in ChatHuggingFace to properly format chat messages (system/user roles)
chat_model = ChatHuggingFace(llm=LLM)

print("Model loaded successfully! Generating response...")

# 5. Invoke the model
result = chat_model.invoke(
    """ Who are You and what you can do for me? """
    )
print(result.content)