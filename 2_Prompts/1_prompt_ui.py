from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.header("Research Assistant Tool")
# user_input = st.text_input("Enter Your Prompt") # we are asking the user for the prompt and this is call the static prompt
# but we know that a single char change in the prompt can change the whole answer so LLMs are prompt sensitive and 
# general user does not know how to write a effective prompt thats why we don't use Static prompt too much because 
# it gives user more control over the interaction

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.9,
    max_tokens = None,
    timeout=None
)

# So   the best way is to use the carfully crafted a Dynamic prompt with dynamic tags or names like this

# Please summarize the reseach paper titled "{paper_name}" with the following specifications :
# Explanation Style : {user_style}
# Explanation Length : {output_lenght}
# 1. Mathematical Details :
#   - Include relevent mathematical equations if present in the paper
#   - Explain the mathematical concepts using simple, intuitive code snippets, where applicable
# 2. Analogies :
#   - Use relatable analogies to simplify complex ideas
# If certain information is not avalible in the paper, respond with : "Insufficient information avalible" instead of gussing.
# Ensure the summary is clear, accurate, and aligned with the provided style and lenght.

paper_input = st.selectbox(
    "Select the Research Paper Name",
    ["Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat Gans on Image Synthesis" ]               
    )

output_tone = st.selectbox(
    "Select the Style",
    [   "Beginner-Friendly",
        "Technical",
        "Code-Orinted",
        "Mathematical"
    ]   
)

input_lenght = st.selectbox(
    "Select the Output Lenght",
    [
        "Short (4-5 lines)",
        "Medium (2-3 paragraphs)",
        "Long (detailed explanation)"
    ]
    )

user_template = PromptTemplate(
    template = """
    Please summarize the reseach paper titled '{paper_input}' with the following specifications :
    Explanation Style : {output_tone}
    Explanation Length : {input_lenght}
    1. Mathematical Details :
        - Include relevent mathematical equations if present in the paper
        - Explain the mathematical concepts using simple, intuitive code snippets, where applicable
    2. Analogies :
        - Use relatable analogies to simplify complex ideas
    If certain information is not avalible in the paper, respond with : "Insufficient information avalible" instead of gussing.
    Ensure the summary is clear, accurate, and aligned with the provided style and lenght.
""",
input_variables=['paper_input', 'output_tone','input_lenght']
)

user_prompt = user_template.invoke( {
    'paper_input' : paper_input,
    'output_tone' : output_tone,
    'input_lenght' : input_lenght
} )

if st.button("Summarize") :
    result = model.invoke(user_prompt)
    st.write(result.content)
    
