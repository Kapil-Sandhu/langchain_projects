from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    max_tokens = None,
    temperature = 0.9,
    timeout = None
)

parser = JsonOutputParser() # there is a big draback of using json parser and that is we can not inforce a schema on the output, so llm will make the schema
# at his own but sometimes even with detailed prompt the schema is not perfect of what we need

review = "I'm Alex Carter and I just turned 28. Upgrading to the iPhone 17 Pro has been completely fantastic, easily earning 5 stars. The battery life effortlessly powers through two full days of intense use, and the A20 chip runs every demanding application without a single hiccup. The low-light photography is unmatched, producing crisp and vibrant shots every time."

template = PromptTemplate(
    template = "Give me the name, age, rating and short summary of the review  - \n {review} \n {format_instruction}",
    input_variables= ["review"],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

#prompt = template.invoke( {"review" : review} )
#result = model.invoke(prompt)
#print(result.content)

#final_result = parser.parse(result.content)
#print(final_result)
#print(type(final_result))
#print(final_result["name"]) # as its a json object so we can directly use featurese as dict

"""
```json
{
  "name": "Alex Carter",
  "age": 28,
  "rating": 5,
  "short_summary": "Alex finds the iPhone 17 Pro completely fantastic, praising its excellent two-day battery life, powerful A20 chip performance, and unmatched low-light photography."
}
```
"""

chain = template | model | parser
chain_result = chain.invoke({'review': review})
print(chain_result['name'])