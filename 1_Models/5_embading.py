from langchain_google_genai import GoogleGenerativeAIEmbeddings # this is used for embedding Models
from dotenv import load_dotenv

load_dotenv()

emb = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality = 768, # embedding dimesions
)

vector = emb.embed_query("India")
print(vector)
