from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv("../.env")

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
chain = llm | StrOutputParser()

response = chain.invoke("Explain how AI works in a few words")
print(response)
