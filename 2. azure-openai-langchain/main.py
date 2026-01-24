import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

endpoint = os.getenv("AZURE_ENDPOINT")
# LangChain's AzureChatOpenAI expects the base URL without /openai/v1/
if endpoint and endpoint.endswith("/openai/v1/"):
    endpoint = endpoint.replace("/openai/v1/", "")

# Initialize Azure Chat OpenAI
llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    openai_api_version="2023-05-15",
    api_key=os.getenv("AZURE_API_KEY"),
)

# Invoke the model
response = llm.invoke("What is the capital of France?")

# Print the content of the response
print(response.content)
