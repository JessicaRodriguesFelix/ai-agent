from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import OllamaLLM

load_dotenv()

# set up an LLM
llm = OllamaLLM(model='llama3.2:1b')
response = llm.invoke("What is the capital of France?")
print(response)