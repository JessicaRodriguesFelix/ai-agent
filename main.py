from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv() # might not be required anymore since we are using llma


# response = llm.invoke("What is the capital of France?")
# print(response)

# setup a prompt template
class ResearchResponseModel(BaseModel):
    topic: str
    summary: str
    sources: list[str] = []
    tools_used: list[str] = []

# set up an LLM
llm = ChatOllama(model='llama3.1:8b')

# parser takes the output from the llm and parses it into the model created, this can be later used as a python object inside the code
myParser = PydanticOutputParser(pydantic_object=ResearchResponseModel)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=myParser.get_format_instructions())


agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=[])

agent_executor = AgentExecutor(agent=agent, tools=[],verbose=True)
# run the agent executor with a query
raw_response = agent_executor.invoke({"query": "What is the capital of France?"})
print(raw_response)

try:
    structured_response = myParser.parse(raw_response.get('output'))
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw response", raw_response)