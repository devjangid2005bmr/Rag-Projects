from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from rich import print


#TOOL CREATING
@tool 
def get_text_lenght(text: str) -> int:
    """Returns the number of character in a given text """
    return len(text)

tools ={
    "get_text_lenght": get_text_lenght
}


#LLM 
llm = ChatGoogleGenerativeAI(
    model= 'gemini-3.6-flash'
)
llm_with_tool = llm.bind_tools([get_text_lenght])

message=[]
prompt=input("you:")
query = HumanMessage(prompt)
message.append(query)
print(message)

result=llm_with_tool.invoke(message)

message.append(result)


if result.tool_calls:
    tool_name=result.tool_calls[0]["name"]
    tool_message=tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
    
    
result=llm_with_tool.invoke(message)
print(result.text)