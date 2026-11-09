from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from rich import print

@tool 
def get_text_lenght(text: str) -> int:
    """Returns the number of character in a given text """
    return len(text)

llm = ChatGoogleGenerativeAI(
    model= 'gemini-3.8-flash'
)

llm_with_tool = llm.bind_tools([get_text_lenght])
    
result = llm.invoke("hello")

print(result)