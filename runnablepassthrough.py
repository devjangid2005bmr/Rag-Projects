from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableLambda , RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

parser = StrOutputParser(
    
)

code_prompt= ChatPromptTemplate.from_messages([
    ("system","you are a code generator"),
    ("human","{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helpfull assistant who explain code in simple terms"),
    ("human","Explain the following code in simple simple words:\n{code}")
])