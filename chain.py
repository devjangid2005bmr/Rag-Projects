from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
#model

model = ChatGoogleGenerativeAI(model ="gemini-3.8-flash")

#prompt template 

prompt_template = ChatPromptTemplate.from_template(
    "explain {topic} in simple words with the perfect bullet points and the use real examples"
)

#output parser

parser = StrOutputParser()

#chain (everything connected automatically)

chain = prompt_template | model | parser

#run the chain 
result = chain.invoke({"topic":"machine Learning"})

print(result)
