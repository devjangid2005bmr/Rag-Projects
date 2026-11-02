from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate



search_tool = TavilySearchResults(
    max_result=5
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

prompt = ChatPromptTemplate([
    """you are a helpful assistance
    summaeize the following news into the clear bullet points
    {news}"""]
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.run("latest ai news of 2026")

result = chain.invoke({"news":news_result})


print(result)
