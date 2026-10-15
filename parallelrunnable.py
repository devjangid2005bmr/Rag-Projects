from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser



#prompt template



#model
model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash"
)

#output parser
parser = StrOutputParser()


short_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in the simple way in only 1-2 lines"
    
)


detailed_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in the deep dive and the best for understanding and explain in brif with points , real examples etc"
    
)

topic = "machine learning"


from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    short=short_prompt | model | parser,
    detailed=detailed_prompt | model | parser,
)

result = chain.invoke({"topic":"Machine Learning"})

print(result['short'])
print(result['detailed'])