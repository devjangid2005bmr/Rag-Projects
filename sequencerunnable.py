from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser



#prompt template

prompt = ChatPromptTemplate.from_template(
    "explain {topic} in the simple way and only in 2 paragraphs and 4 bullet points"
    
)

#model
model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash"
)

#output parser
parser = StrOutputParser()

# #step by step manual flow 

# #formate the prompt
# formated_prompt=prompt.format_messages(topic="machine learning")

# #call the model manually
# response = model.invoke(formated_prompt)

# #parse the output manually
# content = response.content if isinstance(response.content, str) else str(response.content)
# final_output = parser.parse(content)


chains = prompt | model | parser

result = chains.invoke({"topic": "machine learning"})
print(result)