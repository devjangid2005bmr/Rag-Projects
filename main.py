from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------
# 1. Load environment variables
# --------------------------------

load_dotenv()


# --------------------------------
# 2. Load document
# --------------------------------

data = TextLoader("documents_loader/sample3.txt")

docs = data.load()


# --------------------------------
# 3. Create prompt
# --------------------------------

template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant. Summarize the given text clearly."
    ),
    (
        "human",
        "{data}"
    )
])


# --------------------------------
# 4. Put document text into prompt
# --------------------------------

prompt = template.format_messages(
    data=docs[0].page_content
)


# --------------------------------
# 5. Create Gemini model
# --------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# --------------------------------
# 6. Send prompt to Gemini
# --------------------------------

result = model.invoke(prompt)


# --------------------------------
# 7. Print only the answer
# --------------------------------

print(result.content)