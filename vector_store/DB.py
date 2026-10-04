from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings 
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document


docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

result = vector_store.similarity_search("what is data analysis")

for r in result:
    print(r.page_content)
    print(r.metadata)


retriever = vector_store.as_retriever()

docs = retriever.invoke("what is deep learning??")

for d in docs:
    print(d.page_content)