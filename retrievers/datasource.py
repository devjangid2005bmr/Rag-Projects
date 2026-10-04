from langchain_community.retrievers import ArxivRetriever

import arxiv
from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True,
    arxiv_search=arxiv.Search,
    arxiv_exceptions=(arxiv.ArxivError,),
)