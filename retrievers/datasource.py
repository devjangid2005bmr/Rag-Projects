import arxiv

client = arxiv.Client()

search = arxiv.Search(
    query="large language model",
    max_results=2,
    sort_by=arxiv.SortCriterion.Relevance,
)

results = client.results(search)

for i, result in enumerate(results):
    print(f"\n===== RESULT {i + 1} =====")
    print("Title:", result.title)
    print("Authors:", ", ".join(a.name for a in result.authors))
    print("Summary:", result.summary)
   