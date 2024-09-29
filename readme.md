## Curator

### Software Design

command line use:
- curate "course query blah blah"
- returns top 5 courses

args:
- first positional argument (str): user query (optional)
- i (str): input filename (either csv or txt or excel; single column)
- o (str): output filename
- n (int) number of responses
- k (int) original pool
- l (flag): load cosmo export
[update above]

### How it works
- the cosmo export (from URL: ...) is used to load course titles and course descriptions.
- the courses are filtered for this criteria:
- we create embeddings of the 8,000+ course descriptions for a similarity search (these are stored in a chroma vector database)
- to improve quality of recommendations, a locally-hosted LLM reranking model is used to further score the recommended courses for relevance.
- script returns the top k results (k can be modified)

[diagram]

NOTE: this only uses locally-hosted code so there are no data security concerns.

### How to Use

### Best practices
- this script ultimately matches your query to the courses that are most semantically similar as expressed by the long descriptions. As such, a query with more context will generally yield better results.
- while course titles can work, you will get better results if there is any more context you can provide
