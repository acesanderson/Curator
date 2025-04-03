# Goal

Allow for a server mode. Currently it is "file" mode, i.e. needs an excel file and generates a persistent chroma database.
This is a separate branch, so chop it into server mode first and then we can add routing for file / vs. server mode based on user.

# Roadmap
## Server Mode
- [x] implement Chroma_curate script in Kramer
- [ ] Make sure query function works there
- [ ] rewrite Curate.py to use the CRUDdy functions from Chroma_curate
## MCP Server
- [ ] create an MCP server that accesses Chroma server
- [ ] connect to Claude desktop as POC
- [ ] experiment with clients (in ask / twig scripts etc.)
## Evaluation
- [ ] allow for swapping out embedding models as plugins
- [ ] experiment with prompting approaches to embeddings
- [ ] allow for swapping out reranking models as plugins
- [ ] create training / validation set for evaluation


