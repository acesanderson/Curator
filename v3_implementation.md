# Goal

Allow for a server mode. Currently it is "file" mode, i.e. needs an excel file and generates a persistent chroma database.
This is a separate branch, so chop it into server mode first and then we can add routing for file / vs. server mode based on user.

# Considerations
- Chroma no longer needs logic in the Curator project at all (for server mode), since Kramer handles instantiation and querying of the database.
- reranking needs a rethink, especially since I want it to be modular

# Roadmap
## Server Mode
- [x] implement Chroma_curate script in Kramer
- [x] Make sure query function works there
- [x] rewrite Curate.py to use the CRUDdy functions from Chroma_curate
- [ ] create an actual Curator server -- this access chroma server, yes, but also runs the reranking bits.
 - [ ] FastAPI -- implement 1+ endpoints for the core Curator functionality
- [ ] add MCP logic to the Curator server. This would still use FastAPI (not FastMCP)
- [ ] connect to Claude desktop as POC
- [ ] implement client logic within Chain/ReACT and Tool class
- [ ] experiment with clients (in ask / twig scripts etc.)
## Evaluation
- [ ] allow for swapping out embedding models as plugins
- [ ] experiment with prompting approaches to embeddings
- [ ] allow for swapping out reranking models as plugins
- [ ] create training / validation set for evaluation


