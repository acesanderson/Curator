# Curator

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**AI-powered course recommendation engine that finds relevant courses using semantic search and advanced reranking.**

Curator transforms natural language queries into precise course recommendations by combining vector similarity search with state-of-the-art reranking models - all running locally for complete data privacy.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Find courses with a simple query
python Curate.py "machine learning for beginners"

# Get more results
python Curate.py "data visualization" -k 10

# Process multiple queries from file
python Curate.py -i queries.txt -o recommendations.csv
```

**First run?** Curator automatically builds its vector database - grab a coffee while it processes your course catalog (5-10 minutes).

## Installation

**Prerequisites:**
- Python 3.7+
- Course catalog file (Excel/CSV format)

**Setup:**
```bash
git clone https://github.com/acesanderson/Curator
cd Curator
pip install -r requirements.txt

# Place your course catalog file in the project directory
# Run your first query to initialize the database
python Curate.py "your first query"
```

## How It Works

Curator uses a sophisticated two-stage pipeline:

1. **Vector Search**: Converts your query and course descriptions into semantic embeddings using ChromaDB
2. **AI Reranking**: Applies advanced reranking models (BGE, MixedBread, Cohere, etc.) to refine results based on true relevance

```python
# Use as a Python module
from Curator import Curate

results = Curate("javascript machine learning", k=5)
# Returns: [('Learning TensorFlow with JavaScript', 3.31), ...]
```

## Core Features

- **Semantic Understanding**: Matches intent, not just keywords
- **Multiple Reranking Models**: BGE, MixedBread, Cohere, FlashRank, and more
- **Smart Caching**: Instant results for repeated queries
- **Batch Processing**: Handle multiple queries efficiently
- **Privacy-First**: Runs entirely locally (except for optional API-based rankers)

## Command Line Options

```bash
python Curate.py "query" [options]

# Key parameters
-k 10              # Number of final recommendations (default: 5)
-n 50              # Initial search pool size (default: 30)
-i input.csv       # Batch process from file
-o results.txt     # Save results to file
-s                 # Check system status
```

## Architecture

```
Query → Vector Search → Reranking → Top Results
        (ChromaDB)     (AI Models)   (Ranked List)
```

**Data Flow:**
- Course catalog → Embeddings → Vector database
- User query → Semantic search → AI reranking → Recommendations

## Server Mode (Beta)

Run Curator as a web service with MCP (Model Context Protocol) support:

```bash
python CurateServer.py
```

Provides both REST API and MCP endpoints for integration with AI assistants and other tools.

## Configuration

**Reranking Models:**
- `bge` (default): BAAI/bge-reranker-large
- `mxbai`: MixedBread AI reranker
- `cohere`: Cohere Rerank API
- `flash`: FlashRank (lightweight)

**Environment Variables:**
```bash
export COHERE_API_KEY="your-key"
export JINA_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

## Examples

```bash
# Specific technical topics
python Curate.py "kubernetes microservices architecture"

# Skill level targeting
python Curate.py "python for data science beginners"

# Business domains
python Curate.py "agile project management certification"

# Programming languages
python Curate.py "advanced javascript frameworks"
```

## Performance

- **Cold start**: 5-10 minutes (database initialization)
- **Warm queries**: 1-3 seconds
- **Cached queries**: <100ms
- **Batch processing**: ~2 seconds per query

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- `chromadb` - Vector database
- `rerankers` - AI reranking models
- `rich` - CLI interface
- `fastapi` - Server mode (optional)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests: `python -m pytest tests/`
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/acesanderson/Curator/issues)
- **Email**: bianderson@linkedin.com
- **Status Check**: `python Curate.py -s`

---

*Built with ❤️ for better course discovery*
