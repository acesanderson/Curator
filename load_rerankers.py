"""
NOTE: you need to specify MPS since torch will otherwise default to CUDA.
"""


from rerankers import Reranker

# Cross-encoder default. You can specify a 'lang' parameter to load a multilingual version!
ranker = Reranker('cross-encoder')

# Specific cross-encoder
ranker = Reranker('mixedbread-ai/mxbai-rerank-large-v1', model_type='cross-encoder')

# FlashRank default. You can specify a 'lang' parameter to load a multilingual version!
ranker = Reranker('flashrank')

# Specific flashrank model.
ranker = Reranker('ce-esci-MiniLM-L12-v2', model_type='flashrank')

# Default T5 Seq2Seq reranker
ranker = Reranker("t5")

# Specific T5 Seq2Seq reranker
ranker = Reranker("unicamp-dl/InRanker-base", model_type = "t5")

# API (Cohere)
# ranker = Reranker("cohere", lang='en' (or 'other'), api_key = API_KEY)

# # Custom Cohere model? No problem!
# ranker = Reranker("my_model_name", api_provider = "cohere", api_key = API_KEY)

# # API (Jina)
# ranker = Reranker("jina", api_key = API_KEY)

# # RankGPT4-turbo
# ranker = Reranker("rankgpt", api_key = API_KEY)

# # RankGPT3-turbo
# ranker = Reranker("rankgpt3", api_key = API_KEY)

# # RankGPT with another LLM provider
# ranker = Reranker("MY_LLM_NAME" (check litellm docs), model_type = "rankgpt", api_key = API_KEY)

# # RankLLM with default GPT (GPT-4o)
# ranker = Reranker("rankllm", api_key = API_KEY)

# # RankLLM with specified GPT models
# ranker = Reranker('gpt-4-turbo', model_type="rankllm", api_key = API_KEY)

# # ColBERTv2 reranker
# ranker = Reranker("colbert")

# # LLM Layerwise Reranker
# ranker = Reranker('llm-layerwise')



docs = [
	"I've glimpsed the future of coding.",
	"Programming's future is clear to me.",
	"The next era of software development is upon us.",
	"I have witnessed the evolution of programming.",
	"Future trends in coding are in my view.",
	"The destiny of software craftsmanship has unfolded.",
	"Programming as we know it is transforming.",
	"I've seen what's next in computer science.",
	"There's a new horizon for developers.",
	"Programming paradigms are shifting forward.",
	"I perceive the journey of technological innovation.",
	"The evolution of code is visible now.",
	"Future programming is nothing like the past.",
	"I've encountered a vision of digital craftsmanship.",
	"Tomorrow's software solutions are here.",
	"A new chapter of programming begins.",
	"I'm aware of an upcoming coding revolution.",
	"The progression of digital technology is astonishing.",
	"Creativity and algorithms will intertwine like never before.",
	"The future holds unknown paths beyond programming."
]

# T5-based pointwise rankers (InRanker, MonoT5)
# LLM-based pointwise rankers (BAAI/bge-reranker-v2.5-gemma2-lightweight)
# Jina, Voyage, and MixedBread API rerankers
# FlashRank (these are CPU-optimized)
# Colbert-based reranker (not designed for this but works)

ranker = Reranker('unicamp-dl/InRanker-base', device='cuda', batch_size=32, verbose=0) # T5Ranker

"""
APIRanker
AVAILABLE_RANKERS
ColBERTRanker
FlashRankRanker
LLMLayerWiseRanker
RankGPTRanker
T5Ranker
TransformerRanker
api_rankers
colbert_ranker
flashrank_ranker
llm_layerwise_ranker
ranker
rankgpt_rankers
t5ranker
transformer_ranker"""


# WHat I HAVE
"""
TransformerRanker
APIRanker
RankGPTRanker
T5Ranker
ColBERTRanker
FlashRankRanker
LLMLayerWiseRanker
""".strip().split('\n')


reranker = Reranker(
    model_name='BAAI/bge-reranker-base',
    model_type='cross-encoder',
    device='mps'
)

ranker = Reranker(
  model_name='mixedbread-ai/mxbai-rerank-large-v1',
  model_type='cross-encoder',
  device='mps'
)