from functools import lru_cache
from langchain_fireworks import FireworksEmbeddings


@lru_cache
def get_embeddings():
    return FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
