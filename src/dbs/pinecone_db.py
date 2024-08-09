import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pinecone_api = os.getenv("PINECONE_API_KEY")
index = os.getenv("PINECONE_INDEX_NAME")
namespace = os.getenv("PINECONE_NAMESPACE")


def get_index():
    pc = Pinecone(api_key=pinecone_api)
    return pc.Index(index)