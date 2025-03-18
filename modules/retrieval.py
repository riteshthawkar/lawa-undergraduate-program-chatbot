import asyncio
import time
import httpx
import os
from typing import List, Dict
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from modules.config import logger

# Initialize retrieval components
MAX_RETRIES = 3
PINECONE_INDEX = "mbzuai-site-index"
BM25_FILE = "./MBZUAI_BM25_ENCODER.json"

def initialize_pinecone():
    """Initialize the Pinecone retriever with retries"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    for attempt in range(MAX_RETRIES):
        try:
            index = pc.Index(PINECONE_INDEX)
            bm25 = BM25Encoder().load(BM25_FILE)
            
            embed_model = HuggingFaceEmbeddings(
                model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
                model_kwargs={"trust_remote_code": True}
            )
            
            return (
                PineconeHybridSearchRetriever(
                    embeddings=embed_model,
                    sparse_encoder=bm25,
                    index=index,
                    top_k=40,  # Hardcoded as required
                    alpha=0.6,  # Hardcoded as required
                ),
                pc
            )
        except Exception as e:
            logger.warning(f"Pinecone initialization attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                logger.exception("Failed to initialize Pinecone after multiple attempts.")
                raise
            time.sleep(2 ** attempt)

def rerank_docs(query: str, docs: List[dict], pc_client: Pinecone) -> List[dict]:
    """Reranks documents using Pinecone reranking"""
    try:
        result = pc_client.inference.rerank(
            model="cohere-rerank-3.5",
            query=query,
            documents=docs,
            rank_fields=["chunk"],
            top_n=20,
            return_documents=True
        )
        ranked_docs = [{
            "page_source": ele.document.page_source,
            "chunk": ele.document.chunk,
            "summary": ele.document.summary
        } for ele in result.data]
        return ranked_docs
    except Exception as e:
        logger.exception("Error in rerank_docs:")
        raise

async def tavily_search(question: str) -> List[dict]:
    """Fallback search using Tavily API"""
    try:
        # Get API key from environment
        api_key = os.getenv("TAVILY_API_KEY")
        url = "https://api.tavily.com/search"
        payload = {
            "query": question,
            "search_depth": "advanced",
            "topic": "general",
            "max_results": 5,
            "include_answer": False,
            "include_raw_content": True,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        result_docs = []
        for result in results:
            obj = {
                "page_source": result.get("url", ""),
                "chunk": result.get("raw_content", "")
            }
            result_docs.append(obj)
        return result_docs
    except Exception as e:
        logger.exception("Error in tavily_search:")
        # In production you might return an empty list or a fallback response.
        return [] 