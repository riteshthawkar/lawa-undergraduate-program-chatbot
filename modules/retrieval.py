import asyncio
import time
import httpx
import os
import json
from typing import List, Dict, Any
from pinecone import Pinecone, PineconeAsyncio
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from modules.config import (
    logger, RETRIEVER_TOP_K, RERANKER_TOP_N,
    PINECONE_API_KEY, PINECONE_SUMMARY_INDEX_NAME, PINECONE_TEXT_INDEX_NAME
)
from pydantic import BaseModel
import openai
from datetime import datetime

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables.")
client = openai.AsyncOpenAI(api_key=api_key)

class Document_reference(BaseModel):
    index: int
    source: str

class Ranked_Documents(BaseModel):
    ranked_documents: List[Document_reference]

# Configuration for retrieval components
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
BM25_FILE = "./MBZUAI_BM25_ENCODER.json" # Ensure this path is correct
HYBRID_ALPHA = 0.5 # Alpha for hybrid search (0.0 for sparse, 1.0 for dense)

def get_reranker_system_prompt():
    current_date_str = datetime.now().strftime("%B %d, %Y")
    return f"""
    As an AI assistant developed by MBZUAI (Mohamed Bin Zayed University of Artificial Intelligence), your task is to re-rank a list of retrieved documents based on their **strict relevance and reliability** to a user's query related to MBZUAI undergraduate program. The objective is to help the response-generation LLM select only the **most accurate, timely, detailed, and MBZUAI undergraduate program-specific** documents.

        **Current Date:** {current_date_str}

    ---

    ### 📌 MBZUAI-Specific Re-ranking Rules

    1. **MBZUAI Undergraduate program Scope Limitation**:
    - ONLY retain and prioritize documents that are **strictly and directly related to MBZUAI undergraduate program** (its policies, faculty, student services, etc.).
    - Exclude or deprioritize any document that includes irrelevant or out-of-scope content (e.g., non-MBZUAI universities, general AI discussions).

    2. **Temporal Awareness & Recency**:
    - Give strong preference to documents with **recent dates** (especially for deadlines, admissions, events, and news).
    - Assume webpages without dates are regularly updated and typically more current than static PDFs.
    - If two documents are equally relevant, favor the one with more **recent or time-sensitive** information.

    3. **Webpage vs. PDF Priority**:
    - Prefer **Webpages** over **PDFs**, unless the PDF contains **significantly more relevant or complete information**.
    - PDFs with outdated content should only be included if no equally relevant webpage exists.

    4. **Depth and Specificity**:
    - Rank higher the documents that:
        - **Directly and deeply address** the user query.
        - Provide **detailed, specific** answers (e.g., program names, deadlines, eligibility criteria).
        - Are **not surface-level mentions** but offer substantive information.
    - Avoid redundancy by excluding less complete duplicates.

    5. **Official and Authoritative Sources**:
    - Consider whether the document appears to be an **official MBZUAI undergraduate program page**, report, or publication.
    - Prioritize such official sources over less authoritative ones (e.g., newsletters or promotional blurbs).

    6. **Avoiding Weak or Ambiguous Mentions**:
    - Do not include documents that only vaguely mention the topic or contain keyword matches without substance.
    - Exclude any document that does not contain elaborative, structured, and relevant information about MBZUAI.

    7. **Comparative Relevance Judgement**:
    - Evaluate all documents in comparison with one another.
    - Prefer the most comprehensive, authoritative, and structured source for each relevant piece of information.

    8. **Embedded Metadata Extraction**:
    - If a document includes dates, titles, or content indicators (e.g., "Annual Report 2023"), use that information to infer recency and document type, even if not explicitly provided as metadata.

    9. **Result Set Size (Soft Cap)**:
    - Limit the ranked list to a **maximum of 5–7 highly relevant documents**.
    - Exclude marginally useful documents to ensure the downstream model receives only the best context.

    ---

    ### 📌 Input Structure

    You will be provided with:
    - A **user query** (always related to MBZUAI).
    - A list of documents. Each document contains:
        - `ORIGINAL_INDEX`: Numeric index in the original list.
        - `DOCUMENT_SOURCE`: URL or source path.
        - `DOCUMENT_TYPE`: Webpage or PDF.
        - `DOCUMENT_CONTENT`: Text content snippet (may include date or citation markers like [1]).


    Example input document format (you will receive a list of these):
    ========
    **ORIGINAL_INDEX:** 0
    **DOCUMENT_SOURCE:** https://mbzuai.ac.ae/research/latest-initiatives
    **DOCUMENT_TYPE:** Webpage
    **DOCUMENT_CONTENT:**
    MBZUAI is proud to announce its 2024 research focus on 'AI for Sustainable Development Goals', including projects on renewable energy optimization and AI-driven healthcare diagnostics. [1] Our new 'AI in Climate Change' lab launched in January 2024. [1]
    ========
    **ORIGINAL_INDEX:** 1
    **DOCUMENT_SOURCE:** https://mbzuai.ac.ae/assets/pdfs/annual-report-2023.pdf
    **DOCUMENT_TYPE:** PDF
    **DOCUMENT_CONTENT:**
    MBZUAI's 2023 annual report highlights key achievements... [1]
    =====

    ---

    ### 📌 Output Format

    Return a **JSON object** containing a key `"ranked_documents"` with a **list of top-ranked documents**.

    Each item in the list should be a dictionary with the following fields:
    - `"index"`: Original index of the document.
    - `"source"`: URL or source identifier.

    Only include documents that:
    - **Clearly and directly help answer the query**.
    - Follow all the MBZUAI undergraduate program-specific prioritization rules above.
    - Are **reliable and suitable as primary reference material**.

    If none of the documents meet the criteria, return an empty list.

    ---

    ### 📌 Output Example:

    ```json
    {
    "ranked_documents": [
        {
        "index": 0,
        "source": "https://mbzuai.ac.ae/research/latest-initiatives"
        },
        {
        "index": 2,
        "source": "https://mbzuai.ac.ae/study/admission-process"
        }
    ]
    }
    ````

    Do **not** include document content in the output.
    Do **not** summarize or answer the query.
    Do **not** hallucinate document utility — judge based only on what's visible in the input.

    Focus purely on **ranking based on MBZUAI relevance, recency, specificity, and authority**.
    """



class Document_reference(BaseModel):
    index: int
    source: str

class Ranked_Documents(BaseModel):
    ranked_documents: List[Document_reference]

def format_docs(docs: List[dict]) -> str:
    """Format documents for inclusion in prompt"""
    context = ""
    for index, ele in enumerate(docs):
        doc_type = "PDF" if ele.get("page_source") and ".pdf" in ele["page_source"].lower() else "Webpage"
        
        context += f"**ORIGINAL_INDEX:** {index}\n"
        context += f"**DOCUMENT_SOURCE:** {ele.get('page_source', 'N/A')}\n"
        context += f"**DOCUMENT_TYPE:** {doc_type}\n\n"
        
        # Removed blanket PDF warning
            
        context += f"**DOCUMENT_CONTENT:**\n{ele.get('chunk', 'N/A')}\n"
        context += f"\n{'=' * 75}\n"
    if context: 
        context += f"{'=' * 75}\n"
    return context

async def openai_rerank_and_filter_docs(query: str, original_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reranks and filters documents using OpenAI's gpt-4o-mini model."""
    documents_for_llm = []
    for i, doc in enumerate(original_docs):
        # Handle both nested metadata and direct metadata formats
        if 'metadata' in doc and isinstance(doc['metadata'], dict):
            # Format from app.py's docs_for_reranking
            # Use page_source for URL as indicated by user
            source = doc['metadata'].get('page_source', doc['metadata'].get('source', 'N/A'))
            content = doc.get('page_content', '')
        else:
            # Direct format
            source = doc.get('page_source', doc.get('source', 'N/A'))
            content = doc.get('page_content', doc.get('chunk', ''))
            
        documents_for_llm.append({
            "source": source,
            "chunk": content
        })

    if not documents_for_llm:
        return []

    try:
        formatted_docs = format_docs(documents_for_llm)
        llm_payload = f"User Query: {query}\n\n{formatted_docs}"

        response = await client.beta.chat.completions.parse(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {"role": "system", "content": get_reranker_system_prompt()},
                {"role": "user", "content": llm_payload}
            ],
            response_format=Ranked_Documents,
            temperature=0.1
        )

        llm_output_str = response.choices[0].message.content
        if not llm_output_str:
            logger.warning("OpenAI reranker returned empty content.")
            return original_docs

        logger.debug("Raw LLM output for reranking: %s", llm_output_str)
        
        # Safer JSON parsing
        try:
            llm_response_dict = json.loads(llm_output_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse reranker output as JSON: %s", e)
            logger.error("Problematic output: %r", llm_output_str)
            return original_docs[:RERANKER_TOP_N]

        if not isinstance(llm_response_dict, dict):
            logger.error("OpenAI reranker did not return a JSON object. Got: %s", type(llm_response_dict))
            return original_docs[:RERANKER_TOP_N]

        ranked_items = llm_response_dict.get("ranked_documents")
        if not isinstance(ranked_items, list):
            logger.error("Reranker did not return a list under 'ranked_documents'. Found: %s", type(ranked_items))
            return original_docs[:RERANKER_TOP_N]

        final_ranked_docs = []
        seen_indices = set()
        for item in ranked_items:
            if not isinstance(item, dict) or "index" not in item or "source" not in item:
                logger.warning("Invalid item format from reranker: %r", item)
                continue
            
            original_index = item.get("index")
            if isinstance(original_index, int) and 0 <= original_index < len(original_docs) and original_index not in seen_indices:
                final_ranked_docs.append(original_docs[original_index])
                seen_indices.add(original_index)
            else:
                logger.warning("Reranker referred to an invalid or duplicate index: %r", original_index)
        
        logger.info("Reranked %d docs down to %d.", len(original_docs), len(final_ranked_docs))
        return final_ranked_docs

    except Exception as e:
        logger.error("An error occurred during OpenAI reranking: %s", str(e))
        # Fallback: prioritize web pages over PDFs
        web_pages = []
        pdfs = []
        for doc in original_docs:
            source = doc.get("metadata", {}).get("source", "").lower()
            if ".pdf" in source:
                pdfs.append(doc)
            else:
                web_pages.append(doc)
        
        fallback_docs = (web_pages + pdfs)[:RERANKER_TOP_N]
        logger.info("Reranker failed. Returning prioritized fallback documents.")
        return fallback_docs

async def rerank_docs(query: str, docs: List[Dict[str, Any]], is_time_sensitive: bool = False) -> List[Dict[str, Any]]:
    """
    Reranks documents using an OpenAI model.
    """
    if not docs:
        return []

    # The is_time_sensitive flag is handled by the reranker's system prompt, not as an argument.
    reranked_docs = await openai_rerank_and_filter_docs(query, docs)

    return reranked_docs

async def fetch_balanced_documents(
    metadata_query: str,
    natural_language_query: str,
    pinecone_summary_index: Any, # pinecone.Index
    pinecone_text_index: Any, # pinecone.Index
    embed_model: HuggingFaceEmbeddings,
    bm25_encoder: BM25Encoder,
    num_summary_docs: int = RETRIEVER_TOP_K,
    num_text_docs: int = RETRIEVER_TOP_K
) -> List[Document]:
    """
    Retrieves documents from both summary and text indexes in Pinecone using hybrid search,
    combining dense vectors and sparse BM25 vectors for improved retrieval.
    Uses specifically tailored queries for each index.
    """
    async def query_hybrid_retriever(index, query, k):
        retriever = PineconeHybridSearchRetriever(
            index=index,
            embeddings=embed_model,
            sparse_encoder=bm25_encoder,
            top_k=k,
            alpha=HYBRID_ALPHA
        )
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, 
                retriever.get_relevant_documents, 
                query
            )
        except Exception as e:
            logger.error(f"Error during hybrid search on index {index.name} for query '{query}': {e}")
            return []

    try:
        logger.info(f"Querying summary index with: '{metadata_query}'")
        logger.info(f"Querying text index with: '{natural_language_query}'")

        summary_task = query_hybrid_retriever(pinecone_summary_index, metadata_query, num_summary_docs)
        text_task = query_hybrid_retriever(pinecone_text_index, natural_language_query, num_text_docs)

        results = await asyncio.gather(summary_task, text_task, return_exceptions=True)

        all_docs = []
        if isinstance(results[0], list):
            logger.info(f"Retrieved {len(results[0])} documents from summary index using hybrid search.")
            all_docs.extend(results[0])
        else:
            logger.error(f"Error querying summary index with hybrid search: {results[0]}")

        if isinstance(results[1], list):
            logger.info(f"Retrieved {len(results[1])} documents from text index using hybrid search.")
            all_docs.extend(results[1])
        else:
            logger.error(f"Error querying text index with hybrid search: {results[1]}")

        unique_docs = {}
        skipped_count = 0
        for doc in all_docs:
            # Use page_id as the primary deduplication key
            page_id = doc.metadata.get('page_id')

            
            if not page_id:
                # Fallback to page_source + chunk_id if page_id is not available
                page_id = f"{doc.metadata.get('page_source', '')}_{doc.metadata.get('chunk_id', '')}"
            
            if page_id not in unique_docs:
                unique_docs[page_id] = doc
            else:
                skipped_count += 1
        
        final_docs = list(unique_docs.values())
        logger.info(f"Deduplication: Started with {len(all_docs)} total documents")
        logger.info(f"Deduplication: Skipped {skipped_count} duplicate documents")
        logger.info(f"Returning {len(final_docs)} unique documents after deduplication.")
        return final_docs

    except Exception as e:
        logger.exception("An error occurred during balanced document fetching.")
        # Propagate the exception to be handled by the caller
        raise

def format_docs_for_llm_prompt(docs: List[Dict[str, Any]]) -> str:
    """
    Formats a list of documents into a string suitable for an LLM prompt.
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        page_content = doc.page_content if isinstance(doc, Document) else doc.get('page_content', '')
        metadata = doc.metadata if isinstance(doc, Document) else doc.get('metadata', {})

        title = metadata.get("document_title", "No Title Available")
        summary = metadata.get("document_summary", "No Summary Available")
        keywords = metadata.get("keywords", "[]")
        source = metadata.get("page_source", metadata.get("source", "N/A"))
        
        if isinstance(keywords, str):
            try:
                keywords_list = json.loads(keywords.replace("'", '"'))
                keywords_str = ", ".join(keywords_list)
            except json.JSONDecodeError:
                keywords_str = keywords
        elif isinstance(keywords, list):
            keywords_str = ", ".join(keywords)
        else:
            keywords_str = "None"

        doc_str = (
            f"========\n"
            f"**ORIGINAL_INDEX:** {i}\n"
            f"**DOCUMENT_SOURCE:** {source}\n"
            f"**DOCUMENT_TITLE:** {title}\n"
            f"**DOCUMENT_SUMMARY:** {summary}\n"
            f"**KEYWORDS:** {keywords_str}\n"
            f"**DOCUMENT_CONTENT:**\n{page_content}\n"
            f"========"
        )
        formatted_docs.append(doc_str)
    
    return "\n".join(formatted_docs) 

def initialize_pinecone_clients():
    """Initializes and returns the Pinecone index clients."""
    logger.info("Initializing Pinecone clients...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        summary_index = pc.Index(PINECONE_SUMMARY_INDEX_NAME)
        text_index = pc.Index(PINECONE_TEXT_INDEX_NAME)
        logger.info("Pinecone clients initialized successfully.")
        return summary_index, text_index
    except Exception as e:
        logger.exception("Failed to initialize Pinecone clients.")
        raise e

def initialize_retrieval_components():
    """Initializes and returns the embedding model and BM25 encoder."""
    logger.info("Initializing retrieval components...")
    try:
        embed_model = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"trust_remote_code": True}
        )
        bm25_encoder = BM25Encoder().load(BM25_FILE)
        logger.info("Retrieval components initialized successfully.")
        return embed_model, bm25_encoder
    except Exception as e:
        logger.exception("Failed to initialize retrieval components.")
        raise