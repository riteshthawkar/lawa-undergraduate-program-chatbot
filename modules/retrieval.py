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

from modules.config import logger
from pydantic import BaseModel
import openai
from datetime import datetime

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables.")
client = openai.AsyncOpenAI(api_key=api_key)

class Document_reference(BaseModel):
    index: int
    source_url: str

class Ranked_Documents(BaseModel):
    ranked_documents: List[Document_reference]

# Initialize retrieval components
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
PINECONE_INDEX = "final-mbzuai-index-2" # Ensure this index name is correct
PINECONE_WEBPAGES_INDEX = "mbzuai-webpages-index-latest" # Define missing constant
PINECONE_PDFS_INDEX = "mbzuai-pdfs-index-latest"       # Define missing constant
BM25_FILE = "./MBZUAI_BM25_ENCODER.json" # Ensure this path is correct

# Constants for pre-filtering before sending to LLM reranker
NUM_WEBPAGES_TO_SELECT = 5
NUM_PDFS_TO_SELECT = 3

# New constants for balanced fetching
TARGET_INITIAL_WEB_RESULTS = 20  # Number of webpages to fetch initially
TARGET_INITIAL_PDF_RESULTS = 20  # Number of PDFs to fetch initially
METADATA_FIELD_FOR_TYPE = "document_type" # Metadata field in Pinecone for document type
METADATA_VALUE_WEBPAGE = "WEBPAGE"    # Value for webpage documents
METADATA_VALUE_PDF = "PDF"            # Value for PDF documents

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
    - `"source_url"`: URL or source identifier.

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
        "source_url": "https://mbzuai.ac.ae/research/latest-initiatives"
        },
        {
        "index": 2,
        "source_url": "https://mbzuai.ac.ae/study/admission-process"
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
    source_url: str

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
        # Assuming doc structure is like Langchain Document converted to dict
        # {'page_content': '...', 'metadata': {'source': '...', 'title': '...'}}
        documents_for_llm.append({
            "page_source": doc.get("page_source", {}),
            "chunk": doc.get("chunk", "")
        })

    if not documents_for_llm:
        return []

    try:
        formatted_docs = format_docs(documents_for_llm)
        llm_payload = f"User Query: {query}\n\n{formatted_docs}"
        
        # Check payload size if necessary, though gpt-4o-mini has a large context window
        response = await client.beta.chat.completions.parse(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {"role": "system", "content": get_reranker_system_prompt()},
                {"role": "user", "content": llm_payload}
            ],
            response_format=Ranked_Documents, # Request JSON output
            temperature=0.1 # Low temperature for more deterministic ranking
        )

        llm_output_str = response.choices[0].message.content
        if not llm_output_str:
            logger.warning("OpenAI reranker returned empty content.")
            return original_docs # Fallback

        logger.debug(f"Raw LLM output for reranking: {llm_output_str}")

        llm_response_dict = json.loads(llm_output_str)

        if not isinstance(llm_response_dict, dict):
            logger.error(f"OpenAI reranker did not return a JSON object (dict) as expected. Got: {type(llm_response_dict)}")
            return original_docs # Fallback

        ranked_items = llm_response_dict.get("ranked_documents")

        if not isinstance(ranked_items, list):
            logger.error(f"OpenAI reranker did not return a list under 'ranked_documents' key. Found: {type(ranked_items)}. Full response: {llm_response_dict}")
            return original_docs # Fallback

        final_ranked_docs = []
        seen_indices = set()
        for item in ranked_items:
            if not isinstance(item, dict) or "index" not in item or "source_url" not in item:
                logger.warning(f"Invalid item format from OpenAI reranker: {item}")
                continue
            original_index = item.get("index")
            if isinstance(original_index, int) and 0 <= original_index < len(original_docs) and original_index not in seen_indices:
                final_ranked_docs.append(original_docs[original_index])
                seen_indices.add(original_index)
            else:
                logger.warning(f"LLM reranker referred to an invalid or duplicate index: {original_index}")
        
        logger.info(f"OpenAI Reranker selected {len(final_ranked_docs)} out of {len(original_docs)} documents.")
        return final_ranked_docs

    except json.JSONDecodeError:
        logger.error("Failed to decode LLM reranker response as JSON.")
        return original_docs # Fallback
    except openai.APIError as e:
        logger.error(f"OpenAI API error during reranking: {e}")
        return original_docs # Fallback
    except Exception as e:
        logger.exception(f"Unexpected error in openai_rerank_and_filter_docs: {e}")
        return original_docs # Fallback

async def rerank_docs(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]: # docs are Langchain-style dicts from app.py
    """
    Pre-filters documents to a specified number of webpages and PDFs, 
    transforms them, then reranks them using the OpenAI LLM.
    """
    if not docs:
        logger.info("rerank_docs received no documents to process.")
        return []

    # 1. Separate incoming Langchain-style docs into webpage and PDF lists
    webpage_lc_docs = []
    pdf_lc_docs = []
    for lc_doc in docs:
        metadata = lc_doc.get("metadata")
        doc_type = None
        source_url_raw = None

        if metadata:
            doc_type = metadata.get(METADATA_FIELD_FOR_TYPE)
            source_url_raw = metadata.get("page_source")

        if doc_type == METADATA_VALUE_WEBPAGE:
            webpage_lc_docs.append(lc_doc)
        elif doc_type == METADATA_VALUE_PDF:
            pdf_lc_docs.append(lc_doc)
        elif source_url_raw: # Fallback to URL check if source_type is missing/unknown but URL exists
            source_url_lower = source_url_raw.lower()
            if ".pdf" not in source_url_lower:
                webpage_lc_docs.append(lc_doc)
            else:
                pdf_lc_docs.append(lc_doc)
        else:
            logger.warning(
                f"Document could not be classified as webpage/PDF and has no source URL: {lc_doc.get('page_content', '')[:100]}..."
            )
    
    logger.info(f"Initial separation: {len(webpage_lc_docs)} webpages, {len(pdf_lc_docs)} PDFs.")

    # 2. Select top N from each list (still Langchain-style docs)
    selected_lc_docs = []
    selected_lc_docs.extend(webpage_lc_docs[:NUM_WEBPAGES_TO_SELECT])
    selected_lc_docs.extend(pdf_lc_docs[:NUM_PDFS_TO_SELECT])

    if not selected_lc_docs:
        logger.info("No documents selected after web/PDF pre-filtering for LLM reranker.")
        return []
    
    logger.info(f"Pre-filtered to {len(selected_lc_docs)} documents for LLM reranker input.")

    # 3. Transform selected Langchain-style docs to the page_source/chunk format
    #    that openai_rerank_and_filter_docs expects for its 'original_docs' parameter.
    docs_for_llm_reranker_input = []
    for lc_doc in selected_lc_docs:
        page_source = ""
        metadata = lc_doc.get("metadata")
        if metadata:
            page_source = metadata.get("page_source", metadata.get("source", ""))
        
        chunk_content = lc_doc.get("page_content", "") # Use 'page_content' for the chunk

        docs_for_llm_reranker_input.append({
            "page_source": page_source,
            "chunk": chunk_content
        })

    # Log the documents being sent to the reranker
    reranker_input_sources = [doc.get("page_source", "NO_URL_FOUND") for doc in docs_for_llm_reranker_input]
    logger.info(f"Sending {len(reranker_input_sources)} docs to LLM reranker. Sources: {reranker_input_sources}")

    # 4. Call the LLM reranker with the pre-filtered and transformed documents
    try:
        # openai_rerank_and_filter_docs now receives a list of {'page_source': ..., 'chunk': ...} dicts.
        # It will handle creating the final prompt for the LLM, including adding indices.
        llm_reranked_docs = await openai_rerank_and_filter_docs(query, docs_for_llm_reranker_input)

        if not llm_reranked_docs:
            logger.info("No documents returned after OpenAI reranking and filtering step.")
            return []
        
        # The returned llm_reranked_docs are already in the desired format 
        # (list of {'page_source': ..., 'chunk': ...} dicts, ranked by LLM)
        # and filtered by the LLM.
        reranker_output_sources = [doc.get("page_source", "NO_URL_FOUND") for doc in llm_reranked_docs]
        logger.info(f"LLM reranker returned {len(reranker_output_sources)} docs. Sources: {reranker_output_sources}")
        logger.info(f"rerank_docs returning {len(llm_reranked_docs)} documents after LLM processing.")
        return llm_reranked_docs

    except Exception as e:
        logger.exception("Error during the main rerank_docs processing with OpenAI reranker:")
        # Fallback: return the pre-selected (but not LLM-reranked) docs, or empty, or re-raise
        # For now, returning the pre-filtered, transformed docs to avoid full failure.
        # Consider if this is the best fallback (e.g., maybe return 'docs' or '[]').
        return docs_for_llm_reranker_input 

async def fetch_balanced_documents(
    query: str, 
    pc_client: Pinecone, 
    embed_model: HuggingFaceEmbeddings, 
    bm25_encoder: BM25Encoder, 
    num_webpages: int = TARGET_INITIAL_WEB_RESULTS,
    num_pdfs: int = TARGET_INITIAL_PDF_RESULTS
) -> List[Dict[str, Any]]:
    """
    Fetches a balanced set of webpage and PDF documents from Pinecone
    by performing two separate, concurrent filtered queries.

    Args:
        query: The user's search query.
        pc_client: Synchronous Pinecone client instance.
        embed_model: Initialized HuggingFaceEmbeddings model.
        bm25_encoder: Initialized BM25Encoder.
        num_webpages: The target number of webpage documents to retrieve.
        num_pdfs: The target number of PDF documents to retrieve.

    Returns:
        A list of unique documents (dictionaries) combined from both sources,
        with 'page_content' and 'metadata' keys.
    """
    all_docs = []
    try:
        # Get Pinecone Index objects for webpages and PDFs
        web_pinecone_index = pc_client.Index(PINECONE_WEBPAGES_INDEX)
        pdf_pinecone_index = pc_client.Index(PINECONE_PDFS_INDEX)

        # Create PineconeHybridSearchRetriever for webpages
        web_retriever = PineconeHybridSearchRetriever(
            embeddings=embed_model,
            sparse_encoder=bm25_encoder,
            index=web_pinecone_index,
            top_k=num_webpages, # Use num_webpages for top_k
            alpha=0.6 
        )

        # Create PineconeHybridSearchRetriever for PDFs
        pdf_retriever = PineconeHybridSearchRetriever(
            embeddings=embed_model,
            sparse_encoder=bm25_encoder,
            index=pdf_pinecone_index,
            top_k=num_pdfs, # Use num_pdfs for top_k
            alpha=0.6
        )

        # Helper function to fetch documents from a given retriever
        async def _get_relevant_documents_async(current_retriever: PineconeHybridSearchRetriever, current_query: str):
            # No filter needed as indexes are type-specific
            return await current_retriever.ainvoke(current_query) # Corrected method name

        # Perform concurrent retrievals
        tasks = [
            _get_relevant_documents_async(web_retriever, query),
            _get_relevant_documents_async(pdf_retriever, query)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        webpage_docs_result = results[0]
        pdf_docs_result = results[1]

        webpage_docs = []
        if isinstance(webpage_docs_result, Exception):
            logger.error(f"Error fetching webpage documents for query '{query}': {webpage_docs_result}")
        else:
            for doc in webpage_docs_result:
                doc.metadata[METADATA_FIELD_FOR_TYPE] = METADATA_VALUE_WEBPAGE
                webpage_docs.append(doc)

        pdf_docs = []
        if isinstance(pdf_docs_result, Exception):
            logger.error(f"Error fetching PDF documents for query '{query}': {pdf_docs_result}")
        else:
            for doc in pdf_docs_result:
                doc.metadata[METADATA_FIELD_FOR_TYPE] = METADATA_VALUE_PDF
                pdf_docs.append(doc)

        # logger.info(f"Retrieved {len(webpage_docs)} webpages and {len(pdf_docs)} PDFs for query '{query}' before deduplication.")

        # Combine and deduplicate documents
        # Using a dictionary to store documents by a unique identifier (e.g., URL or a hash of content if URL is not always unique)
        # to ensure uniqueness before converting to list of dicts.
        # Langchain documents have a 'metadata' attribute which usually contains 'source'.
        # We also need to handle potential lack of 'source' or if it's not a good unique key.
        # For now, let's assume 'source' in metadata is a URL and can be used for deduplication.
        # If not, a more robust content-based hashing might be needed.

        unique_docs_by_source = {}

        for doc in webpage_docs + pdf_docs:
            source = doc.metadata.get('source', '') 
            # If source is missing or empty, use a hash of page_content for uniqueness.
            # This is a simple way, might need more robust handling.
            unique_key = source if source else str(hash(doc.page_content))
            if unique_key not in unique_docs_by_source:
                unique_docs_by_source[unique_key] = doc

        # Convert Langchain Document objects to dictionaries
        unique_docs_as_dicts = [
            {"page_content": doc.page_content, "metadata": doc.metadata} 
            for doc in unique_docs_by_source.values()
        ]
        
        # logger.info(f"Returning {len(unique_docs_as_dicts)} unique documents for query '{query}'.")
        return unique_docs_as_dicts

    except Exception as e:
        logger.exception(f"Error in fetch_balanced_documents for query '{query}': {e}")
        return [] # Return empty list on error

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

def initialize_pinecone():
    """Initialize the Pinecone retriever with retries"""
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    async_pc = PineconeAsyncio(api_key=api_key)
    
    for attempt in range(MAX_RETRIES):
        try:
            # Index object is no longer created here directly for a single index
            bm25 = BM25Encoder().load(BM25_FILE)
            
            embed_model = HuggingFaceEmbeddings(
                model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
                model_kwargs={"trust_remote_code": True}
            )
            
            # Return components instead of a pre-configured retriever
            return embed_model, bm25, pc, async_pc
            
        except Exception as e:
            logger.warning(f"Pinecone initialization attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                logger.exception("Failed to initialize Pinecone after multiple attempts.")
                raise
            time.sleep(2 ** attempt)

def initialize_pinecone_with_embeddings(embed_model):
    """Initialize the Pinecone retriever with a pre-initialized embedding model"""
    api_key = os.getenv("PINECONE_API_KEY")
    # Initialize both sync and async clients
    pc = Pinecone(api_key=api_key)
    async_pc = PineconeAsyncio(api_key=api_key)
    
    for attempt in range(MAX_RETRIES):
        try:
            # Index object is no longer created here directly for a single index
            bm25 = BM25Encoder().load(BM25_FILE)
            
            # Return components (bm25, pc, async_pc) as embed_model is already provided
            return bm25, pc, async_pc
            
        except Exception as e:
            logger.warning(f"Pinecone initialization attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                logger.exception("Failed to initialize Pinecone after multiple attempts.")
                raise
            time.sleep(2 ** attempt)