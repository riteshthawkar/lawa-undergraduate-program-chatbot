import asyncio
import os
import json
from typing import List, Dict, Any
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers.pinecone_hybrid_search import PineconeHybridSearchRetriever
from pydantic import BaseModel
import httpx
from datetime import datetime
from openai import AsyncOpenAI

from modules.config import (
    logger,
    RETRIEVAL_K,
    RERANKER_TOP_N,
    TOTAL_DOCS_TO_RERANK,
    OPENAI_TIMEOUT,
    OPENAI_RERANKER_MODEL,
    OPENAI_RERANKER_REASONING_EFFORT,
    HYBRID_ALPHA,
    EMBEDDING_MODEL_NAME,
    BM25_FILE_PATH,
)

# --- OpenAI Client Initialization ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables.")
client = AsyncOpenAI(api_key=api_key)

# --- Pydantic Models for Type Hinting and Validation ---
class Document_reference(BaseModel):
    index: int
    source: str

class Ranked_Documents(BaseModel):
    ranked_documents: List[Document_reference]

# --- Constants ---
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
BM25_FILE = BM25_FILE_PATH


# --- Reranker System Prompt ---
def get_reranker_system_prompt(is_time_sensitive: bool = False):
    current_date_str = datetime.now().strftime("%B %d, %Y")

    base_prompt = (
        """
    As an AI assistant developed by MBZUAI (Mohamed Bin Zayed University of Artificial Intelligence), your task is to re-rank a list of retrieved documents based on their **strict relevance and reliability** to a user's query related to MBZUAI undergraduate program. The objective is to help the response-generation LLM select only the **most accurate, timely, detailed, and MBZUAI undergraduate program-specific** documents.

        **Current Date:** """ + current_date_str + """

    ---

    ### 📌 MBZUAI-Specific Re-ranking Rules

    1. **Scope and Domain Relevance**:
    - Focus on documents clearly about MBZUAI.
    - Prefer undergraduate-program documents when the query concerns undergraduate topics (admissions, curriculum, student life, scholarships).
    - Do not exclude official MBZUAI pages that directly answer the query (e.g., leadership/office pages) even if not explicitly labeled as undergraduate.
    - Deprioritize graduate-program (MSc/PhD) documents unless the query explicitly concerns graduate programs. When ambiguous, prefer undergraduate or general-official MBZUAI sources.

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
    
    5a. **GitBook Source Priority (CRITICAL)**:
    - GitBook sources (mbzuai.gitbook.io) are OFFICIAL student handbooks and have ABSOLUTE HIGHEST PRIORITY
    - ALWAYS include GitBook sources in the top 3 positions when they exist in the document list
    - Even if GitBook content seems less detailed, prioritize it over ALL other sources
    - GitBook sources should NEVER be excluded from the final ranked list
    - If multiple GitBook sources exist, rank them by relevance but keep them all in top positions

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
    ```

    Do **not** include document content in the output.
    Do **not** summarize or answer the query.
    Do **not** hallucinate document utility — judge based only on what's visible in the input.

    ### 📌 Query Intent Awareness
    - Infer the user's intent from the query (e.g., identity/role, policy, deadline, process, contact, facility, event).
    - For identity/role queries (e.g., "Who is the provost?"), prioritize authoritative leadership/office pages that explicitly name the role holders.
    - For policy/process/deadline queries, prioritize the most recent official undergraduate pages that directly state the required information.

    Focus purely on **ranking based on MBZUAI relevance, recency, specificity, authority, and inferred intent alignment**.
    """
    )

    return base_prompt



# --- Reranking Function ---
async def openai_rerank_and_filter_docs(query: str, original_docs: List[Dict[str, Any]], is_time_sensitive: bool = False) -> List[Dict[str, Any]]:
    """
    Reranks and filters documents using the configured OpenAI reranker model.
    """
    if not original_docs:
        return []

    formatted_docs_str = format_docs_for_llm_prompt(original_docs)
    system_prompt = get_reranker_system_prompt(is_time_sensitive=is_time_sensitive)
    user_prompt = f"User Query: \"{query}\"\n\nDocuments:\n{formatted_docs_str}"

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=OPENAI_RERANKER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                reasoning_effort=OPENAI_RERANKER_REASONING_EFFORT,
                response_format={"type": "json_object"},
                timeout=OPENAI_TIMEOUT,
            )
            content = response.choices[0].message.content
            parsed_json = json.loads(content)
            
            ranked_data = Ranked_Documents.model_validate(parsed_json)

            # Preserve the order returned by the reranker and guard index bounds
            ordered_docs: List[Dict[str, Any]] = []
            for ref in ranked_data.ranked_documents:
                idx = ref.index
                if isinstance(idx, int) and 0 <= idx < len(original_docs):
                    ordered_docs.append(original_docs[idx])

            return ordered_docs if ordered_docs else original_docs

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                logger.error("Max retries reached. Reranking failed.")
                return original_docs
            await asyncio.sleep(RETRY_DELAY)
    return original_docs

async def rerank_docs(query: str, docs: List[Dict[str, Any]], is_time_sensitive: bool = False) -> List[Dict[str, Any]]:
    """
    Selects, transforms, and reranks documents using a metadata-aware LLM reranker.
    """
    if not docs:
        return []

    docs_to_rerank = docs[:TOTAL_DOCS_TO_RERANK]
    logger.info(f"Sending {len(docs_to_rerank)} documents to be reranked.")

    reranked_docs = await openai_rerank_and_filter_docs(query, docs_to_rerank, is_time_sensitive=is_time_sensitive)
    return reranked_docs

async def fetch_balanced_documents(
    rewritten_queries: Dict[str, str],
    pinecone_summary_index: Any, # pinecone.Index
    pinecone_text_index: Any, # pinecone.Index
    embed_model: HuggingFaceEmbeddings,
    bm25_encoder: BM25Encoder,
    num_summary_docs: int = RETRIEVAL_K,
    num_text_docs: int = RETRIEVAL_K
) -> List[Document]:
    """
    Retrieves documents from both summary and text indexes in Pinecone using hybrid search,
    combining dense vectors and sparse BM25 vectors for improved retrieval.
    Uses specifically tailored queries for each index.
    Deduplicates and returns the combined results.
    """
    try:
        # Extract specialized queries for each index
        metadata_query = rewritten_queries.get("metadata_query", "")
        natural_language_query = rewritten_queries.get("natural_language_query", "")

        async def query_hybrid_retriever(index, query, k):
            try:
                # Create PineconeHybridSearchRetriever with both dense and sparse encoders
                hybrid_retriever = PineconeHybridSearchRetriever(
                    embeddings=embed_model,
                    sparse_encoder=bm25_encoder,
                    index=index,
                    alpha=HYBRID_ALPHA,  # Balance between dense and sparse (0.5 = equal weight)
                    top_k=k  # Number of results to return
                )

                # Use the retriever to perform hybrid search
                logger.info(f"Performing hybrid search with query: {query[:50]}...")
                results = await asyncio.to_thread(
                    hybrid_retriever.invoke,
                    query
                )

                logger.info(f"Hybrid search retrieved {len(results)} documents")
                return results
            except Exception as e:
                logger.exception(f"Error in hybrid search: {e}")
                raise

        # Perform hybrid search on both indexes concurrently
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

def initialize_retrieval_components():
    """Initializes and returns the embedding model and BM25 encoder."""
    logger.info("Initializing retrieval components...")
    try:
        embed_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"trust_remote_code": True}
        )
        bm25_encoder = BM25Encoder().load(BM25_FILE)
        logger.info("Retrieval components initialized successfully.")
        return embed_model, bm25_encoder
    except Exception as e:
        logger.exception("Failed to initialize retrieval components.")
        raise


# Tavily fallback removed: prefer explicit user-facing failure messaging in application layer.
