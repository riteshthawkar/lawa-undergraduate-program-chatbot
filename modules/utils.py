import logging
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any

from modules.config import logger

async def safe_send(websocket: WebSocket, message: dict):
    """Send messages safely over the websocket with proper error handling"""
    try:
        await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info("Client disconnected during send")
        raise
    except Exception as e:
        logger.exception("Error sending message:")
        raise

def format_docs(docs: List[Dict[str, Any]]) -> str:
    """Formats a list of documents for use in a prompt, safely handling curly braces."""
    doc_strings = []
    for i, doc in enumerate(docs):
        # Use .format() to avoid issues with curly braces in the document chunk
        template = "- Document {index} (Source: {source}):\n{chunk}"
        # Handle both Document objects and dictionaries for backward compatibility if needed
        if hasattr(doc, 'metadata'): # It's a Document object
            source = doc.metadata.get('page_source', doc.metadata.get('source', 'N/A'))
            content = doc.page_content
        else: # It's a dictionary
            source = doc.get('metadata', {}).get('page_source', doc.get('metadata', {}).get('source', 'N/A'))
            content = doc.get('page_content', doc.get('chunk', ''))

        doc_strings.append(template.format(
            index=i + 1,
            source=source,
            chunk=content
        ))
    return "\n".join(doc_strings)

def format_query(query: str, language: str, formatted_docs: str) -> str:
    """Formats the user query with context from documents."""
    return f"Answer the user's query in {language} based on the following context:\n\n{formatted_docs}\n\nUser Query: {query}"

async def deduplicate_documents(documents):
    """
    Deduplicates a list of LangChain Document objects based on page_content.

    Args:
        documents (list): A list of LangChain Document objects.

    Returns:
        list: A new list containing unique Document objects.
    """
    logger.info(f"Deduplication: Started with {len(documents)} total documents")
    seen_content = set()
    unique_docs = []
    skipped_count = 0

    for doc in documents:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_docs.append(doc)
        else:
            skipped_count += 1
    
    logger.info(f"Deduplication: Skipped {skipped_count} duplicate documents")
    logger.info(f"Returning {len(unique_docs)} unique documents after deduplication.")
    return unique_docs 