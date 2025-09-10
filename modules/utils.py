import logging
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict

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

def format_docs(docs: List) -> str:
    """Format documents for inclusion in prompt. Works with LangChain Document objects."""
    context = ""
    for index, doc in enumerate(docs):
        # Get source from metadata if available
        # Prefer page_source when present as it is canonical in this codebase
        if hasattr(doc, 'metadata'):
            source = doc.metadata.get('page_source', doc.metadata.get('source', 'N/A'))
        else:
            source = 'N/A'
        
        # Check if this is a GitBook source for special formatting
        is_gitbook = 'mbzuai.gitbook.io' in source
        
        context += f"\n{'=' * 75}\n"
        context += f"**DOCUMENT CITATION INDEX:** {index + 1}\n"
        context += f"**DOCUMENT SOURCE:** {source}\n"
        
        # Add special GitBook priority marker
        if is_gitbook:
            context += f"🔥 **GITBOOK OFFICIAL SOURCE - ABSOLUTE HIGHEST PRIORITY - MUST BE CITED** 🔥\n"
            context += f"📚 **OFFICIAL STUDENT HANDBOOK - PRIMARY SOURCE FOR ALL INFORMATION** 📚\n"
        
        context += "\n"
        
        # Get content from page_content for Document objects or context field for dict-like objects
        content = ""
        if hasattr(doc, 'page_content'):
            content = doc.page_content
        elif hasattr(doc, 'context'):
            content = doc.context
        else:
            # Fallback for dict-like objects
            content = doc.get('context', doc.get('chunk', doc.get('page_content', 'N/A')))
            
        context += f"**CONTENT:**\n{content}\n"
    
    if context: 
        context += f"{'=' * 75}\n"

    return context

def format_query(query: str, language: str, docs: List[dict]) -> str:
    """Format the query with language and document context"""
    formatted_docs = format_docs(docs)
    
    # Check if any GitBook documents are present
    has_gitbook = any('mbzuai.gitbook.io' in str(doc.metadata.get('page_source', '')) for doc in docs)
    
    gitbook_instruction = ""
    if has_gitbook:
        gitbook_instruction = "\n🔥 **CRITICAL: GITBOOK SOURCES DETECTED - YOU MUST CITE THEM** 🔥\n"
    
    return f"**USER QUERY:** {query}\n**LANGUAGE:** {language}{gitbook_instruction}\n**CONTEXT:**\n{formatted_docs}" 