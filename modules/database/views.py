"""
API views for database operations.
"""
from fastapi import APIRouter, HTTPException, status, Request
from typing import List, Dict, Any, Optional
import datetime

from ..config import logger, RAG_APP_NAME
from .models import ChatHistoryEntry, FeedbackUpdate
from .repository import ChatRepository

# Create a router for database operations
router = APIRouter(tags=["history"])

# IMPORTANT: Routes with fixed paths must come BEFORE routes with path parameters
# to prevent FastAPI from interpreting 'stats' or 'count' as a chat_id

# Define fixed path routes first, before any parameterized routes

@router.get("/history/export", status_code=status.HTTP_200_OK)
async def export_all_chats(request: Request):
    """
    Export all chat history data in JSON format.
    
    Returns:
        dict: All chat history data
    """
    try:
        pool = request.app.state.pool
        if not pool:
            logger.error("Database pool not available in request state for /history/export")
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Get all chats without pagination using await
        all_chats = await ChatRepository.get_all_chats(pool, limit=10000, offset=0)
        
        # Convert to a format suitable for JSON export
        export_data = {
            "exported_at": datetime.datetime.now().isoformat(),
            "total_chats": len(all_chats),
            "chats": [chat.model_dump() for chat in all_chats]
        }
        
        return export_data
    except Exception as e:
        logger.error(f"Error exporting chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export chat history"
        )

@router.get("/history/stats", status_code=status.HTTP_200_OK)
async def get_history_stats(request: Request):
    """
    Get statistics about chat history.
    
    Returns:
        dict: Statistics about chat history including total_chats, liked, disliked, and no_feedback counts
    """
    try:
        pool = request.app.state.pool
        if not pool:
            logger.error("Database pool not available in request state for /history/stats")
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        async with pool.acquire() as conn:
            # Get total chats
            total = await conn.fetchval("SELECT COUNT(*) FROM chat_history")
            
            # Get liked chats
            liked = await conn.fetchval("SELECT COUNT(*) FROM chat_history WHERE feedback = 'like'")
            
            # Get disliked chats
            disliked = await conn.fetchval("SELECT COUNT(*) FROM chat_history WHERE feedback = 'dislike'")
            
            # Calculate no feedback chats (ensure counts are not None)
            total = total or 0
            liked = liked or 0
            disliked = disliked or 0
            no_feedback = total - liked - disliked
            
            return {
                "total_chats": total,
                "liked": liked,
                "disliked": disliked,
                "no_feedback": no_feedback
            }
    except Exception as e:
        logger.error(f"Error getting history stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get history stats"
        )

@router.get("/history/count", status_code=status.HTTP_200_OK)
async def get_history_count(request: Request):
    """
    Get the total count of chat history entries.
    
    Returns:
        dict: Dictionary with the count
    """
    try:
        pool = request.app.state.pool
        if not pool:
            logger.error("Database pool not available in request state for /history/count")
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM chat_history")
            return {"count": count or 0} # Return 0 if count is None
    except Exception as e:
        logger.error(f"Error getting history count: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get history count"
        )

@router.get("/history", response_model=List[ChatHistoryEntry])
async def get_all_chats_paginated(request: Request, limit: int = 100, offset: int = 0) -> List[ChatHistoryEntry]:
    """
    Get all chat entries with pagination.
    
    Args:
        limit: Maximum number of entries to return
        offset: Number of entries to skip
        
    Returns:
        List[ChatHistoryEntry]: List of chat entries
    """
    try:
        pool = request.app.state.pool
        if not pool:
            logger.error("Database pool not available in request state for /history")
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Use await for the async repository method
        return await ChatRepository.get_all_chats(pool, limit, offset)
    except Exception as e:
        logger.error(f"Error getting all chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get all chats"
        )

# Now define the parameterized routes

@router.post("/history", status_code=status.HTTP_201_CREATED, response_model=Dict[str, Any]) # Allow string ID
async def save_chat_history(request: Request, query: str, response: str, sources: List[Dict[str, str]], custom_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Save a chat interaction to the database.
    
    Args:
        query: The user's query
        response: The LLM's response
        sources: List of citation sources
        custom_id: Optional custom ID for the chat
        
    Returns:
        Dict[str, Any]: Dictionary with the ID (numeric or string) of the saved chat
    """
    try:
        pool = request.app.state.pool
        if not pool:
            logger.error("Database pool not available in request state for /history")
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Use await for the async repository method; include default app name
        chat_id = await ChatRepository.save_chat(pool, query, response, sources, custom_id, RAG_APP_NAME)
        if chat_id is None and custom_id is None:
             raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save chat history (no ID returned)"
            )
        # Return the custom ID if provided, otherwise the generated numeric ID
        return {"id": custom_id if custom_id else chat_id}
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save chat history"
        )

@router.put("/history/{chat_id}/feedback", status_code=status.HTTP_200_OK)
@router.post("/history/{chat_id}/feedback", status_code=status.HTTP_200_OK)
async def update_chat_feedback(request: Request, chat_id: str, feedback_update: FeedbackUpdate) -> Dict[str, str]:
    """
    Update the feedback for a chat entry.
    
    Args:
        chat_id: The ID of the chat entry from the path
        feedback_update: The feedback update data from the request body
        
    Returns:
        Dict[str, str]: Success message
    """
    try:
        pool = request.app.state.pool
        if not pool:
            logger.error("Database pool not available in request state for /history/{chat_id}/feedback")
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Use await for the async repository method
        success = await ChatRepository.update_feedback(pool, chat_id=chat_id, feedback=feedback_update.feedback)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat with ID {chat_id} not found or feedback update failed"
            )
        
        return {"message": f"Feedback updated for chat ID: {chat_id}"}
    except Exception as e:
        logger.error(f"Error updating chat feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update chat feedback"
        )

@router.get("/history/{chat_id}", response_model=ChatHistoryEntry)
async def get_chat_by_id(request: Request, chat_id: str) -> ChatHistoryEntry:
    """
    Get a chat entry by ID.
    
    Args:
        chat_id: The ID of the chat entry
        
    Returns:
        ChatHistoryEntry: The chat entry
    """
    try:
        pool = request.app.state.pool
        if not pool:
            logger.error("Database pool not available in request state for /history/{chat_id}")
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Use await for the async repository method
        chat = await ChatRepository.get_chat_by_id(pool, chat_id)
        
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat with ID {chat_id} not found"
            )
        
        return chat
    except Exception as e:
        logger.error(f"Error getting chat by ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get chat by ID"
        )

@router.delete("/history/{chat_id}", status_code=status.HTTP_200_OK)
async def delete_chat(request: Request, chat_id: str) -> Dict[str, str]:
    """
    Delete a chat entry and its associated sources.
    
    Args:
        chat_id: The ID of the chat entry to delete
        
    Returns:
        Dict[str, str]: Success message
    """
    try:
        pool = request.app.state.pool
        if not pool:
            logger.error("Database pool not available in request state for /history/{chat_id}")
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Use await for the async repository method
        success = await ChatRepository.delete_chat(pool, chat_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat with ID {chat_id} not found or deletion failed"
            )
        
        return {"message": f"Chat with ID {chat_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat"
        )
