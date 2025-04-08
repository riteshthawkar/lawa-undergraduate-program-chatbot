"""
API views for database operations.
"""
from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any
import datetime

from ..config import logger
from .models import ChatHistoryEntry, FeedbackUpdate
from .repository import ChatRepository
from .database import get_db_connection, TABLE_PREFIX

# Create a router for database operations
router = APIRouter(tags=["history"])

# IMPORTANT: Routes with fixed paths must come BEFORE routes with path parameters
# to prevent FastAPI from interpreting 'stats' or 'count' as a chat_id

# Define fixed path routes first, before any parameterized routes

@router.get("/history/export", status_code=status.HTTP_200_OK)
async def export_all_chats():
    """
    Export all chat history data in JSON format.
    
    Returns:
        dict: All chat history data
    """
    try:
        # Get all chats without pagination
        all_chats = ChatRepository.get_all_chats(limit=10000, offset=0)
        
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
async def get_history_stats():
    """
    Get statistics about chat history.
    
    Returns:
        dict: Statistics about chat history including total_chats, liked, disliked, and no_feedback counts
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get total chats
            cursor.execute("SELECT COUNT(*) FROM chat_history")
            total = cursor.fetchone()['count']
            
            # Get liked chats
            cursor.execute("SELECT COUNT(*) FROM chat_history WHERE feedback = 'like'")
            liked = cursor.fetchone()['count']
            
            # Get disliked chats
            cursor.execute("SELECT COUNT(*) FROM chat_history WHERE feedback = 'dislike'")
            disliked = cursor.fetchone()['count']
            
            # Calculate no feedback chats
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
async def get_history_count():
    """
    Get the total count of chat history entries.
    
    Returns:
        dict: Dictionary with the count
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chat_history")
            count = cursor.fetchone()['count']
            return {"count": count}
    except Exception as e:
        logger.error(f"Error getting history count: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get history count"
        )

@router.get("/history", response_model=List[ChatHistoryEntry])
async def get_all_chats_paginated(limit: int = 100, offset: int = 0) -> List[ChatHistoryEntry]:
    """
    Get all chat entries with pagination.
    
    Args:
        limit: Maximum number of entries to return
        offset: Number of entries to skip
        
    Returns:
        List[ChatHistoryEntry]: List of chat entries
    """
    return ChatRepository.get_all_chats(limit, offset)

# Now define the parameterized routes

@router.post("/history", status_code=status.HTTP_201_CREATED, response_model=Dict[str, int])
async def save_chat_history(query: str, response: str, sources: List[Dict[str, str]]) -> Dict[str, int]:
    """
    Save a chat interaction to the database.
    
    Args:
        query: The user's query
        response: The LLM's response
        sources: List of citation sources
        
    Returns:
        Dict[str, int]: Dictionary with the ID of the saved chat
    """
    try:
        chat_id = ChatRepository.save_chat(query, response, sources)
        return {"id": chat_id}
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save chat history"
        )

@router.put("/history/{chat_id}/feedback", status_code=status.HTTP_200_OK)
@router.post("/history/{chat_id}/feedback", status_code=status.HTTP_200_OK)
async def update_chat_feedback(chat_id: str, feedback_update: FeedbackUpdate) -> Dict[str, str]:
    """
    Update the feedback for a chat entry.
    
    Args:
        chat_id: The ID of the chat entry from the path
        feedback_update: The feedback update data from the request body
        
    Returns:
        Dict[str, str]: Success message
    """
    success = ChatRepository.update_feedback(
        chat_id=chat_id, 
        feedback=feedback_update.feedback
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat with ID {chat_id} not found or feedback update failed"
        )
    
    return {"message": f"Feedback updated for chat ID: {chat_id}"}

# This route was moved to the top of the file

@router.get("/history/{chat_id}", response_model=ChatHistoryEntry)
async def get_chat_by_id(chat_id: str) -> ChatHistoryEntry:
    """
    Get a chat entry by ID.
    
    Args:
        chat_id: The ID of the chat entry
        
    Returns:
        ChatHistoryEntry: The chat entry
    """
    chat = ChatRepository.get_chat_by_id(chat_id)
    
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat with ID {chat_id} not found"
        )
    
    return chat

# This route was moved to the top of the file

# This route was moved to the top of the file

# All fixed routes have been moved to the top of the file

@router.delete("/history/{chat_id}", status_code=status.HTTP_200_OK)
async def delete_chat(chat_id: str) -> Dict[str, str]:
    """
    Delete a chat entry and its associated sources.
    
    Args:
        chat_id: The ID of the chat entry to delete
        
    Returns:
        Dict[str, str]: Success message
    """
    success = ChatRepository.delete_chat(chat_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat with ID {chat_id} not found or deletion failed"
        )
    
    return {"message": f"Chat with ID {chat_id} deleted successfully"}
