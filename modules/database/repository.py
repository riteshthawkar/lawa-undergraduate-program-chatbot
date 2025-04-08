"""
Repository for database operations.
"""
import psycopg2
from typing import List, Optional, Dict, Any, Tuple

from ..config import logger
from .database import get_db_connection, TABLE_PREFIX
from .models import ChatHistoryEntry, Source


class ChatRepository:
    """Repository for chat history operations"""
    
    @staticmethod
    def save_chat(query: str, response: str, sources: List[Dict[str, str]], custom_id: Optional[str] = None) -> int:
        """
        Save a chat interaction to the database.
        
        Args:
            query: The user's query
            response: The LLM's response
            sources: List of citation sources
            custom_id: Optional custom ID for the chat (passed to the id_str column)
        
        Returns:
            int: The ID of the inserted chat entry
        """
        with get_db_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Insert chat history entry with optional custom ID
                if custom_id:
                    cursor.execute(
                        "INSERT INTO chat_history (query, response, id_str) VALUES (%s, %s, %s) RETURNING id",
                        (query, response, custom_id)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO chat_history (query, response) VALUES (%s, %s) RETURNING id",
                        (query, response)
                    )
                # In PostgreSQL, we get the ID using RETURNING clause
                chat_id = cursor.fetchone()['id']
                
                # Insert sources
                if sources:
                    for source in sources:
                        cursor.execute(
                            "INSERT INTO sources (chat_id, url, cite_num) VALUES (%s, %s, %s)",
                            (chat_id, source["url"], source["cite_num"])
                        )
                
                conn.commit()
                logger.info(f"Chat saved with ID: {chat_id}")
                return chat_id
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error saving chat: {e}")
                raise
    
    @staticmethod
    def update_feedback(chat_id, feedback: str) -> bool:
        """
        Update feedback for a chat entry.
        
        Args:
            chat_id: The ID of the chat entry (can be numeric or string UUID)
            feedback: The feedback to set ('like' or 'dislike')
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        with get_db_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Determine if the ID is a numeric ID or a string ID
                try:
                    # Try to convert to integer for numeric ID lookup
                    numeric_id = int(chat_id)
                    # Check if the chat exists
                    cursor.execute("SELECT id FROM chat_history WHERE id = %s", (numeric_id,))
                    chat_row = cursor.fetchone()
                    if not chat_row:
                        logger.error(f"Chat with numeric ID {numeric_id} not found")
                        return False
                    
                    # Update feedback by numeric ID
                    cursor.execute(
                        "UPDATE chat_history SET feedback = %s WHERE id = %s",
                        (feedback, numeric_id)
                    )
                except (ValueError, TypeError):
                    # If conversion fails, treat it as a string ID
                    string_id = str(chat_id)
                    # Check if the chat exists
                    cursor.execute("SELECT id FROM chat_history WHERE id_str = %s", (string_id,))
                    chat_row = cursor.fetchone()
                    if not chat_row:
                        logger.error(f"Chat with string ID {string_id} not found")
                        return False
                    
                    # Update feedback by string ID
                    cursor.execute(
                        "UPDATE chat_history SET feedback = %s WHERE id_str = %s",
                        (feedback, string_id)
                    )
                
                conn.commit()
                logger.info(f"Feedback updated for chat ID: {chat_id}")
                return True
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error updating feedback: {e}")
                conn.rollback()
                return False
            
    @staticmethod
    def delete_chat(chat_id) -> bool:
        """
        Delete a chat entry and its associated sources.
        
        Args:
            chat_id: The ID of the chat entry to delete (can be numeric or string UUID)
        
        Returns:
            bool: True if the deletion was successful, False otherwise
        """
        with get_db_connection() as conn:
            try:
                cursor = conn.cursor()
                db_id = None
                
                # Determine if the ID is a numeric ID or a string ID
                try:
                    # Try to convert to integer for numeric ID lookup
                    numeric_id = int(chat_id)
                    # Check if the chat exists
                    cursor.execute("SELECT id FROM chat_history WHERE id = %s", (numeric_id,))
                    chat_row = cursor.fetchone()
                    if not chat_row:
                        logger.error(f"Chat with numeric ID {numeric_id} not found")
                        return False
                    db_id = chat_row['id']
                    
                    # Delete by numeric ID
                    cursor.execute("DELETE FROM sources WHERE chat_id = %s", (db_id,))
                    cursor.execute("DELETE FROM chat_history WHERE id = %s", (numeric_id,))
                except (ValueError, TypeError):
                    # If conversion fails, treat it as a string ID
                    string_id = str(chat_id)
                    # Check if the chat exists
                    cursor.execute("SELECT id FROM chat_history WHERE id_str = %s", (string_id,))
                    chat_row = cursor.fetchone()
                    if not chat_row:
                        logger.error(f"Chat with string ID {string_id} not found")
                        return False
                    db_id = chat_row['id']
                    
                    # Delete the sources first using the numeric ID from the database
                    cursor.execute("DELETE FROM sources WHERE chat_id = %s", (db_id,))
                    # Then delete the chat by string ID
                    cursor.execute("DELETE FROM chat_history WHERE id_str = %s", (string_id,))
                
                conn.commit()
                logger.info(f"Chat with ID {chat_id} deleted successfully")
                return True
            except psycopg2.Error as e:
                logger.error(f"Error deleting chat: {e}")
                conn.rollback()
                return False
    
    @staticmethod
    def get_chat_by_id(chat_id) -> Optional[ChatHistoryEntry]:
        """
        Get a chat entry by ID. Can use either numeric ID or string ID (UUID).
        
        Args:
            chat_id: The ID of the chat entry (can be an integer or a string UUID)
            
        Returns:
            Optional[ChatHistoryEntry]: The chat entry if found, None otherwise
        """
        with get_db_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Determine if the ID is a numeric ID or a string ID
                try:
                    # Try to convert to integer for numeric ID lookup
                    numeric_id = int(chat_id)
                    cursor.execute(
                        "SELECT * FROM chat_history WHERE id = %s",
                        (numeric_id,)
                    )
                except (ValueError, TypeError):
                    # If conversion fails, treat it as a string ID
                    cursor.execute(
                        "SELECT * FROM chat_history WHERE id_str = %s",
                        (str(chat_id),)
                    )
                    
                chat_row = cursor.fetchone()
                    
                if not chat_row:
                    return None
                    
                # Get sources for this chat
                cursor.execute(
                    "SELECT url, cite_num FROM sources WHERE chat_id = %s",
                    (chat_row['id'],)
                )
                source_rows = cursor.fetchall()
                
                sources = [Source(url=row['url'], cite_num=row['cite_num']) for row in source_rows]
                
                return ChatHistoryEntry(
                    id=chat_row['id'],
                    query=chat_row['query'],
                    response=chat_row['response'],
                    sources=sources,
                    timestamp=chat_row['timestamp'],
                    feedback=chat_row['feedback'],
                    id_str=chat_row['id_str']
                )
            except psycopg2.Error as e:
                logger.error(f"Error retrieving chat: {e}")
                return None
    
    @staticmethod
    def get_all_chats(limit: int = 100, offset: int = 0) -> List[ChatHistoryEntry]:
        """
        Get all chat entries with pagination.
        
        Args:
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            
        Returns:
            List[ChatHistoryEntry]: List of chat entries
        """
        # Ensure limit and offset are integers
        try:
            limit = int(limit)
            offset = int(offset)
        except (ValueError, TypeError):
            logger.error(f"Invalid limit or offset format: limit={limit}, offset={offset}")
            return []
            
        with get_db_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Get chat history entries
                cursor.execute(
                    "SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT %s OFFSET %s",
                    (limit, offset)
                )
                chat_rows = cursor.fetchall()
                
                result = []
                for chat_row in chat_rows:
                    chat_id = chat_row['id']
                    
                    # Get sources for this chat
                    cursor.execute(
                        "SELECT url, cite_num FROM sources WHERE chat_id = %s",
                        (chat_id,)
                    )
                    source_rows = cursor.fetchall()
                    
                    sources = [Source(url=row['url'], cite_num=row['cite_num']) for row in source_rows]
                    
                    result.append(ChatHistoryEntry(
                        id=chat_row['id'],
                        query=chat_row['query'],
                        response=chat_row['response'],
                        sources=sources,
                        timestamp=chat_row['timestamp'],
                        feedback=chat_row['feedback']
                    ))
                
                return result
            except psycopg2.Error as e:
                logger.error(f"Error retrieving chats: {e}")
                return []
