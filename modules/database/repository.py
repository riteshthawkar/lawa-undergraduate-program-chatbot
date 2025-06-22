"""
Repository for database operations using asyncpg.
"""
import asyncpg
from typing import List, Optional, Dict, Any, Tuple

from ..config import logger
from .models import ChatHistoryEntry, Source


class ChatRepository:
    """Repository for chat history operations using asyncpg."""
    
    @staticmethod
    async def save_chat(pool: asyncpg.Pool, query: str, response: str, sources: List[Dict[str, str]], custom_id: Optional[str] = None, rag_app_name: Optional[str] = None) -> Optional[int]:
        """
        Save a chat interaction to the database asynchronously.
        
        Args:
            pool: asyncpg Pool instance
            query: The user's query
            response: The LLM's response
            sources: List of citation sources
            custom_id: Optional custom ID for the chat (passed to the id_str column)
        
        Returns:
            Optional[int]: The ID of the inserted chat entry, or None if save failed
        """
        if not pool:
            logger.error("Database pool instance was not provided to save_chat.")
            return None
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Insert chat history entry with optional custom ID
                    # Use $1, $2 placeholders for asyncpg
                    # Base query and params
                    sql_query = "INSERT INTO chat_history (query, response, rag_app_name"
                    params = [query, response, rag_app_name]
                    
                    # Add custom_id if it exists
                    if custom_id:
                        sql_query += ", id_str"
                        params.append(custom_id)
                    
                    # Finalize query string
                    placeholders = ", ".join([f"${i+1}" for i in range(len(params))])
                    sql_query += f") VALUES ({placeholders}) RETURNING id"

                    # Insert chat history entry
                    chat_record = await conn.fetchrow(sql_query, *params)
                    
                    if not chat_record:
                        raise Exception("Failed to insert chat record or retrieve ID.")
                        
                    chat_id = chat_record['id']
                    
                    # Insert sources if any
                    if sources:
                        # Prepare data for executemany
                        source_data = [(chat_id, source["url"], source["cite_num"]) for source in sources]
                        await conn.executemany(
                            "INSERT INTO sources (chat_id, url, cite_num) VALUES ($1, $2, $3)",
                            source_data
                        )
                    
                    logger.info(f"Chat saved with ID: {chat_id}")
                    return chat_id
                except asyncpg.PostgresError as e:
                    logger.error(f"Error saving chat: {e}")
                    # Transaction automatically rolled back
                    return None
                except Exception as e:
                    logger.error(f"Unexpected error saving chat: {e}")
                    # Transaction automatically rolled back
                    return None
    
    @staticmethod
    async def update_feedback(pool: asyncpg.Pool, chat_id, feedback: str) -> bool:
        """
        Update feedback for a chat entry asynchronously.
        
        Args:
            pool: asyncpg Pool instance
            chat_id: The ID of the chat entry (can be numeric or string UUID)
            feedback: The feedback to set ('like' or 'dislike')
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        if not pool:
            logger.error("Database pool instance was not provided to update_feedback.")
            return False

        async with pool.acquire() as conn:
            async with conn.transaction():
                try:
                    rows_affected = 0
                    # Determine if the ID is a numeric ID or a string ID
                    try:
                        # Try to convert to integer for numeric ID lookup
                        numeric_id = int(chat_id)
                        result = await conn.execute(
                            "UPDATE chat_history SET feedback = $1 WHERE id = $2",
                            feedback, numeric_id
                        )
                        # 'UPDATE 1' -> 1 row affected
                        rows_affected = int(result.split(' ')[1]) if result else 0
                            
                    except (ValueError, TypeError):
                        # If conversion fails, treat it as a string ID
                        string_id = str(chat_id)
                        result = await conn.execute(
                            "UPDATE chat_history SET feedback = $1 WHERE id_str = $2",
                            feedback, string_id
                        )
                        rows_affected = int(result.split(' ')[1]) if result else 0
                        
                    if rows_affected > 0:
                        logger.info(f"Feedback '{feedback}' updated for chat ID: {chat_id}")
                        return True
                    else:
                        logger.warning(f"No chat found with ID: {chat_id} to update feedback.")
                        return False
                        
                except asyncpg.PostgresError as e:
                    logger.error(f"Error updating feedback for chat ID {chat_id}: {e}")
                    # Transaction automatically rolled back
                    return False
                except Exception as e:
                    logger.error(f"Unexpected error updating feedback for chat ID {chat_id}: {e}")
                    # Transaction automatically rolled back
                    return False

    @staticmethod
    async def delete_chat(pool: asyncpg.Pool, chat_id) -> bool:
        """
        Delete a chat entry and its associated sources asynchronously.
        
        Args:
            pool: asyncpg Pool instance
            chat_id: The ID of the chat entry to delete (can be numeric or string UUID)
        
        Returns:
            bool: True if the deletion was successful, False otherwise
        """
        if not pool:
            logger.error("Database pool instance was not provided to delete_chat.")
            return False

        async with pool.acquire() as conn:
            async with conn.transaction():
                try:
                    rows_affected = 0
                    # Determine ID type and execute delete
                    try:
                        numeric_id = int(chat_id)
                        # Deletion cascades to sources table due to FOREIGN KEY constraint
                        result = await conn.execute(
                            "DELETE FROM chat_history WHERE id = $1",
                            numeric_id
                        )
                        rows_affected = int(result.split(' ')[1]) if result else 0
                    except (ValueError, TypeError):
                        string_id = str(chat_id)
                        result = await conn.execute(
                            "DELETE FROM chat_history WHERE id_str = $1",
                            string_id
                        )
                        rows_affected = int(result.split(' ')[1]) if result else 0

                    if rows_affected > 0:
                        logger.info(f"Chat entry with ID {chat_id} deleted successfully.")
                        return True
                    else:
                        logger.warning(f"No chat entry found with ID {chat_id} to delete.")
                        return False
                        
                except asyncpg.PostgresError as e:
                    logger.error(f"Error deleting chat ID {chat_id}: {e}")
                    # Transaction automatically rolled back
                    return False
                except Exception as e:
                    logger.error(f"Unexpected error deleting chat ID {chat_id}: {e}")
                    # Transaction automatically rolled back
                    return False

    @staticmethod
    async def get_chat_by_id(pool: asyncpg.Pool, chat_id) -> Optional[ChatHistoryEntry]:
        """
        Get a chat entry by ID asynchronously. Can use either numeric ID or string ID (UUID).
        
        Args:
            pool: asyncpg Pool instance
            chat_id: The ID of the chat entry (can be an integer or a string UUID)
            
        Returns:
            Optional[ChatHistoryEntry]: The chat entry if found, None otherwise
        """
        if not pool:
            logger.error("Database pool instance was not provided to get_chat_by_id.")
            return None
            
        async with pool.acquire() as conn:
            try:
                chat_row = None
                # Determine ID type and fetch chat
                try:
                    numeric_id = int(chat_id)
                    chat_row = await conn.fetchrow(
                        "SELECT * FROM chat_history WHERE id = $1",
                        numeric_id
                    )
                except (ValueError, TypeError):
                    string_id = str(chat_id)
                    chat_row = await conn.fetchrow(
                        "SELECT * FROM chat_history WHERE id_str = $1",
                        string_id
                    )
                    
                if not chat_row:
                    logger.warning(f"No chat found with ID: {chat_id}")
                    return None
                    
                # Get sources for this chat using the numeric ID from chat_row
                actual_chat_id = chat_row['id']
                source_rows = await conn.fetch(
                    "SELECT url, cite_num FROM sources WHERE chat_id = $1",
                    actual_chat_id
                )
                
                sources = [Source(url=row['url'], cite_num=row['cite_num']) for row in source_rows]
                
                return ChatHistoryEntry(
                    id=chat_row['id'],
                    query=chat_row['query'],
                    response=chat_row['response'],
                    sources=sources,
                    timestamp=chat_row['timestamp'],
                    feedback=chat_row['feedback'],
                    id_str=chat_row['id_str'],
                    rag_app_name=chat_row.get('rag_app_name') # Use .get for safety

                )
            except asyncpg.PostgresError as e:
                logger.error(f"Error retrieving chat ID {chat_id}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error retrieving chat ID {chat_id}: {e}")
                return None
    
    @staticmethod
    async def get_all_chats(pool: asyncpg.Pool, limit: int = 100, offset: int = 0) -> List[ChatHistoryEntry]:
        """
        Get all chat entries asynchronously with pagination.
        
        Args:
            pool: asyncpg Pool instance
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            
        Returns:
            List[ChatHistoryEntry]: List of chat entries
        """
        if not pool:
            logger.error("Database pool instance was not provided to get_all_chats.")
            return []
            
        # Ensure limit and offset are integers
        try:
            limit = int(limit)
            offset = int(offset)
        except (ValueError, TypeError):
            logger.error(f"Invalid limit or offset format: limit={limit}, offset={offset}")
            return []
            
        async with pool.acquire() as conn:
            try:
                # Get chat history entries
                chat_rows = await conn.fetch(
                    "SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT $1 OFFSET $2",
                    limit, offset
                )
                
                result = []
                if not chat_rows:
                    return []
                
                # Fetch sources for all retrieved chat IDs in a single query for efficiency
                chat_ids = [row['id'] for row in chat_rows]
                all_source_rows = await conn.fetch(
                    "SELECT chat_id, url, cite_num FROM sources WHERE chat_id = ANY($1::int[])",
                    chat_ids
                )
                
                # Group sources by chat_id
                sources_by_chat_id = {}
                for src_row in all_source_rows:
                    chat_id = src_row['chat_id']
                    if chat_id not in sources_by_chat_id:
                        sources_by_chat_id[chat_id] = []
                    sources_by_chat_id[chat_id].append(Source(url=src_row['url'], cite_num=src_row['cite_num']))

                # Build the final list of ChatHistoryEntry objects
                for chat_row in chat_rows:
                    chat_id = chat_row['id']
                    sources = sources_by_chat_id.get(chat_id, []) # Get sources or empty list
                    result.append(ChatHistoryEntry(
                        id=chat_id,
                        query=chat_row['query'],
                        response=chat_row['response'],
                        sources=sources,
                        timestamp=chat_row['timestamp'],
                        feedback=chat_row['feedback'],
                        id_str=chat_row['id_str'], # Include id_str
                        rag_app_name=chat_row.get('rag_app_name') # Use .get for safety
                    ))
                
                return result
            except asyncpg.PostgresError as e:
                logger.error(f"Error retrieving chats: {e}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error retrieving chats: {e}")
                return []
