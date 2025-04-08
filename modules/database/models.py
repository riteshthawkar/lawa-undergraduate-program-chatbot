"""
Database models and schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Source(BaseModel):
    """Model for a citation source"""
    url: str
    cite_num: str


class ChatHistoryEntry(BaseModel):
    """Model for a chat history entry"""
    id: Optional[int] = None
    query: str
    response: str
    sources: List[Source] = []
    timestamp: Optional[datetime] = None
    feedback: Optional[str] = None
    id_str: Optional[str] = None  # String ID (UUID) for the chat entry
    
    class Config:
        from_attributes = True


class FeedbackUpdate(BaseModel):
    """Model for updating feedback on a chat history entry"""
    feedback: str = Field(..., pattern="^(like|dislike)$")
