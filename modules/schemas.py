from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ChatRequest(BaseModel):
    """Request model for chat endpoints"""
    question: str = Field(..., max_length=1024)
    language: str
    # Use default_factory to avoid shared mutable defaults
    previous_chats: List[dict] = Field(default_factory=list)

class CitationSource(BaseModel):
    """Model for citation sources"""
    url: str
    cite_num: str 