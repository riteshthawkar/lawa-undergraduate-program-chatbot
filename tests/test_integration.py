import asyncio
import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.query_rewriting import query_rewriting_agent
from modules.retrieval import rerank_docs
from modules.citations import process_citations

class MockWebSocket:
    """Mock WebSocket class for testing."""
    
    def __init__(self):
        self.sent_messages = []
    
    async def send_json(self, data):
        self.sent_messages.append(data)
        
    async def receive_json(self):
        return {
            "question": "Who are you?",
            "language": "en",
            "previous_chats": []
        }
        
    async def accept(self):
        pass

class TestIntegration(unittest.TestCase):
    """Integration tests for the application."""
    
    @patch('modules.query_rewriting.openai_client.chat.completions.create', new_callable=AsyncMock)
    def test_identity_query_flow(self, mock_create):
        """Test the flow for an identity query."""
        # Mock the OpenAI client response for the identity query
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({
            "action": "identity",
            "response": "I am an AI assistant developed by lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on MBZUAI admissions program."
        })
        mock_create.return_value = mock_completion
        
        # Call the query rewriting agent and check the result
        async def run_test():
            result = await query_rewriting_agent("Who are you?", "en", [])
            self.assertEqual(result["action"], "respond")
            self.assertEqual(
                result["response"],
                "I am an AI assistant developed by lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on MBZUAI admissions program."
            )
        
        # Run the async test
        asyncio.run(run_test())
    
    @patch('modules.query_rewriting.openai_client.chat.completions.create', new_callable=AsyncMock)
    def test_rewrite_query_flow(self, mock_create):
        """Test the flow for a query that gets rewritten."""
        # Set up the mock to return different values on subsequent calls
        mock_completion1 = MagicMock()
        mock_completion1.choices = [MagicMock()]
        mock_completion1.choices[0].message.content = json.dumps({
            "action": "rewrite",
            "rewritten_query": "What are the admission requirements for the MBZUAI BSE undergraduate program?",
            "relevant_history_indices": []
        })
        
        mock_completion2 = MagicMock()
        mock_completion2.choices = [MagicMock()]
        mock_completion2.choices[0].message.content = "What are the admission requirements and application process for the MBZUAI BSE undergraduate program?"
        
        mock_create.side_effect = [mock_completion1, mock_completion2]
        
        # Call the query rewriting agent and check the result
        async def run_test():
            result = await query_rewriting_agent("Tell me about admissions", "en", [])
            self.assertEqual(result["action"], "rewrite")
            self.assertIn("admission", result["rewritten_query"].lower())
            self.assertIn("mbzuai", result["rewritten_query"].lower())
        
        # Run the async test
        asyncio.run(run_test())
    
    @patch('modules.query_rewriting.openai_client.chat.completions.create', new_callable=AsyncMock)
    def test_out_of_scope_query_flow(self, mock_create):
        """Test the flow for an out-of-scope query."""
        # Mock the OpenAI client response for the out-of-scope query
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({
            "action": "respond",
            "response": "I'm sorry, but questions about iPhone repairs are outside my scope. I can only answer questions related to MBZUAI's BSE undergraduate program, including admissions, campus life, and related matters."
        })
        mock_create.return_value = mock_completion
        
        # Call the query rewriting agent and check the result
        async def run_test():
            result = await query_rewriting_agent("How do I fix my broken iPhone screen?", "en", [])
            self.assertEqual(result["action"], "respond")
            self.assertIn("outside my scope", result["response"].lower())
        
        # Run the async test
        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main() 