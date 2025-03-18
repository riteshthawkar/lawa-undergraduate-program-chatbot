import asyncio
import unittest
import sys
import os
import json

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.query_rewriting import query_rewriting_agent

class TestQueryRewritingAgent(unittest.TestCase):
    """Test cases for the query rewriting agent."""

    def setUp(self):
        """Set up the test environment."""
        # Create a sample message history for testing
        self.message_history = [
            {"role": "user", "content": "What are the admission requirements for BSE program?"},
            {"role": "assistant", "content": "MBZUAI's BSE undergraduate program requires strong academics, particularly in mathematics and computer science..."},
            {"role": "user", "content": "What documents are needed for application?"},
            {"role": "assistant", "content": "For MBZUAI's BSE undergraduate application, you generally need transcripts, standardized test scores, recommendation letters..."},
        ]

    def test_rewrite_query(self):
        """Test that a MBZUAI-related query gets rewritten."""
        result = asyncio.run(query_rewriting_agent("Tell me about admissions", "en", []))
        self.assertEqual(result["action"], "rewrite")
        self.assertIn("admission", result["rewritten_query"].lower())
        self.assertIn("mbzuai", result["rewritten_query"].lower())

    def test_out_of_scope_query(self):
        """Test that an out-of-scope query gets a respond action."""
        result = asyncio.run(query_rewriting_agent("How do I fix my broken iPhone screen?", "en", []))
        self.assertEqual(result["action"], "respond")
        self.assertIn("outside my scope", result["response"].lower())

    def test_greeting_query(self):
        """Test that a greeting gets a respond action with a friendly response."""
        result = asyncio.run(query_rewriting_agent("Hello, how are you today?", "en", []))
        self.assertEqual(result["action"], "respond")
        self.assertIn("hello", result["response"].lower())

    def test_vague_query(self):
        """Test that a vague query gets a clarify action."""
        result = asyncio.run(query_rewriting_agent("What are the requirements?", "en", []))
        self.assertEqual(result["action"], "clarify")
        self.assertIn("specify", result["response"].lower())

    def test_identity_query_who(self):
        """Test that an identity question about 'who are you' gets the standard identity response."""
        result = asyncio.run(query_rewriting_agent("Who are you?", "en", []))
        self.assertEqual(result["action"], "respond")
        self.assertEqual(
            result["response"],
            "I am an AI assistant developed by lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on MBZUAI admissions program."
        )

    def test_identity_query_what(self):
        """Test that an identity question about 'what are you' gets the standard identity response."""
        result = asyncio.run(query_rewriting_agent("What are you?", "en", []))
        self.assertEqual(result["action"], "respond")
        self.assertEqual(
            result["response"],
            "I am an AI assistant developed by lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on MBZUAI admissions program."
        )

    def test_identity_query_model(self):
        """Test that an identity question about the model gets the standard identity response."""
        result = asyncio.run(query_rewriting_agent("Which model are you using?", "en", []))
        self.assertEqual(result["action"], "respond")
        self.assertEqual(
            result["response"],
            "I am an AI assistant developed by lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on MBZUAI admissions program."
        )

    def test_identity_query_capabilities(self):
        """Test that an identity question about capabilities gets the standard identity response."""
        result = asyncio.run(query_rewriting_agent("What can you do?", "en", []))
        self.assertEqual(result["action"], "respond")
        self.assertEqual(
            result["response"],
            "I am an AI assistant developed by lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on MBZUAI admissions program."
        )

    def test_query_with_history(self):
        """Test that a query with relevant history incorporates that history."""
        result = asyncio.run(query_rewriting_agent("How long does the application process take?", "en", self.message_history))
        self.assertEqual(result["action"], "rewrite")
        self.assertIn("application", result["rewritten_query"].lower())
        self.assertTrue(len(result["relevant_history_indices"]) > 0)
        
    def test_masters_phd_query(self):
        """Test that a query about Masters or PhD programs gets marked as out of scope."""
        result = asyncio.run(query_rewriting_agent("Tell me about PhD admissions at MBZUAI", "en", []))
        self.assertEqual(result["action"], "respond")
        self.assertIn("outside my scope", result["response"].lower())
        self.assertIn("undergraduate", result["response"].lower())

    def test_error_handling(self):
        """Test that the agent handles errors gracefully by returning the original query."""
        # Mock a scenario where the LLM call would fail
        # This is a bit tricky to test without mocking, but we can at least ensure the function doesn't crash
        try:
            result = asyncio.run(query_rewriting_agent("", "en", []))
            # The agent might return either "rewrite" or "respond" for an empty query
            # Both are acceptable behaviors
            self.assertIn(result["action"], ["rewrite", "respond"])
            if result["action"] == "rewrite":
                self.assertEqual(result["rewritten_query"], "")
            else:
                self.assertIn("response", result)
        except Exception as e:
            self.fail(f"query_rewriting_agent raised an exception: {e}")

if __name__ == "__main__":
    unittest.main() 