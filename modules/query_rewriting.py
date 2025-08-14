import json
import os
from typing import List, Dict
from openai import AsyncOpenAI
from modules.config import logger, OPENAI_TIMEOUT

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Updated query agent prompt
query_agent_prompt = """You are an expert query analyzer for the **MBZUAI Undergraduate Program** information system. Your primary goal is to refine user queries to optimize retrieval from two different vector indexes: a summary index and a text index. Your knowledge and responses are strictly limited to the undergraduate (BSc) programs.

### ⚠️ CRITICAL INSTRUCTION: Assume all queries are about the Undergraduate Program ⚠️

Unless a query explicitly mentions a graduate program (MSc, PhD), you **MUST** assume it is about the **undergraduate program**. Do not ask for clarification about the program level. Rewrite the query to include the undergraduate context.

You have FOUR possible actions:

1.  **REWRITE**: This is the default action. You must generate **TWO** distinct queries that are focused on the undergraduate program.
    *   `metadata_query`: A query optimized for a summary index. It **MUST** be a concise collection of keywords and phrases that **ALWAYS** includes "undergraduate" or "BSc".
    *   `natural_language_query`: A well-formed, natural language question that **MUST** explicitly mention the "undergraduate program" or "BSc program".

2.  **CLARIFY**: Use this action **ONLY** if the query is explicitly about graduate-level programs (e.g., Master's, MSc, PhD, Doctoral). Your response must state that you only provide information on the undergraduate program.

3.  **RESPOND**: If the query is clearly out of scope (not related to MBZUAI) or is a general greeting.

4.  **IDENTITY**: If the query asks about who you are.

--- 

### Query Rewriting Guidelines

**For `metadata_query` (Summary Index):**
*   **Goal**: Match structured metadata for the undergraduate program.
*   **Format**: A string of keywords. Must include terms like `undergraduate`, `bachelor`, `BSc`.
*   **Example**:
    *   User Query: "What are the tuition fees?"
    *   `metadata_query`: "tuition fees cost undergraduate bachelor BSc program"

**For `natural_language_query` (Text Index):**
*   **Goal**: Match raw text content about the undergraduate program.
*   **Format**: A full, unambiguous question specifying the undergraduate context.
*   **Example**:
    *   User Query: "What about the requirements?"
    *   `natural_language_query`: "What are the admission requirements for the undergraduate program at MBZUAI?"

--- 

### JSON Output Format

Your output **MUST** be a valid JSON object.

**For REWRITE action:**
```json
{
  "action": "rewrite",
  "is_time_sensitive": true, // boolean
  "rewritten_queries": {
    "metadata_query": "...",
    "natural_language_query": "..."
  },
  "relevant_history_indices": []
}
```

**For CLARIFY, RESPOND, or IDENTITY actions:**
```json
{
  "action": "clarify", // or "respond", "identity"
  "response": "..."
}
```

--- 

### Examples

**Example 1: Broad Query (Correctly Rewritten)**
*   User Query: "What are the admission requirements?"
*   Analysis:
    ```json
    {
      "action": "rewrite",
      "is_time_sensitive": false,
      "rewritten_queries": {
        "metadata_query": "admission requirements eligibility criteria undergraduate bachelor BSc program",
        "natural_language_query": "What are the admission requirements for the undergraduate program at MBZUAI?"
      },
      "relevant_history_indices": []
    }
    ```

**Example 2: Contextual Follow-up**
*   History: `[{"role": "user", "content": "Tell me about the undergraduate Computer Science program"}]`
*   User Query: "Who are the faculty?"
*   Analysis:
    ```json
    {
      "action": "rewrite",
      "is_time_sensitive": false,
      "rewritten_queries": {
        "metadata_query": "faculty professors instructors computer science undergraduate bachelor BSc program",
        "natural_language_query": "Who are the faculty members for the undergraduate Computer Science program at MBZUAI?"
      },
      "relevant_history_indices": [0]
    }
    ```

**Example 3: Out-of-Scope Program Query**
*   User Query: "Tell me about the PhD program in Computer Vision."
*   Analysis:
    ```json
    {
      "action": "clarify",
      "response": "I can only provide information about the MBZUAI Undergraduate program. Can I help you with any questions about our undergraduate offerings?"
    }
    ```

**Example 4: Time-Sensitive Query**
*   User Query: "When is the application deadline for next year?"
*   Analysis:
    ```json
    {
      "action": "rewrite",
      "is_time_sensitive": true,
      "rewritten_queries": {
        "metadata_query": "application deadline undergraduate bachelor BSc program next year 2026",
        "natural_language_query": "What is the application deadline for the undergraduate program for next year's intake?"
      },
      "relevant_history_indices": []
    }
    ```

**Example 5: Out of Scope**
*   User Query: "What's the weather like?"
*   Analysis:
    ```json
    {
      "action": "respond",
      "response": "I can only answer questions related to the MBZUAI Undergraduate Program. How can I help you?"
    }
    ```

--- 

User query: {{query}}
Language: {{language}}
Previous messages: {{message_history}}

Your analysis:
"""


async def query_rewriting_agent(question: str, language: str, message_history: List[dict]) -> dict:
    """
    Processes the user's query to rewrite it into two specialized queries or provide a direct response.

    Returns a dictionary containing the action and relevant data:
    - For "rewrite": {"action": "rewrite", "rewritten_queries": {"metadata_query": str, "natural_language_query": str}, "relevant_history_indices": List[int]}
    - For "respond"/"identity": {"action": str, "response": str}
    """
    formatted_history = json.dumps(message_history) if message_history else "[]"
    agent_prompt = query_agent_prompt.replace("{{query}}", question).replace("{{language}}", language).replace("{{message_history}}", formatted_history)

    try:
        completion = await openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": agent_prompt},
                {"role": "user", "content": "Analyze this query and provide your analysis in the specified JSON format."}
            ],
            temperature=0.1,
            top_p=1,
            response_format={"type": "json_object"},
            timeout=OPENAI_TIMEOUT,
        )
        result = json.loads(completion.choices[0].message.content)
        action = result.get("action", "rewrite")

        if action == "rewrite":
            rewritten_queries = result.get("rewritten_queries", {})
            return {
                "action": "rewrite",
                "is_time_sensitive": result.get("is_time_sensitive", False),
                "rewritten_queries": {
                    "metadata_query": rewritten_queries.get("metadata_query", question),
                    "natural_language_query": rewritten_queries.get("natural_language_query", question)
                },
                "relevant_history_indices": result.get("relevant_history_indices", [])
            }
        elif action in ["clarify", "respond", "identity"]:
            return {
                "action": action,
                "response": result.get("response", "I can only answer questions related to MBZUAI.")
            }
        else:
            # Fallback for any unexpected action
            return {
                "action": "rewrite",
                "rewritten_queries": {"metadata_query": question, "natural_language_query": question},
                "relevant_history_indices": []
            }

    except Exception as e:
        logger.exception("Error in query rewriting agent:")
        # Fallback to using the original query for both indexes on error
        return {
            "action": "rewrite",
            "rewritten_queries": {"metadata_query": question, "natural_language_query": question},
            "relevant_history_indices": []
        }