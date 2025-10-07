import json
import os
from typing import List, Dict
from openai import AsyncOpenAI
from modules.config import logger, OPENAI_TIMEOUT

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

"""Query rewriting: keep general across queries while preserving UG scope in answers.

We no longer hard-code leadership detection here. Retrieval stays broad and neutral.
Final answer scope is still enforced by the chat system prompt (UG-only content).
"""

# Updated query agent prompt
query_agent_prompt = """You are an expert query analyzer for the **MBZUAI information system**. Your primary goal is to refine user queries to optimize retrieval from two different vector indexes: a summary index and a text index. Keep rewrites neutral unless the query clearly targets the undergraduate (BSc) program.

### ⚠️ CRITICAL INSTRUCTION: Default to REWRITE; RESPOND only for greetings or clearly unrelated ⚠️

- Do NOT inject a program level unless it is explicitly requested or obviously implied by the user (e.g., the user clearly asks about the undergraduate program).
- NEVER add graduate-level (MSc/PhD) terms or context to the rewritten queries.
- Treat campus services and institutional utilities (e.g., campus Wi‑Fi, email, LMS, ID cards, transport, housing, dining, events, offices/leadership) as in‑scope for MBZUAI. These MUST NOT trigger an out‑of‑scope response.
- "RESPOND" is allowed ONLY when the message is:
  - a greeting or small talk (e.g., "hi", "hello", "good morning"), OR
  - clearly unrelated to universities/education, MBZUAI, Abu Dhabi/UAE higher‑ed context, AI, technology, or student life (e.g., generic weather, celebrity gossip, unrelated math puzzles).
- If MBZUAI, universities, education, AI, or technology are even implicitly relevant, you MUST choose REWRITE.

### 🔍 CONTEXT AWARENESS: Use Chat History to Enhance Query Rewriting

- **ALWAYS analyze the chat history first** to enhance query rewriting.
- If previous messages clarify an ambiguous query, use that context to make specific rewritten queries.
- Look for recent topics, programs, or ongoing discussions that disambiguate vague queries.

You have THREE possible actions:

1.  **REWRITE**: This is the default action. You must generate **TWO** distinct queries.
    *   `metadata_query`: A concise collection of keywords optimized for a summary index. Include program markers like "undergraduate", "bachelor", or "BSc" only if the query clearly concerns the undergraduate program. Otherwise, keep it neutral.
    *   `natural_language_query`: A well-formed, natural language question that preserves the user's original intent. Only add "undergraduate" or "BSc" if explicitly relevant; otherwise, keep it neutral.

2.  **RESPOND**: Only if the query is a greeting/small talk OR is clearly unrelated to universities/education/MBZUAI/AI/technology/student life.

3.  **IDENTITY**: If the query asks about who you are.

--- 

### Query Rewriting Guidelines

**For `metadata_query` (Summary Index):**
*   **Goal**: Match structured metadata for the user's topic.
*   **Format**: A string of keywords. Include `undergraduate`, `bachelor`, `BSc` only when clearly relevant; otherwise avoid program-level markers.
*   **Example**:
    *   User Query: "What are the tuition fees?"
    *   `metadata_query`: "tuition fees cost undergraduate bachelor BSc program"

**For `natural_language_query` (Text Index):**
*   **Goal**: Match raw text content for the user's topic.
*   **Format**: A full, unambiguous question that stays neutral unless the query clearly targets the undergraduate program.
*   **Example**:
    *   User Query: "What about the requirements?"
    *   `natural_language_query`: "What are the admission requirements for the undergraduate program at MBZUAI?"

**For `REWRITE` action (Default for all queries):**
*   **Use when**: The query has any plausible connection to MBZUAI, universities/education, AI/technology, student life, or campus services/utilities.
*   **IMPORTANT**: Always check chat history first! Use context to create more specific and targeted rewritten queries.
*   **Examples of queries that should be rewritten (NOT respond):**
    *   "What are the requirements?" → Search broadly for all types of requirements
    *   "Tell me about the program" → Search for comprehensive program information
    *   "How much does it cost?" → Search for all cost-related information
    *   "How can I connect to MBZUAI Wi‑Fi?" → Treat as in‑scope campus utilities; search for IT/Wi‑Fi setup instructions
    *   "Where can I get my student ID?" → Search for student affairs/ID office information
    *   "Who can help me?" → Search for contact and support information
*   **Response format**: Generate two optimized queries for different retrieval strategies

--- 

### JSON Output Format

Your output **MUST** be a valid JSON object.

**For REWRITE action:**
```json
{
  "action": "rewrite",
  "is_time_sensitive": true,
  "rewritten_queries": {
    "metadata_query": "...",
    "natural_language_query": "..."
  }
}
```

**For RESPOND or IDENTITY actions:**
```json
{
  "action": "respond", // or "identity"
  "response": "..."
}
```

--- 

### Examples

**Example A: Campus Utility (In‑Scope, REWRITE)**
*   User Query: "How can I connect to MBZUAI wifi?"
*   Analysis:
    ```json
    {
      "action": "rewrite",
      "is_time_sensitive": false,
      "rewritten_queries": {
        "metadata_query": "wifi network campus internet it support onboarding setup student services",
        "natural_language_query": "How do I connect to the MBZUAI campus Wi‑Fi network?"
      }
    }
    ```

**Example B: Broad Query (REWRITE)**
*   User Query: "What are the admission requirements?"
*   Analysis:
    ```json
    {
      "action": "rewrite",
      "is_time_sensitive": false,
      "rewritten_queries": {
        "metadata_query": "admission requirements eligibility criteria undergraduate bachelor BSc program",
        "natural_language_query": "What are the admission requirements for the undergraduate program at MBZUAI?"
      }
    }
    ```

**Example C: Greeting (RESPOND)**
*   User Query: "hi there"
*   Analysis:
    ```json
    {
      "action": "respond",
      "response": "Hello! How can I help you with MBZUAI or undergraduate program information?"
    }
    ```

**Example D: Clearly Unrelated (RESPOND)**
*   User Query: "What's the weather like?"
*   Analysis:
    ```json
    {
      "action": "respond",
      "response": "I can help with MBZUAI undergraduate and campus questions. Try asking about admissions, programs, or campus services."
    }
    ```

**Example E: Identity (IDENTITY)**
*   User Query: "Who are you?"
*   Analysis:
    ```json
    {
      "action": "identity",
      "response": "I’m the MBZUAI undergraduate information assistant."
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
                }
            }
        elif action in ["respond", "identity"]:
            return {
                "action": action,
                "response": result.get("response", "I can only answer questions related to MBZUAI.")
            }
        else:
            # Fallback for any unexpected action
            return {
                "action": "rewrite",
                "rewritten_queries": {"metadata_query": question, "natural_language_query": question}
            }

    except Exception as e:
        logger.exception("Error in query rewriting agent:")
        # Fallback to using the original query for both indexes on error
        return {
            "action": "rewrite",
            "rewritten_queries": {"metadata_query": question, "natural_language_query": question}
        }