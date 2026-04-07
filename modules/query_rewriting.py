import json
import os
from typing import List, Dict
from openai import AsyncOpenAI
from modules.config import logger, OPENAI_TIMEOUT, QUERY_REWRITING_MODEL

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

"""Query rewriting rules:

1) If the user's query is ambiguous (no explicit program level), add undergraduate/BSc context to improve retrieval accuracy.
2) If the user explicitly asks about graduate programs (e.g., Master's/MSc/PhD/Doctoral), return a CLARIFY action asking if they want undergraduate information instead.
3) Do NOT modify queries about executive leadership, department chairs, named persons, or entities. Keep these neutral.
4) Otherwise, preserve the user's intent.

Final answer scope remains enforced by the chat system prompt (UG-only content).
"""

query_agent_prompt = """You are an expert query analyzer for the **MBZUAI information system**. Your primary goal is to refine user queries to optimize retrieval from two different vector indexes: a summary index and a text index.

### ⚠️ CRITICAL INSTRUCTION: Default to REWRITE; add Undergraduate context when ambiguous ⚠️

- If the user does NOT explicitly specify a program level (ambiguous query), you MUST add Undergraduate/BSc context to make retrieval more accurate and effective.
- If the user explicitly asks about a graduate program (Master/MSc/PhD/Doctoral), use **CLARIFY** action: tell them this agent provides undergraduate information only and ask if they want the undergraduate equivalent.
- NEVER add graduate-level (MSc/PhD) terms or context to the rewritten queries.
- Do NOT modify queries about executive leadership, department chairs, named persons, or entities; keep these queries neutral (no undergraduate injection).
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

1.  **REWRITE**: Default action. Generate **TWO** distinct queries.
    *   `metadata_query`: A concise collection of keywords optimized for a summary index. If the original query is ambiguous (no program level), you MUST add `undergraduate`, `bachelor`, or `BSc` to the keywords to bias retrieval correctly.
    *   `natural_language_query`: A well-formed, natural language question. If ambiguous, explicitly mention "undergraduate program" or "BSc program" to clarify intent.

2.  **CLARIFY**: If the user explicitly mentions graduate-level programs (Master/MSc/PhD/Doctoral). Tell them this assistant only provides undergraduate information and ask whether to proceed with undergraduate information.

3.  **RESPOND**: Only if the query is a greeting/small talk OR is clearly unrelated to universities/education/MBZUAI/AI/technology/student life.

--- 

### Query Rewriting Guidelines

**For `metadata_query` (Summary Index):**
*   **Goal**: Match structured metadata for the user's topic.
*   **Format**: A string of keywords. If ambiguous, include `undergraduate`, `bachelor`, `BSc` to improve retrieval. Otherwise avoid program-level markers.
*   **Example**:
    *   User Query: "What are the tuition fees?"
    *   `metadata_query`: "tuition fees cost undergraduate bachelor BSc program"

**For `natural_language_query` (Text Index):**
*   **Goal**: Match raw text content for the user's topic.
*   **Format**: A full, unambiguous question. If ambiguous, explicitly mention the "undergraduate program" to clarify.
*   **Example**:
    *   User Query: "What about the requirements?"
    *   `natural_language_query`: "What are the admission requirements for the undergraduate program at MBZUAI?"

**For `REWRITE` action (Default for all queries):**
*   **Use when**: The query has any plausible connection to MBZUAI, universities/education, AI/technology, student life, or campus services/utilities.
*   **IMPORTANT**: Always check chat history first! Use context to create more specific and targeted rewritten queries.
*   **Examples of queries that should be rewritten (NOT respond):**
    *   "What are the requirements?" → Search broadly for all types of requirements (add UG context)
    *   "Tell me about the program" → Search for comprehensive program information (add UG context)
    *   "How much does it cost?" → Search for all cost-related information (add UG context)
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

**For CLARIFY, RESPOND or IDENTITY actions:**
```json
{
  "action": "clarify", // or "respond", "identity"
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

**Example B: Broad Query (REWRITE; add UG context)**
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

**Example C: Graduate-Level Mention (CLARIFY)**
*   User Query: "Tell me about the PhD program in Computer Vision."
*   Analysis:
    ```json
    {
      "action": "clarify",
      "response": "I can only provide information about the MBZUAI Undergraduate program. Would you like me to proceed with undergraduate program information instead?"
    }
    ```

**Example D: Greeting (RESPOND)**
*   User Query: "hi there"
*   Analysis:
    ```json
    {
      "action": "respond",
      "response": "Hello! How can I help you with MBZUAI or undergraduate program information?"
    }
    ```

**Example E: Clearly Unrelated (RESPOND)**
*   User Query: "What's the weather like?"
*   Analysis:
    ```json
    {
      "action": "respond",
      "response": "I can help with MBZUAI undergraduate and campus questions. Try asking about admissions, programs, or campus services."
    }
    ```

**Example F: Identity (IDENTITY)**
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
            model=QUERY_REWRITING_MODEL,
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
