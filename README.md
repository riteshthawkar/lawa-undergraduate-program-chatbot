# LAWA RAG Agent

A modular RAG (Retrieval-Augmented Generation) system for MBZUAI BSE undergraduate program queries with query rewriting, context filtering, and domain-specific knowledge expansion.

## Project Structure

The project is organized into modular components:

```
lawa-rag-agent/
├── app.py                    # Main application entry point and API endpoints
├── modules/                  # Modular components
│   ├── __init__.py           # Makes modules a package
│   ├── config.py             # Configuration, environment variables, system prompt
│   ├── citations.py          # Citation processing utilities
│   ├── query_rewriting.py    # Query rewriting and domain knowledge expansion
│   ├── retrieval.py          # Document retrieval and reranking
│   ├── schemas.py            # Pydantic models for data validation
│   └── utils.py              # Utility functions
├── .env                      # Environment variables (not in version control)
└── combined_vectorstore.json # BM25 sparse vectors for hybrid search
```

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_RESPONSE_MODEL=gpt-5.4
   OPENAI_QUERY_REWRITER_MODEL=gpt-5.4-mini
   OPENAI_RERANKER_MODEL=gpt-5.4-nano
   ```

4. Run the application:
   ```
   uvicorn app:app --reload
   ```

   If `8080` is already in use locally, start on another port:
   ```
   PORT=8001 python main.py
   ```

## API Endpoints

- WebSocket: `/chat` - For real-time chat interactions
- HTTP GET: `/health` - Health check endpoint

## Features

- **Query Rewriting**: Rewrites user queries for better retrieval performance
- **Message History Filtering**: Keeps only relevant conversation context
- **Domain Knowledge Expansion**: Enhances queries with MBZUAI BSE-specific terminology
- **Out-of-Scope Detection**: Directly responds to queries outside the system's scope (including Masters/PhD queries)
- **Clarification Requests**: Asks for more information when queries are ambiguous
- **Citation Processing**: Extracts and formats citations from responses
- **Fallback Search**: Uses Tavily search when Pinecone retrieval yields no results

## OpenAI Model Defaults

The backend now defaults to the GPT-5.4 family, with env-driven overrides:

- `OPENAI_RESPONSE_MODEL=gpt-5.4`
- `OPENAI_QUERY_REWRITER_MODEL=gpt-5.4-mini`
- `OPENAI_RERANKER_MODEL=gpt-5.4-nano`
- `OPENAI_RESPONSE_REASONING_EFFORT=low`
- `OPENAI_RESPONSE_MAX_COMPLETION_TOKENS=4096`
- `OPENAI_CLARIFICATION_MAX_COMPLETION_TOKENS=1024`
- `OPENAI_QUERY_REWRITER_REASONING_EFFORT=low`
- `OPENAI_RERANKER_REASONING_EFFORT=low`
