import asyncio
import re
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# Import modules
from modules.config import (
    logger,
    validate_env_vars,
    get_system_prompt,
    RAG_APP_NAME,
    OPENAI_TIMEOUT,
    OPENAI_RESPONSE_MODEL,
    OPENAI_RESPONSE_REASONING_EFFORT,
    OPENAI_RESPONSE_MAX_COMPLETION_TOKENS,
    OPENAI_CLARIFICATION_MAX_COMPLETION_TOKENS,
    OPENAI_QUERY_REWRITER_MODEL,
    OPENAI_RERANKER_MODEL,
)
from modules.schemas import ChatRequest
from modules.utils import safe_send, format_query
from modules.citations import process_citations
from modules.retrieval import initialize_retrieval_components, rerank_docs, fetch_balanced_documents
from pinecone import Pinecone
from modules.query_rewriting import query_rewriting_agent, openai_client

# Import database modules
from modules.database.database import init_db, connect_db, disconnect_db
from modules.database.repository import ChatRepository
from modules.database.views import router as history_router

# ------------------------------------------------------------------------------
# Initialize application and validate environment
# ------------------------------------------------------------------------------
validate_env_vars()



# ------------------------------------------------------------------------------
# Lifespan manager for database connection pool
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize all components
    logger.info("Application startup: Initializing components...")
    
    # --- Database Connection ---
    pool = await connect_db()
    await init_db(pool)
    app.state.pool = pool
    logger.info("Database pool initialized and assigned to app.state.")

    # --- Retrieval Components (Embeddings, BM25) ---
    app.state.embed_model, app.state.bm25_encoder = initialize_retrieval_components()
    logger.info("Embedding model and BM25 encoder initialized and assigned to app.state.")

    # --- Pinecone Client and Async Indexes ---
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    
    summary_index_name = os.getenv("PINECONE_SUMMARY_INDEX_NAME", "mbzuai-undergraduate-summary-only-index")
    text_index_name = os.getenv("PINECONE_TEXT_INDEX_NAME", "mbzuai-undergraduate-text-only-index")

    logger.info(f"Connecting to Pinecone indexes: '{summary_index_name}' and '{text_index_name}'...")
    app.state.pinecone_summary_index = pc.Index(summary_index_name)
    app.state.pinecone_text_index = pc.Index(text_index_name)
    logger.info("Pinecone async index objects created and assigned to app.state.")

    yield # Application runs here

    # Shutdown: Disconnect from all services
    logger.info("Application shutdown: Cleaning up resources...")
    await disconnect_db(app.state.pool)
    logger.info("Database connection pool closed.")
    # Note: pinecone-client > 4.0.0 does not require explicit close for indexes.

# ------------------------------------------------------------------------------
# Initialize FastAPI app with CORS middleware and lifespan manager
# ------------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# CORS configuration (allow all origins)
# Note: When allowing all origins, credentials must be disabled by browser rules.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Setup static files and templates
# ------------------------------------------------------------------------------
# Get the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------
# Include database history router (endpoints within will need async updates too)
# ------------------------------------------------------------------------------
app.include_router(history_router, prefix="/api")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_health_probe(headers: Any) -> bool:
    try:
        value = headers.get("x-health-probe") if hasattr(headers, "get") else None
        if value is None and isinstance(headers, dict):
            value = headers.get("x-health-probe") or headers.get("X-Health-Probe")
        return str(value or "").strip().lower() in {"1", "true", "yes"}
    except Exception:
        return False


def _extract_completion_text(completion: Any) -> str:
    if completion and getattr(completion, "choices", None):
        message = completion.choices[0].message
        return (getattr(message, "content", None) or "").strip()
    return ""


def _probe_error_response(error_code: str, message: str, *, details: Optional[str] = None) -> JSONResponse:
    payload = {
        "status": "unhealthy",
        "probe": True,
        "timestamp": _utc_timestamp(),
        "error": error_code,
        "message": message,
        "sources": [],
    }
    if details:
        payload["details"] = details
    return JSONResponse(status_code=503, content=payload)

# ------------------------------------------------------------------------------
# WebSocket endpoint for chat functionality with improved error handling
# ------------------------------------------------------------------------------
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New client connected")
    # Notify frontend that the connection is established
    try:
        await safe_send(websocket, {"status": "connected", "message": "Connected to assistant."})
    except Exception:
        pass
    
    while True:
        try:
            # Wait for client messages without a timeout
            data = await websocket.receive_json()
            # Inform the client we received the request
            await safe_send(websocket, {"status": "received", "message": "Request received. Preparing to process..."})
            
            # --- Feedback Handling --- 
            if "feedback" in data and "id" in data:
                try:
                    chat_id_str = data.get("id") 
                    feedback = data.get("feedback")
                    
                    if chat_id_str and feedback in ["like", "dislike"]:
                        logger.info(f"Received feedback '{feedback}' for chat ID: {chat_id_str}")
                        success = await ChatRepository.update_feedback(websocket.app.state.pool, chat_id_str, feedback)
                        if success:
                            await safe_send(websocket, {"status": "feedback_updated", "id": chat_id_str})
                        else:
                            await safe_send(websocket, {"status": "feedback_failed", "id": chat_id_str})
                    else:
                        # Invalid feedback content, but it was intended as feedback
                        logger.warning(f"Invalid feedback data received: id={chat_id_str}, feedback={feedback}")
                        await safe_send(websocket, {"status": "invalid_feedback_data", "id": chat_id_str})
                
                except Exception as feedback_err:
                    # Error during feedback processing
                    logger.exception(f"Error processing feedback: {feedback_err}")
                    await safe_send(websocket, {"status": "feedback_error", "id": data.get('id', 'unknown')})
                
                # Skip to the next message if it was feedback (processed or error)
                continue 
            
            # --- Chat Message Handling (only if not feedback) ---
            try:
                chat_request = ChatRequest(**data)
            except ValidationError as ve:
                error_details = str(ve)
                logger.exception(f"Validation error: {error_details}")
                await safe_send(websocket, {
                    "response": f"Invalid request format. Please ensure your message contains the required fields: question, language, and previous_chats.", 
                    "error": "validation_error",
                    "error_details": error_details,
                    "sources": []
                })
                continue # Skip processing this invalid message

            question = chat_request.question
            language = chat_request.language
            previous_chats = chat_request.previous_chats

            # Apply query rewriting agent to analyze and possibly rewrite the query
            agent_result = await query_rewriting_agent(question, language, previous_chats)
            
            # Handle direct responses (out of scope or identity questions)
            if agent_result["action"] in ["respond", "identity"]:
                # Generate a unique string ID for this chat
                chat_str_id = str(uuid.uuid4())
                
                # Send the response to the client with the string ID
                await safe_send(websocket, {
                    "response": agent_result["response"], 
                    "sources": [],
                    "id": chat_str_id  # Include the string ID with the response
                })
                
                # Save the query-rewriting agent's response to the database
                try:
                    # Save to the database using our custom string ID
                    chat_id = await ChatRepository.save_chat(app.state.pool, question, agent_result["response"], [], chat_str_id, RAG_APP_NAME)
                    logger.info(f"Query-rewriting agent response saved to database with numeric ID: {chat_id}, string ID: {chat_str_id}, and app: {RAG_APP_NAME}")
                except Exception as e:
                    logger.exception(f"Failed to save query-rewriting agent response to database: {e}")
                
                continue
                
            # The agent returns two specialized queries. Use them for retrieval.
            metadata_query = agent_result.get("metadata_query", question)
            natural_language_query = agent_result.get("natural_language_query", question)
            logger.info(f"Using Metadata Query: '{metadata_query}' and Natural Language Query: '{natural_language_query}'")

            # Use last 5 message-response pairs for context (simple and reliable)
            relevant_history = []
            if previous_chats:
                # Take the last 10 messages (5 pairs) or all if fewer
                max_messages = min(10, len(previous_chats))
                relevant_history = previous_chats[-max_messages:]
                logger.info(f"Using last {len(relevant_history)} messages for context (out of {len(previous_chats)} total)")
            else:
                # If no previous chats, use empty history
                relevant_history = []
            
            await safe_send(websocket, {"status": "rewriting", "message": "Analyzing and optimizing your question for best results..."})
            start_time_retrieval = time.time()

            # Leverage the previously computed agent_result
            if agent_result.get("action") != "rewrite":
                await safe_send(websocket, {"type": "final_answer", "response": agent_result.get("response", "..."), "sources": []})
                return

            rewritten_queries = agent_result.get("rewritten_queries", {})
            metadata_query = rewritten_queries.get("metadata_query", question)
            natural_language_query = rewritten_queries.get("natural_language_query", question)
            logger.info(f"Rewritten queries generated: metadata='{metadata_query}', natural_language='{natural_language_query}'")
            await safe_send(websocket, {"status": "rewritten", "message": "Your question has been optimized. Searching knowledge base..."})

            # Step 3: Fetch documents from Pinecone using the rewritten queries
            logger.info("Step 3: Fetching documents from Pinecone...")
            await safe_send(websocket, {"status": "retrieving", "message": "Searching our knowledge base for relevant documents..."})
            try:
                docs = await fetch_balanced_documents(
                    rewritten_queries=rewritten_queries,
                    pinecone_summary_index=websocket.app.state.pinecone_summary_index,
                    pinecone_text_index=websocket.app.state.pinecone_text_index,
                    embed_model=websocket.app.state.embed_model,
                    bm25_encoder=websocket.app.state.bm25_encoder
                )
                if not docs:
                    logger.warning("No documents found in Pinecone.")
                    # Use main response generation agent to provide intelligent clarification
                    await safe_send(websocket, {"status": "no_docs", "message": "No relevant documents found. Generating clarification response..."})
                    
                    # Prepare messages for clarification response
                    messages = [{"role": "system", "content": get_system_prompt()}]
                    messages.extend(relevant_history)
                    messages.append({"role": "user", "content": f"Query: {question}\nLanguage: {language}\n\nI searched our MBZUAI knowledge base but couldn't find relevant documents to answer this question. Please provide an intelligent clarification response that explains what I searched for and asks for specific details to help the user refine their question."})
                    
                    try:
                        completion = await openai_client.chat.completions.create(
                            model=OPENAI_RESPONSE_MODEL,
                            messages=messages,
                            reasoning_effort=OPENAI_RESPONSE_REASONING_EFFORT,
                            max_completion_tokens=OPENAI_CLARIFICATION_MAX_COMPLETION_TOKENS,
                            timeout=OPENAI_TIMEOUT,
                        )
                        clarification_response = completion.choices[0].message.content
                        
                        # Generate a unique string ID for this chat
                        chat_str_id = str(uuid.uuid4())
                        
                        # Send the clarification response
                        await safe_send(websocket, {
                            "response": clarification_response,
                            "sources": [],
                            "id": chat_str_id
                        })
                        
                        # Save the clarification response to the database
                        try:
                            chat_id = await ChatRepository.save_chat(app.state.pool, question, clarification_response, [], chat_str_id, RAG_APP_NAME)
                            logger.info(f"Clarification response saved to database with numeric ID: {chat_id}, string ID: {chat_str_id}")
                        except Exception as e:
                            logger.exception(f"Failed to save clarification response to database: {e}")
                        
                        return
                    except Exception as e:
                        logger.exception("Error generating clarification response:")
                        await safe_send(websocket, {"status": "error", "message": "I couldn't find relevant information and had trouble generating a clarification response. Please try rephrasing your question with more specific details."})
                        return

            except Exception as e:
                logger.exception("Failed to fetch documents from Pinecone.")
                await safe_send(websocket, {"status": "error", "message": "I’m having trouble retrieving information right now. Please try again later."})
                return

            end_time_retrieval = time.time()
            retrieval_time = end_time_retrieval - start_time_retrieval
            logger.info(f"[PERF] Initial retrieval took {retrieval_time:.4f} seconds.")
            logger.info(f"Retrieved {len(docs)} documents initially for query: {rewritten_queries}")

            # Log the sources of the initial documents for debugging
            if docs:
                logger.info("--- Documents After Deduplication (Before Reranking) ---")
                for i, doc in enumerate(docs):
                    source = doc.metadata.get('page_source', 'N/A')
                    title = doc.metadata.get('title', 'N/A')
                    logger.info(f"Doc {i+1}: Source: {source}, Title: {title}")
                logger.info("-------------------------------------------------------")

            await safe_send(websocket, {"status": "reranking", "message": f"Found {len(docs)} documents. Ranking them by relevance..."})

            # Rerank documents
            if docs:
                is_time_sensitive = agent_result.get("is_time_sensitive", False)
                ranked_docs = await rerank_docs(natural_language_query, docs, is_time_sensitive=is_time_sensitive)
                logger.info(f"Reranked and received {len(ranked_docs)} documents from rerank_docs.")
            else:
                ranked_docs = [] # No docs to rerank
            await safe_send(websocket, {"status": "reranked", "message": f"Selected top {len(ranked_docs)} most relevant documents."})

            # Log the sources of the reranked documents for debugging
            if ranked_docs:
                logger.info("--- Documents Provided to Response Generation ---")
                for i, doc in enumerate(ranked_docs):
                    source = doc.metadata.get('page_source', 'N/A')
                    title = doc.metadata.get('title', 'N/A')
                    logger.info(f"Doc {i+1}: Source: {source}, Title: {title}")
                logger.info("-------------------------------------------------")

            await safe_send(websocket, {"status": "generating", "message": "Generating a detailed answer based on the top documents..."})

            # Prepare the conversation messages
            messages = [{"role": "system", "content": get_system_prompt()}]
            messages.extend(relevant_history)  # Use only the relevant history
            messages.append({"role": "user", "content": format_query(question, language, ranked_docs)})

            complete_answer = ""
            chunk_buffer = ""
            isResponseAvailable = True

            # Generate and stream the chat response
            try:
                completion = await openai_client.chat.completions.create(
                    model=OPENAI_RESPONSE_MODEL,
                    messages=messages,
                    reasoning_effort=OPENAI_RESPONSE_REASONING_EFFORT,
                    max_completion_tokens=OPENAI_RESPONSE_MAX_COMPLETION_TOKENS,
                    stream=True,
                    timeout=OPENAI_TIMEOUT,
                )
                # Inform the client that streaming has started
                await safe_send(websocket, {"status": "streaming", "message": "Composing your answer..."})
                async for chunk in completion:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        # Only stop on specific patterns that clearly indicate the LLM wants to stop
                        # Check for the exact phrases from the system prompt that should trigger stopping
                        if ("🛑" in delta_content and 
                            ("out of my scope" in delta_content.lower() or 
                             "out of scope" in delta_content.lower() or
                             "not contain relevant information" in delta_content.lower())):
                            logger.info(f"Stopping response generation due to scope violation pattern: {delta_content}")
                            isResponseAvailable = False
                            break
                        complete_answer += delta_content
                        # Remove inline citation markers from the streamed chunk before sending
                        cleaned_content = re.sub(r'\[\d+\]', '', delta_content)
                        chunk_buffer += cleaned_content
                        if len(chunk_buffer) >= 1:
                            await safe_send(websocket, {"response": chunk_buffer})
                            chunk_buffer = ""
                if chunk_buffer:
                    await safe_send(websocket, {"response": chunk_buffer})
            except Exception as e:
                error_details = str(e)
                error_type = type(e).__name__
                logger.exception(f"Error during streaming: {error_details}")
                
                # Provide more specific error messages based on error type
                if "rate limit" in error_details.lower():
                    error_message = "I’m receiving too many requests right now. Please try again shortly."
                    error_code = "rate_limit_exceeded"
                elif "timeout" in error_details.lower():
                    error_message = "This is taking longer than expected. Please try again with a simpler or more specific question."
                    error_code = "request_timeout"
                else:
                    error_message = "I ran into an issue while generating a response. Please try again later."
                    error_code = "generation_error"
                
                await safe_send(websocket, {
                    "response": error_message,
                    "error": error_code,
                    "error_details": error_details,
                    "status": "error",
                    "sources": []
                })
                continue

            # If the response indicates no answer available, send a clear message and stop.
            if not isResponseAvailable:
                await safe_send(websocket, {
                    "response": "I couldn’t generate a confident answer right now. Try rephrasing with more details (e.g., program, year, topic) or ask a simpler version.",
                    "sources": []
                })
                continue

            # Process citations (run sync function in thread)
            await safe_send(websocket, {"status": "finalizing", "message": "Finalizing answer and formatting citations..."})
            updated_answer, citations = await asyncio.to_thread(
                process_citations, complete_answer, ranked_docs
            )
            logger.info(f"Citations after processing (WS path): {len(citations)} items")

            # Generate a unique string ID for this chat
            chat_str_id = str(uuid.uuid4())

            # Send the response to the client with the string ID immediately
            await safe_send(websocket, {
                "response": updated_answer,
                "sources": citations,
                "id": chat_str_id  # Include the string ID with the response
            })
            # Signal completion
            await safe_send(websocket, {"status": "done", "message": "Answer ready."})

            # Save chat to DB asynchronously using await
            try:
                logger.info(f"Saving chat with {len(citations)} citations (WS path)")
                # Await the async save_chat method
                chat_id = await ChatRepository.save_chat(app.state.pool, question, updated_answer, citations, chat_str_id, RAG_APP_NAME)
                if chat_id:
                    logger.info(f"Chat saved to database with numeric ID: {chat_id}, string ID: {chat_str_id}, and app: {RAG_APP_NAME}")
                else:
                    logger.error(f"Failed to save chat (string ID: {chat_str_id}) to database.")
            except Exception as db_err:
                logger.exception(f"Database error saving chat (string ID: {chat_str_id}): {db_err}")

        except WebSocketDisconnect:
            logger.info("Client disconnected")
            break
        except Exception as e:
            error_details = str(e)
            error_type = type(e).__name__
            logger.exception(f"Unexpected error in websocket endpoint: {error_type} - {error_details}")
            
            try:
                # Provide a more detailed error message with a unique error ID for tracking
                error_id = f"err-{int(time.time())}"
                error_message = f"An unexpected error occurred (ID: {error_id}). Our team has been notified and is working to resolve it."
                
                await safe_send(websocket, {
                    "response": error_message,
                    "error": "unexpected_error",
                    "error_id": error_id,
                    "sources": []
                })
                
                # Log the error with the error ID for easier tracking
                logger.error(f"Error ID {error_id}: {error_type} - {error_details}")
            except Exception as send_error:
                logger.info(f"Could not send error message as websocket appears to be closed: {str(send_error)}")
            break

# ------------------------------------------------------------------------------
# HTTP endpoint for Telegram chat
# ------------------------------------------------------------------------------
@app.post("/telegram-chat")
async def telegram_chat(chat_request: ChatRequest, request: Request):
    question = chat_request.question
    language = chat_request.language
    previous_chats = chat_request.previous_chats
    probe_mode = _is_health_probe(request.headers)

    agent_result = await query_rewriting_agent(question, language, previous_chats)

    if agent_result["action"] in ["respond", "identity"]:
        if probe_mode:
            return _probe_error_response(
                "probe_bypassed_generation",
                f"Probe did not exercise retrieval/generation path (action={agent_result['action']}).",
            )
        direct_response = agent_result.get("response", "I can only answer MBZUAI undergraduate related questions.")
        chat_str_id = str(uuid.uuid4())
        try:
            await ChatRepository.save_chat(request.app.state.pool, question, direct_response, [], chat_str_id, RAG_APP_NAME)
        except Exception as exc:
            logger.exception(f"Failed to persist direct response: {exc}")
        return {"response": direct_response, "sources": [], "id": chat_str_id}

    rewritten_queries = agent_result.get("rewritten_queries") or {
        "metadata_query": question,
        "natural_language_query": question,
    }

    try:
        docs = await fetch_balanced_documents(
            rewritten_queries=rewritten_queries,
            pinecone_summary_index=request.app.state.pinecone_summary_index,
            pinecone_text_index=request.app.state.pinecone_text_index,
            embed_model=request.app.state.embed_model,
            bm25_encoder=request.app.state.bm25_encoder,
        )
    except Exception as exc:
        logger.exception("Failed to fetch documents from Pinecone.")
        if probe_mode:
            return _probe_error_response("retrieval_failure", "Probe failed during retrieval.", details=str(exc))
        return JSONResponse(
            status_code=500,
            content={"response": "Something went wrong while retrieving documents. Please try again.", "sources": [], "id": str(uuid.uuid4())},
        )

    if not docs:
        if probe_mode:
            return _probe_error_response("documents_not_found", "Probe failed because retrieval returned no documents.")
        return JSONResponse(
            status_code=404,
            content={"response": "No relevant information found to answer your question.", "sources": [], "id": str(uuid.uuid4())},
        )

    ranked_docs = await rerank_docs(
        rewritten_queries.get("natural_language_query", question),
        docs,
        is_time_sensitive=agent_result.get("is_time_sensitive", False),
    )
    if not ranked_docs:
        if probe_mode:
            return _probe_error_response("reranker_failure", "Probe failed because reranking returned no documents.")
        return JSONResponse(
            status_code=500,
            content={"response": "No relevant information found to answer your question.", "sources": [], "id": str(uuid.uuid4())},
        )

    relevant_history = previous_chats[-10:] if previous_chats else []
    messages = [{"role": "system", "content": get_system_prompt()}]
    messages.extend(relevant_history)
    messages.append({"role": "user", "content": format_query(question, language, ranked_docs)})

    try:
        completion = await openai_client.chat.completions.create(
            model=OPENAI_RESPONSE_MODEL,
            messages=messages,
            reasoning_effort=OPENAI_RESPONSE_REASONING_EFFORT,
            max_completion_tokens=OPENAI_RESPONSE_MAX_COMPLETION_TOKENS,
            stream=False,
            timeout=OPENAI_TIMEOUT,
        )
    except Exception as exc:
        logger.exception("Error during response generation.")
        if probe_mode:
            return _probe_error_response("generation_failure", "Probe failed during response generation.", details=str(exc))
        return JSONResponse(
            status_code=500,
            content={"response": "Response generation failed. Please try again later.", "sources": [], "id": str(uuid.uuid4())},
        )

    complete_answer = _extract_completion_text(completion)
    if not complete_answer or "🛑" in complete_answer:
        if probe_mode:
            return _probe_error_response("no_generation_output", "Probe failed because generation returned no usable output.")
        return JSONResponse(
            status_code=500,
            content={"response": "I couldn’t generate a confident answer right now. Try rephrasing your question.", "sources": [], "id": str(uuid.uuid4())},
        )

    updated_answer, citations = await asyncio.to_thread(process_citations, complete_answer, ranked_docs)
    chat_str_id = str(uuid.uuid4())
    try:
        await ChatRepository.save_chat(request.app.state.pool, question, updated_answer, citations, chat_str_id, RAG_APP_NAME)
    except Exception as exc:
        logger.exception(f"Failed to persist telegram response: {exc}")

    response_payload = {"response": updated_answer, "sources": citations, "id": chat_str_id}
    if probe_mode:
        response_payload["status"] = "healthy"
        response_payload["probe"] = True
    return response_payload

# ------------------------------------------------------------------------------
# Health check endpoint
# ------------------------------------------------------------------------------
@app.get("/", response_class=JSONResponse)
async def root():
    return JSONResponse(content={"status": "working"})


@app.get("/api", response_class=JSONResponse)
async def api_root():
    return JSONResponse(content={"message": "API is working"})

@app.get("/health")
async def health(request: Request):
    # Add a database connection check
    db_status = "connected"
    db_message = "Database connection successful."
    pool = request.app.state.pool
    if not pool or getattr(pool, "closed", False):
         db_status = "disconnected"
         db_message = "Database pool is closed or not initialized."
    else:
        try:
            # Try a simple query
            async with pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
        except Exception as db_err:
            logger.error(f"Database health check failed: {db_err}")
            db_status = "error"
            db_message = f"Database connection error: {str(db_err)[:100]}..."
            
    try:
        # Check if components are loaded in app.state
        if not hasattr(request.app.state, 'embed_model') or not request.app.state.embed_model:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Embedding model not initialized"}
            )
        
        if not hasattr(request.app.state, 'pinecone_summary_index') or not hasattr(request.app.state, 'pinecone_text_index'):
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Pinecone indexes not initialized"}
            )
            
        # Return success if all checks pass
        return JSONResponse(
            content={
                "status": "healthy",
                "message": "API is operational",
                "components": {
                    "embedding_model": "initialized",
                    "pinecone": "connected",
                    "database": db_status,
                    "openai_models": {
                        "response": OPENAI_RESPONSE_MODEL,
                        "query_rewriter": OPENAI_QUERY_REWRITER_MODEL,
                        "reranker": OPENAI_RERANKER_MODEL,
                    }
                }
            },
            media_type="application/json"
        )
    except Exception as e:
        logger.exception("Health check failed:")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/health/generation")
async def health_generation(request: Request):
    checks = {
        "openai": {"status": "healthy", "model": OPENAI_RESPONSE_MODEL},
        "generation": {"status": "degraded", "error": "probe not executed", "model": OPENAI_RESPONSE_MODEL},
    }

    pool = request.app.state.pool
    if not pool or getattr(pool, "closed", False):
        checks["database"] = {"status": "unhealthy", "error": "Database pool is closed or not initialized"}
    else:
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            checks["database"] = {"status": "healthy"}
        except Exception as exc:
            checks["database"] = {"status": "unhealthy", "error": str(exc)}

    if not hasattr(request.app.state, "embed_model") or not request.app.state.embed_model:
        checks["embedding_model"] = {"status": "unhealthy", "error": "Embedding model not initialized"}
    else:
        checks["embedding_model"] = {"status": "healthy"}

    if not hasattr(request.app.state, "pinecone_summary_index") or not hasattr(request.app.state, "pinecone_text_index"):
        checks["vector_store"] = {"status": "unhealthy", "error": "Pinecone indexes not initialized"}
    else:
        checks["vector_store"] = {"status": "healthy"}

    preflight_statuses = [checks[k]["status"] for k in ("database", "embedding_model", "vector_store")]
    if any(status == "unhealthy" for status in preflight_statuses):
        payload = {
            "status": "unhealthy",
            "service": os.getenv("SERVICE_IDENTIFIER", "mbzuai-ug"),
            "timestamp": _utc_timestamp(),
            "probe": True,
            "checks": checks,
        }
        return JSONResponse(status_code=503, content=payload)

    try:
        completion = await openai_client.chat.completions.create(
            model=OPENAI_RESPONSE_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            reasoning_effort=OPENAI_RESPONSE_REASONING_EFFORT,
            max_completion_tokens=32,
            stream=False,
            timeout=min(OPENAI_TIMEOUT, 10),
        )
        generated_text = _extract_completion_text(completion)
    except Exception as exc:
        checks["openai"] = {"status": "unhealthy", "error": str(exc), "model": OPENAI_RESPONSE_MODEL}
        checks["generation"] = {"status": "unhealthy", "error": str(exc), "model": OPENAI_RESPONSE_MODEL}
        payload = {
            "status": "unhealthy",
            "service": os.getenv("SERVICE_IDENTIFIER", "mbzuai-ug"),
            "timestamp": _utc_timestamp(),
            "probe": True,
            "checks": checks,
        }
        return JSONResponse(status_code=503, content=payload)

    if not generated_text:
        checks["generation"] = {"status": "unhealthy", "error": "generation returned no usable text", "model": OPENAI_RESPONSE_MODEL}
        payload = {
            "status": "unhealthy",
            "service": os.getenv("SERVICE_IDENTIFIER", "mbzuai-ug"),
            "timestamp": _utc_timestamp(),
            "probe": True,
            "checks": checks,
        }
        return JSONResponse(status_code=503, content=payload)

    checks["generation"] = {"status": "healthy", "model": OPENAI_RESPONSE_MODEL, "preview": generated_text[:160]}
    payload = {
        "status": "healthy",
        "service": os.getenv("SERVICE_IDENTIFIER", "mbzuai-ug"),
        "timestamp": _utc_timestamp(),
        "probe": True,
        "checks": checks,
    }
    return JSONResponse(status_code=200, content=payload)
