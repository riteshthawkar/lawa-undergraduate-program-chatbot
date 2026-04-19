import asyncio
import re
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
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
    GENERATION_MODEL,
    QUERY_REWRITE_MODEL,
    MAX_GENERATION_TOKENS,
    EMBEDDING_MODEL_NAME,
    SERVICE_IDENTIFIER,
    SERVICE_DISPLAY_NAME,
    SERVICE_TYPE,
    SERVICE_ENVIRONMENT,
    HEALTH_PROBE_QUERY,
    HEALTH_PROBE_LANGUAGE,
    HEALTH_PROBE_TOP_DOCS,
    RELEASE_VERSION,
    RELEASE_COMMIT_SHA,
    RELEASE_DEPLOYED_AT,
    SERVICE_OWNER,
    RUNBOOK_URL,
    DASHBOARD_SERVICE_ID,
    REPOSITORY_URL,
    PUBLIC_BASE_URL,
)
from modules.schemas import ChatRequest
from modules.utils import safe_send, format_query
from modules.citations import process_citations
from modules.retrieval import initialize_retrieval_components, rerank_docs, fetch_balanced_documents
from pinecone import Pinecone
from modules.query_rewriting import query_rewriting_agent, openai_client
from modules.monitoring import (
    CONTRACT_VERSION,
    HEALTHY,
    UNHEALTHY,
    build_contract_payload,
    health_status_code,
)

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
                            model=GENERATION_MODEL,
                            messages=messages,
                            temperature=0,
                            max_tokens=min(512, MAX_GENERATION_TOKENS),
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
                    model=GENERATION_MODEL,
                    messages=messages,
                    temperature=0,
                    max_tokens=MAX_GENERATION_TOKENS,
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
async def telegram_chat(chat_request: ChatRequest, background_tasks: BackgroundTasks):
    # Extract the question and language from the validated request body.
    logger.info(f"Received telegram chat request: {chat_request}")
    
    question = chat_request.question
    language = chat_request.language
    previous_chats = chat_request.previous_chats

    # Apply query rewriting agent to analyze and possibly rewrite the query
    agent_result = await query_rewriting_agent(question=question, language=language, message_history=previous_chats)
    
    # Handle direct responses (out of scope queries)
    if agent_result["action"] in ["respond"]:
        return {
            "response": agent_result["response"],
            "sources": []
        }
        
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
    
    start_time_retrieval = time.time()

    # Step 3: Fetch initial documents from Pinecone
    logger.info("Step 3: Fetching documents from Pinecone...")
    try:
        docs = await fetch_balanced_documents(
            rewritten_queries=agent_result.get("rewritten_queries", {}),
            pinecone_summary_index=app.state.pinecone_summary_index,
            pinecone_text_index=app.state.pinecone_text_index,
            embed_model=app.state.embed_model,
            bm25_encoder=app.state.bm25_encoder
        )
        if not docs:
            logger.warning("No documents found in Pinecone for the query.")
            # Use main response generation agent to provide intelligent clarification
            try:
                # Prepare messages for clarification response
                messages = [{"role": "system", "content": get_system_prompt()}]
                messages.extend(relevant_history)
                messages.append({"role": "user", "content": f"Query: {question}\nLanguage: {language}\n\nI searched our MBZUAI knowledge base but couldn't find relevant documents to answer this question. Please provide an intelligent clarification response that explains what I searched for and asks for specific details to help the user refine their question."})
                
                completion = await openai_client.chat.completions.create(
                    model=GENERATION_MODEL,
                    messages=messages,
                    temperature=0,
                    max_tokens=min(512, MAX_GENERATION_TOKENS),
                    timeout=OPENAI_TIMEOUT,
                )
                clarification_response = completion.choices[0].message.content
                
                # Generate a unique string ID for this chat
                chat_str_id = str(uuid.uuid4())
                
                # Save the clarification response to the database
                try:
                    asyncio.create_task(
                        ChatRepository.save_chat(app.state.pool, question, clarification_response, [], chat_str_id, RAG_APP_NAME)
                    )
                except Exception as e:
                    logger.exception(f"Failed to schedule clarification response save: {e}")
                
                return {"response": clarification_response, "sources": [], "id": chat_str_id}
            except Exception as e:
                logger.exception("Error generating clarification response:")
                return {"response": "I couldn't find relevant information and had trouble generating a clarification response. Please try rephrasing your question with more specific details.", "sources": [], "id": str(uuid.uuid4())}
    except Exception as e:
        logger.exception("Failed to fetch documents from Pinecone.")
        return JSONResponse(
            status_code=500,
            content={"response": "Something went wrong. Please try again.", "sources": [], "id": str(uuid.uuid4())}
        )

    end_time_retrieval = time.time()
    retrieval_time = end_time_retrieval - start_time_retrieval
    logger.info(f"[PERF] Initial retrieval took {retrieval_time:.4f} seconds.")
    logger.info(f"Retrieved {len(docs)} documents initially for query: {natural_language_query}")

    # Rerank documents
    if docs:
        ranked_docs = await rerank_docs(natural_language_query, docs)
        logger.info(f"Reranked and received {len(ranked_docs)} documents from rerank_docs.")
    else:
        ranked_docs = [] # No docs to rerank

    if not ranked_docs:
        return {
            "response": "No relevant information found to answer your question.",
            "sources": []
        }

    # Prepare the conversation messages.
    messages = [{"role": "system", "content": get_system_prompt()}]
    messages.extend(relevant_history)  # Use only the relevant history
    messages.append({"role": "user", "content": format_query(question, language, ranked_docs)})

    complete_answer = ""
    isResponseAvailable = True

    # Generate and stream the chat response.
    try:
        completion = await openai_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=MAX_GENERATION_TOKENS,
            stream=True,
            timeout=OPENAI_TIMEOUT,
        )
        cleaned_answer = ""
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
                # Remove inline citation markers from the streamed chunk and store it
                cleaned_content = re.sub(r'\[\d+\]', '', delta_content)
                cleaned_answer += cleaned_content
    except Exception as e:
        logger.exception("Error during streaming response:")
        return {
            "response": "Response generation failed. Please try again later.",
            "sources": []
        }

    # If the initial response indicates no answer, return a clear failure message.
    if not isResponseAvailable:
        return {
            "response": "I couldn’t generate a confident answer right now. Try rephrasing with more details (e.g., program, year, topic) or ask a simpler version.",
            "sources": []
        }

    # Process citations after streaming completes (run sync function in thread)
    if isResponseAvailable:
        updated_answer, citations = await asyncio.to_thread(
            process_citations, complete_answer, ranked_docs
        )
        logger.info(f"Citations after processing (HTTP path): {len(citations)} items")
    else:
        # Handle case where no response was generated (e.g., fallback search failed)
        updated_answer, citations = complete_answer, []

    # Fallback extraction if citations empty but links present (HTTP path)
    if not citations:
        try:
            matches = re.findall(r"\[(\d+)\]\(([^)]+)\)", updated_answer)
            if matches:
                seen = set()
                extracted = []
                for num, url in matches:
                    key = (num, url)
                    if key in seen:
                        continue
                    seen.add(key)
                    extracted.append({"url": url, "cite_num": str(num)})
                citations = sorted(extracted, key=lambda x: int(x["cite_num"]))
                logger.info(f"Extracted {len(citations)} citations from updated answer as fallback (HTTP path).")
        except Exception as _e:
            logger.warning(f"Citation fallback extraction failed (HTTP path): {_e}")

    # Generate a unique string ID for this chat
    chat_str_id = str(uuid.uuid4())
    
    # Save chat asynchronously without blocking response
    try:
        logger.info(f"Saving chat with {len(citations)} citations (HTTP path)")
        asyncio.create_task(
            ChatRepository.save_chat(app.state.pool, question, updated_answer, citations, chat_str_id, RAG_APP_NAME)
        )
    except Exception as e:
        logger.exception(f"Failed to schedule chat save task: {e}")

    # Return the response immediately without waiting for database operation
    return {"response": updated_answer, "sources": citations, "id": chat_str_id}

# ------------------------------------------------------------------------------
# Health check endpoint
# ------------------------------------------------------------------------------
@app.get("/", response_class=JSONResponse)
async def root():
    return JSONResponse(content={"status": "working"})


@app.get("/api", response_class=JSONResponse)
async def api_root():
    return JSONResponse(content={"message": "API is working"})

def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _component_payload(status: str, *, error: Optional[str] = None, **details: Any) -> dict:
    payload = {"status": status}
    if error:
        payload["error"] = error
    for key, value in details.items():
        if value is not None:
            payload[key] = value
    return payload


def _aggregate_status(checks: dict[str, dict]) -> str:
    statuses = [check.get("status", "unhealthy") for check in checks.values()]
    if any(status == "unhealthy" for status in statuses):
        return "unhealthy"
    if any(status == "degraded" for status in statuses):
        return "degraded"
    return "healthy"


def _status_code_for(status: str) -> int:
    return 200 if status in {"healthy", "degraded"} else 503


def _release_metadata_contract() -> dict:
    release = {
        "version": RELEASE_VERSION or "unknown",
    }
    if RELEASE_COMMIT_SHA and RELEASE_COMMIT_SHA != "unknown":
        release["commitSha"] = RELEASE_COMMIT_SHA
    if RELEASE_DEPLOYED_AT and RELEASE_DEPLOYED_AT != "unknown":
        release["deployedAt"] = RELEASE_DEPLOYED_AT
    return release


def _operations_metadata() -> dict:
    metadata = {
        "owner": SERVICE_OWNER,
        "runbook_url": RUNBOOK_URL,
        "dashboard_service_id": DASHBOARD_SERVICE_ID,
        "repository_url": REPOSITORY_URL,
        "public_base_url": PUBLIC_BASE_URL,
    }
    return {key: value for key, value in metadata.items() if value}


def _build_contract_summary(status: str, endpoint: str) -> str:
    if status == HEALTHY:
        return f"{endpoint} checks passed"
    if status == "degraded":
        return f"{endpoint} checks degraded"
    if status == UNHEALTHY:
        return f"{endpoint} checks failed"
    return f"{endpoint} status unknown"


def _contract_response(
    *,
    endpoint_label: str,
    checks: dict[str, dict],
    status: Optional[str] = None,
    journey: Optional[dict] = None,
) -> JSONResponse:
    resolved_status = status or _aggregate_status(checks)
    payload = build_contract_payload(
        service_id=SERVICE_IDENTIFIER,
        service_name=SERVICE_DISPLAY_NAME,
        service_type=SERVICE_TYPE,
        environment=SERVICE_ENVIRONMENT,
        checks=checks,
        journey=journey,
        release=_release_metadata_contract(),
        operations=_operations_metadata(),
        summary=_build_contract_summary(resolved_status, endpoint_label),
        status=resolved_status,
    )
    return JSONResponse(status_code=health_status_code(resolved_status), content=payload)


async def _check_database(pool: Any) -> dict:
    if not pool or getattr(pool, "closed", False):
        return _component_payload("unhealthy", error="database pool is closed or not initialized")
    try:
        started = time.perf_counter()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return _component_payload("healthy", latency_ms=round((time.perf_counter() - started) * 1000, 2))
    except Exception as exc:
        logger.error(f"Database health check failed: {exc}")
        return _component_payload("unhealthy", error=str(exc))


def _check_embedding_model(app_state: Any) -> dict:
    embed_model = getattr(app_state, "embed_model", None)
    if not embed_model:
        return _component_payload("unhealthy", error="embedding model not initialized", model=EMBEDDING_MODEL_NAME)
    return _component_payload("healthy", model=EMBEDDING_MODEL_NAME)


def _check_openai_configuration() -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _component_payload("unhealthy", error="OPENAI_API_KEY is not configured")
    if not GENERATION_MODEL:
        return _component_payload("unhealthy", error="GENERATION_MODEL is not configured")
    return _component_payload(
        "healthy",
        generation_model=GENERATION_MODEL,
        query_rewrite_model=QUERY_REWRITE_MODEL,
    )


def _check_query_rewrite_configuration() -> dict:
    return _component_payload(
        "healthy",
        model=QUERY_REWRITE_MODEL,
    )


def _check_vector_store_presence(app_state: Any) -> dict:
    summary_index = getattr(app_state, "pinecone_summary_index", None)
    text_index = getattr(app_state, "pinecone_text_index", None)
    summary_index_name = os.getenv("PINECONE_SUMMARY_INDEX_NAME", "mbzuai-undergraduate-summary-only-index")
    text_index_name = os.getenv("PINECONE_TEXT_INDEX_NAME", "mbzuai-undergraduate-text-only-index")

    if not summary_index or not text_index:
        return _component_payload(
            "unhealthy",
            error="one or more Pinecone indexes are not initialized",
            summary_index=summary_index_name,
            text_index=text_index_name,
        )
    return _component_payload(
        "healthy",
        summary_index=summary_index_name,
        text_index=text_index_name,
    )


async def _probe_pinecone_index(index: Any, index_name: str) -> dict:
    if not index:
        return _component_payload("unhealthy", error=f"{index_name} is not initialized")
    describe_index_stats = getattr(index, "describe_index_stats", None)
    if not callable(describe_index_stats):
        return _component_payload("degraded", error=f"{index_name} does not expose describe_index_stats")
    try:
        stats = await asyncio.to_thread(describe_index_stats)
        namespace_count = None
        if isinstance(stats, dict):
            namespace_count = len(stats.get("namespaces", {}))
        return _component_payload("healthy", namespaces=namespace_count)
    except Exception as exc:
        logger.error(f"Pinecone health check failed for {index_name}: {exc}")
        return _component_payload("unhealthy", error=str(exc))


async def _check_vector_store(app_state: Any, *, detailed: bool) -> dict:
    base = _check_vector_store_presence(app_state)
    if not detailed or base["status"] == "unhealthy":
        return base

    summary_index_name = os.getenv("PINECONE_SUMMARY_INDEX_NAME", "mbzuai-undergraduate-summary-only-index")
    text_index_name = os.getenv("PINECONE_TEXT_INDEX_NAME", "mbzuai-undergraduate-text-only-index")
    summary_check, text_check = await asyncio.gather(
        _probe_pinecone_index(getattr(app_state, "pinecone_summary_index", None), summary_index_name),
        _probe_pinecone_index(getattr(app_state, "pinecone_text_index", None), text_index_name),
    )
    nested = {"summary": summary_check, "text": text_check}
    return _component_payload(
        _aggregate_status(nested),
        indexes=nested,
        summary_index=summary_index_name,
        text_index=text_index_name,
    )


async def _build_health_payload(request: Request, *, detailed: bool) -> dict:
    checks = {
        "database": await _check_database(getattr(request.app.state, "pool", None)),
        "embedding_model": _check_embedding_model(request.app.state),
        "vector_store": await _check_vector_store(request.app.state, detailed=detailed),
        "openai": _check_openai_configuration(),
        "query_rewrite": _check_query_rewrite_configuration(),
    }
    return {
        "status": _aggregate_status(checks),
        "checks": checks,
    }


async def _run_generation_probe(request: Request) -> dict:
    checks = {
        "database": await _check_database(getattr(request.app.state, "pool", None)),
        "embedding_model": _check_embedding_model(request.app.state),
        "vector_store": _check_vector_store_presence(request.app.state),
        "openai": _check_openai_configuration(),
        "query_rewrite": _component_payload("degraded", error="probe not executed"),
        "retrieval": _component_payload("degraded", error="probe not executed"),
        "generation": _component_payload("degraded", error="probe not executed", model=GENERATION_MODEL),
    }
    preflight = _aggregate_status(
        {
            "database": checks["database"],
            "embedding_model": checks["embedding_model"],
            "vector_store": checks["vector_store"],
            "openai": checks["openai"],
        }
    )
    if preflight == "unhealthy":
        return {"status": "unhealthy", "checks": checks}

    agent_result = await query_rewriting_agent(
        question=HEALTH_PROBE_QUERY,
        language=HEALTH_PROBE_LANGUAGE,
        message_history=[],
    )
    action = agent_result.get("action")
    checks["query_rewrite"] = _component_payload(
        "healthy" if action == "rewrite" else "unhealthy",
        model=QUERY_REWRITE_MODEL,
        action=action,
        error=None if action == "rewrite" else f"probe did not exercise retrieval path (action={action})",
    )
    if action != "rewrite":
        return {"status": "unhealthy", "checks": checks}

    rewritten_queries = agent_result.get("rewritten_queries") or {
        "metadata_query": HEALTH_PROBE_QUERY,
        "natural_language_query": HEALTH_PROBE_QUERY,
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
        checks["vector_store"] = _component_payload("unhealthy", error=str(exc))
        checks["retrieval"] = _component_payload("unhealthy", error=str(exc))
        return {"status": "unhealthy", "checks": checks}

    if not docs:
        checks["retrieval"] = _component_payload("unhealthy", error="probe returned no documents")
        return {"status": "unhealthy", "checks": checks}
    checks["retrieval"] = _component_payload("healthy", documents=len(docs))

    natural_language_query = rewritten_queries.get("natural_language_query", HEALTH_PROBE_QUERY)
    ranked_docs = await rerank_docs(natural_language_query, docs)
    if not ranked_docs:
        checks["retrieval"] = _component_payload("unhealthy", error="reranker returned no documents")
        return {"status": "unhealthy", "checks": checks}

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {
            "role": "user",
            "content": format_query(
                HEALTH_PROBE_QUERY,
                HEALTH_PROBE_LANGUAGE,
                ranked_docs[: max(1, HEALTH_PROBE_TOP_DOCS)],
            ),
        },
    ]
    try:
        completion = await openai_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=min(128, MAX_GENERATION_TOKENS),
            stream=False,
            timeout=OPENAI_TIMEOUT,
        )
        message = completion.choices[0].message.content if completion and completion.choices else ""
        generated_text = (message or "").strip()
    except Exception as exc:
        checks["openai"] = _component_payload(
            "unhealthy",
            error=str(exc),
            generation_model=GENERATION_MODEL,
            query_rewrite_model=QUERY_REWRITE_MODEL,
        )
        checks["generation"] = _component_payload("unhealthy", error=str(exc), model=GENERATION_MODEL)
        return {"status": "unhealthy", "checks": checks}

    if not generated_text:
        checks["generation"] = _component_payload(
            "unhealthy",
            error="generation returned no usable text",
            model=GENERATION_MODEL,
        )
        return {"status": "unhealthy", "checks": checks}

    checks["generation"] = _component_payload(
        "healthy",
        model=GENERATION_MODEL,
        preview=generated_text[:160],
    )
    return {"status": _aggregate_status(checks), "checks": checks}


@app.get("/health")
async def health(request: Request):
    return await health_ready(request)


@app.get("/health/live")
async def health_live(request: Request):
    checks = {
        "application": _component_payload("healthy"),
        "process": _component_payload("healthy", pid=os.getpid()),
        "contract": _component_payload("healthy", version=CONTRACT_VERSION),
    }
    return _contract_response(endpoint_label="live", checks=checks, status="healthy")


@app.get("/health/ready")
async def health_ready(request: Request):
    payload = await _build_health_payload(request, detailed=False)
    return _contract_response(
        endpoint_label="ready",
        checks=payload["checks"],
        status=payload["status"],
    )


@app.get("/health/detailed")
async def health_detailed(request: Request):
    payload = await _build_health_payload(request, detailed=True)
    return _contract_response(
        endpoint_label="detailed",
        checks=payload["checks"],
        status=payload["status"],
    )


@app.get("/health/generation")
async def health_generation(request: Request):
    payload = await _run_generation_probe(request)
    return JSONResponse(status_code=_status_code_for(payload["status"]), content=payload)


@app.get("/health/journey")
async def health_journey(request: Request):
    started = time.perf_counter()
    probe_payload = await _run_generation_probe(request)
    probe_status = probe_payload.get("status", "unhealthy")
    probe_checks = probe_payload.get("checks", {})
    duration_ms = int((time.perf_counter() - started) * 1000)

    generation_message = None
    if isinstance(probe_checks, dict):
        generation_message = probe_checks.get("generation", {}).get("error")
        if not generation_message:
            generation_message = probe_checks.get("generation", {}).get("preview")

    journey = {
        "name": "rag_generation",
        "status": probe_status if isinstance(probe_status, str) else "unhealthy",
        "probeModeSupported": True,
        "sideEffects": "none",
        "durationMs": duration_ms,
    }
    if generation_message:
        journey["message"] = str(generation_message)[:200]

    return _contract_response(
        endpoint_label="journey",
        checks=probe_checks if isinstance(probe_checks, dict) else {},
        status=probe_status if isinstance(probe_status, str) else "unhealthy",
        journey=journey,
    )
