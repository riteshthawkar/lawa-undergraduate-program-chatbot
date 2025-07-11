import asyncio
import re
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from langchain_huggingface import HuggingFaceEmbeddings

# Import modules
from modules.config import logger, validate_env_vars, get_system_prompt, RAG_APP_NAME
from modules.schemas import ChatRequest, CitationSource
from modules.utils import safe_send, format_query
from modules.citations import process_citations
from modules.retrieval import initialize_pinecone_clients, initialize_retrieval_components, rerank_docs, fetch_balanced_documents
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
    # Startup: Connect to DB and initialize
    logger.info("Application startup: Connecting to database...")
    # Call connect_db and store the returned pool
    pool = await connect_db()
    logger.info("Application startup: Initializing database schema...")
    # Pass the created pool to init_db
    await init_db(pool)

    # Assign the pool to app state
    if pool:
        app.state.pool = pool
        logger.info(f"Database pool assigned to app.state. Pool state: {app.state.pool}")
    else:
        logger.error("connect_db returned None. Cannot assign pool to app.state.")
        app.state.pool = None # Explicitly set to None if import failed

    yield # Application runs here

    # Shutdown: Disconnect from DB
    logger.info("Application shutdown: Disconnecting from database...")
    # Pass the pool from app state to disconnect_db
    await disconnect_db(app.state.pool)

# ------------------------------------------------------------------------------
# Initialize FastAPI app with CORS middleware and lifespan manager
# ------------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# CORS configuration with specific origins for better security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # Allow all origins
    allow_credentials=True,           # Allow credentials
    allow_methods=["*"],              # Allow all methods
    allow_headers=["*"],              # Allow all headers
)

# ------------------------------------------------------------------------------
# Setup static files and templates
# ------------------------------------------------------------------------------
# Get the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------
# Initialize retrieval components and Pinecone clients
# ------------------------------------------------------------------------------
embed_model, bm25_encoder = initialize_retrieval_components()
pinecone_summary_index, pinecone_text_index = initialize_pinecone_clients()
logger.info("Retrieval components and Pinecone clients initialized successfully.")

# ------------------------------------------------------------------------------
# Include database history router (endpoints within will need async updates too)
# ------------------------------------------------------------------------------
app.include_router(history_router, prefix="/api")

# ------------------------------------------------------------------------------
# WebSocket endpoint for chat functionality with improved error handling
# ------------------------------------------------------------------------------
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket, background_tasks: BackgroundTasks):
    await websocket.accept()
    logger.info("New client connected")
    
    while True:
        try:
            # Wait for client messages without a timeout
            data = await websocket.receive_json()
            
            # --- Feedback Handling --- 
            if "feedback" in data and "id" in data:
                try:
                    chat_id_str = data.get("id") 
                    feedback = data.get("feedback")
                    
                    if chat_id_str and feedback in ["like", "dislike"]:
                        logger.info(f"Received feedback '{feedback}' for chat ID: {chat_id_str}")
                        success = await ChatRepository.update_feedback(app.state.pool, chat_id_str, feedback)
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
            
            # Handle direct responses (out of scope, clarification requests, or identity questions)
            if agent_result["action"] in ["respond", "clarify", "identity"]:
                # Generate a unique string ID for this chat
                chat_str_id = str(uuid.uuid4())
                
                # Send the response to the client with the string ID
                await safe_send(websocket, {
                    "response": agent_result["response"], 
                    "sources": [],
                    "id": chat_str_id  # Include the string ID with the response
                })
                
                # Save the query-rewriting agent's response to the database using asyncio.create_task
                asyncio.create_task(
                    ChatRepository.save_chat(app.state.pool, question, agent_result["response"], [], chat_str_id, RAG_APP_NAME)
                )
                
                continue
                
            # Extract results from the query rewriting agent
            metadata_query = agent_result.get("metadata_query", question)
            natural_language_query = agent_result.get("natural_language_query", question)
            is_time_sensitive = agent_result.get("is_time_sensitive", False)

            # Filter previous chat messages based on relevance
            relevant_history = []
            if "relevant_history_indices" in agent_result and previous_chats:
                indices = agent_result["relevant_history_indices"]
                indices_to_include = set()
                for idx in indices:
                    if 0 <= idx < len(previous_chats):
                        indices_to_include.add(idx)
                        if idx + 1 < len(previous_chats) and previous_chats[idx]["role"] == "user" and previous_chats[idx + 1]["role"] == "assistant":
                            indices_to_include.add(idx + 1)
                sorted_indices = sorted(indices_to_include)
                relevant_history = [previous_chats[i] for i in sorted_indices]
                if len(relevant_history) < len(previous_chats):
                    logger.info(f"Filtered message history from {len(previous_chats)} to {len(relevant_history)} relevant messages")
            else:
                relevant_history = []

            await safe_send(websocket, {"type": "status", "message": "Starting retrieval..."})
            start_time_retrieval = time.time()

            # Perform balanced retrieval using the dual-query system
            docs_as_dicts = await fetch_balanced_documents(
                metadata_query=metadata_query,
                natural_language_query=natural_language_query,
                pinecone_summary_index=pinecone_summary_index,
                pinecone_text_index=pinecone_text_index,
                embed_model=embed_model,
                bm25_encoder=bm25_encoder
            )

            if not docs_as_dicts:
                logger.info(f"No documents found by Pinecone for query: '{question}'. Falling back to Tavily.")
                await safe_send(websocket, {"type": "status", "message": "No initial documents found, trying web search..."})
                tavily_docs = await tavily_search(natural_language_query) # Use NL query for web search
                if tavily_docs:
                    docs_as_dicts = tavily_docs
                    await safe_send(websocket, {"type": "status", "message": f"Found {len(docs_as_dicts)} documents from web search."})
                else:
                    logger.warning(f"Tavily search also returned no documents for query: '{natural_language_query}'")
                    await safe_send(websocket, {"type": "status", "message": "No documents found from any source."})
                    await safe_send(websocket, {"response": "Sorry, I couldn't find any relevant documents for your query.", "sources": [], "id": str(uuid.uuid4())})
                    continue

            end_time_retrieval = time.time()
            retrieval_time = end_time_retrieval - start_time_retrieval
            logger.info(f"[PERF] Initial retrieval took {retrieval_time:.4f} seconds.")
            logger.info(f"Retrieved {len(docs_as_dicts)} documents initially for query: '{question}'")
            await safe_send(websocket, {"type": "status", "message": f"Retrieved {len(docs_as_dicts)} documents, starting reranking..."})

            # Rerank documents using the original query and time-sensitivity flag
            if docs_as_dicts:
                ranked_docs = await rerank_docs(question, docs_as_dicts, is_time_sensitive)
                logger.info(f"Reranked and received {len(ranked_docs)} documents from rerank_docs.")
            else:
                ranked_docs = []

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
                    model="chatgpt-4o-latest",
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=1024,
                    stream=True
                )
                async for chunk in completion:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        if "🛑" in delta_content:
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
                    error_message = "The system is currently experiencing high demand. Please try again in a moment."
                    error_code = "rate_limit_exceeded"
                elif "timeout" in error_details.lower():
                    error_message = "The request timed out. Please try again with a simpler question."
                    error_code = "request_timeout"
                else:
                    error_message = "There was an issue generating a response. Please try again later."
                    error_code = "generation_error"
                
                await safe_send(websocket, {
                    "response": error_message,
                    "error": error_code,
                    "error_details": error_details,
                    "sources": []
                })
                continue

            # If the response indicates no answer available, perform fallback search and reattempt generation.
            if not isResponseAvailable:
                ranked_docs = await tavily_search(question)
                messages[-1] = {"role": "user", "content": format_query(question, language, ranked_docs)}
                try:
                    completion = await openai_client.chat.completions.create(
                        model="chatgpt-4o-latest",
                        messages=messages,
                        temperature=0.2,
                        max_completion_tokens=1024,
                        stream=True
                    )
                    async for chunk in completion:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
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
                    logger.exception("Error during fallback streaming:")
                    await safe_send(websocket, {"response": "Fallback response generation failed.", "sources": []})
                    continue

            # Process citations (run sync function in thread)
            updated_answer, citations = await asyncio.to_thread(
                process_citations, complete_answer, ranked_docs
            )

            # Generate a unique string ID for this chat
            chat_str_id = str(uuid.uuid4())

            # Send the response to the client with the string ID immediately
            await safe_send(websocket, {
                "response": updated_answer,
                "sources": citations,
                "id": chat_str_id  # Include the string ID with the response
            })

            # Save chat to DB using asyncio.create_task
            save_task = asyncio.create_task(
                ChatRepository.save_chat(app.state.pool, question, updated_answer, citations, chat_str_id, RAG_APP_NAME)
            )
            
            # Add logging to track save operation
            save_task.add_done_callback(
                lambda t: logger.info(f"Chat save task completed with result: {t.result() if not t.exception() else f'Error: {t.exception()}'}")
            )

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
    agent_result = await query_rewriting_agent(question, language, previous_chats)
    
    # Handle direct responses (out of scope or clarification requests)
    if agent_result["action"] in ["respond", "clarify"]:
        return {
            "response": agent_result["response"],
            "sources": []
        }
        
    # Use the rewritten query for retrieval if available
    rewritten_query = agent_result.get("rewritten_query", question)
    
    # Filter previous chat messages based on relevance
    relevant_history = []
    if "relevant_history_indices" in agent_result and previous_chats:
        indices = agent_result["relevant_history_indices"]
        
        # Create a set to track which indices to include (including assistants' responses)
        indices_to_include = set()
        
        # Include each relevant message index
        for idx in indices:
            if 0 <= idx < len(previous_chats):
                indices_to_include.add(idx)
                # If this is a user message and there's an assistant response right after,
                # include the assistant's response too
                if idx + 1 < len(previous_chats) and previous_chats[idx]["role"] == "user" and previous_chats[idx + 1]["role"] == "assistant":
                    indices_to_include.add(idx + 1)
        
        # Sort the indices to maintain conversation order
        sorted_indices = sorted(indices_to_include)
        
        # Get relevant messages in order
        relevant_history = [previous_chats[i] for i in sorted_indices]
        
        # Log the filtering of message history
        if len(relevant_history) < len(previous_chats):
            logger.info(f"Filtered message history from {len(previous_chats)} to {len(relevant_history)} relevant messages")
    else:
        # If no relevance info or no previous chats, use empty history
        relevant_history = []
    
    start_time_retrieval = time.time()

    # Perform balanced retrieval using the new function
    docs_for_rerank = await fetch_balanced_documents(
        query=rewritten_query, 
        pc_client=pc, 
        embed_model=embed_model, 
        bm25_encoder=bm25_encoder,
        num_webpages=20, # Default, can be adjusted
        num_pdfs=20      # Default, can be adjusted
    )

    if not docs_for_rerank:
        logger.info(f"No documents found by Pinecone for query: {rewritten_query}. Falling back to Tavily.")
        tavily_docs = await tavily_search(rewritten_query)
        if tavily_docs:
            docs_for_rerank = tavily_docs
        else:
            logger.warning(f"Tavily search also returned no documents for query: {rewritten_query}")
            # Handle case where no documents are found from any source
            # This might involve returning an empty list of sources or a specific message
            # For now, we'll proceed with an empty docs_for_rerank, which rerank_docs should handle
            pass # rerank_docs will receive an empty list

    end_time_retrieval = time.time()
    retrieval_time = end_time_retrieval - start_time_retrieval
    logger.info(f"[PERF] Initial retrieval took {retrieval_time:.4f} seconds.")
    logger.info(f"Retrieved {len(docs_for_rerank)} documents initially for query: {rewritten_query}")

    # Rerank documents
    if docs_for_rerank:
        ranked_docs = await rerank_docs(rewritten_query, docs_for_rerank)
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
            model="chatgpt-4o-latest",
            messages=messages,
            temperature=0,
            max_completion_tokens=1024,
            stream=True
        )
        cleaned_answer = ""
        async for chunk in completion:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                if "🛑" in delta_content:
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

    # If the initial response indicates no answer, perform a fallback search.
    if not isResponseAvailable:
        try:
            ranked_docs = await tavily_search(question)
            messages[-1] = {"role": "user", "content": format_query(question, language, ranked_docs)}
            # Reset answers for fallback
            complete_answer = ""
            cleaned_answer = ""
            completion = await openai_client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=messages,
                temperature=0.2,
                max_completion_tokens=1024,
                stream=True
            )
            async for chunk in completion:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    complete_answer += delta_content
                    cleaned_content = re.sub(r'\[\d+\]', '', delta_content)
                    cleaned_answer += cleaned_content
        except Exception as e:
            logger.exception("Error during fallback streaming:")
            return {
                "response": "Fallback response generation failed.",
                "sources": []
            }

    # Process citations after streaming completes (run sync function in thread)
    if isResponseAvailable:
        updated_answer, citations = await asyncio.to_thread(
            process_citations, complete_answer, ranked_docs
        )
    else:
        # Handle case where no response was generated (e.g., fallback search failed)
        updated_answer, citations = complete_answer, []

    # Generate a unique string ID for this chat
    chat_str_id = str(uuid.uuid4())
    
    # Define an ASYNC function to save chat in the background
    async def save_chat_to_db(pool, q, answer, cite, str_id):
        try:
            # Await the async repository method
            chat_id = await ChatRepository.save_chat(pool, q, answer, cite, str_id, RAG_APP_NAME)
            if chat_id:
                logger.info(f"Chat saved to database with numeric ID: {chat_id} and string ID: {str_id}")
            else:
                logger.error(f"Failed to save chat to database (String ID: {str_id})")
            return chat_id
        except Exception as e:
            logger.exception(f"Failed to save chat to database (String ID: {str_id}): {e}")
            return None
    
    # Add the save operation to background tasks
    # This will run after the response is sent to the client
    # Define an async function to save chat in the background, now including RAG_APP_NAME
    async def save_chat_to_db(pool, q, answer, cite, str_id, app_name):
        try:
            # Await the async repository method, passing the app name
            chat_id = await ChatRepository.save_chat(pool, q, answer, cite, str_id, app_name)
            if chat_id:
                logger.info(f"Chat saved to database with numeric ID: {chat_id} and string ID: {str_id} from app: {app_name}")
            else:
                logger.error(f"Failed to save chat to database (String ID: {str_id})")
            return chat_id
        except Exception as e:
            logger.exception(f"Failed to save chat to database (String ID: {str_id}): {e}")
            return None

    # Add the save operation to background tasks, including the RAG_APP_NAME
    background_tasks.add_task(save_chat_to_db, app.state.pool, question, updated_answer, citations, chat_str_id, RAG_APP_NAME)

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

@app.get("/health")
async def health(request: Request):
    # Add a database connection check
    db_status = "connected"
    db_message = "Database connection successful."
    pool = request.app.state.pool
    if not pool or pool.is_closing():
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
        # Check if embedding model is loaded
        if embed_model is None:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Embedding model not initialized"}
            )
        
        # Check if Pinecone client is connected
        if pc is None:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Pinecone client not initialized"}
            )
            
        # Return success if all checks pass
        return JSONResponse(
            content={
                "status": "healthy",
                "message": "API is operational",
                "components": {
                    "embedding_model": "initialized",
                    "pinecone": "connected",
                    "database": db_status
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