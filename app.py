import asyncio
import re
import os
import time
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from langchain_huggingface import HuggingFaceEmbeddings

# Import modules
from modules.config import logger, validate_env_vars, system_prompt
from modules.schemas import ChatRequest, CitationSource
from modules.utils import safe_send, format_query
from modules.citations import process_citations
from modules.retrieval import initialize_pinecone_with_embeddings, rerank_docs, tavily_search
from modules.query_rewriting import query_rewriting_agent, openai_client

# Import database modules
from modules.database.database import init_db
from modules.database.repository import ChatRepository
from modules.database.views import router as history_router

# ------------------------------------------------------------------------------
# Initialize application and validate environment
# ------------------------------------------------------------------------------
validate_env_vars()

# ------------------------------------------------------------------------------
# Initialize embedding model globally
# ------------------------------------------------------------------------------
logger.info("Initializing embedding model globally...")
embed_model = HuggingFaceEmbeddings(
    model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
    model_kwargs={"trust_remote_code": True}
)
logger.info("Embedding model initialized successfully")

# ------------------------------------------------------------------------------
# Initialize FastAPI app with CORS middleware (restrict origins in production)
# ------------------------------------------------------------------------------
app = FastAPI()

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
# Initialize Pinecone retriever and client using global embedding model
# ------------------------------------------------------------------------------
retriever, pc = initialize_pinecone_with_embeddings(embed_model)

# ------------------------------------------------------------------------------
# Initialize database and include database router
# ------------------------------------------------------------------------------
init_db()
app.include_router(history_router, prefix="/api")

# ------------------------------------------------------------------------------
# WebSocket endpoint for chat functionality with improved error handling
# ------------------------------------------------------------------------------
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New client connected")
    
    while True:
        try:
            # Wait for client messages without a timeout
            data = await websocket.receive_json()
            
            try:
                # Check if this is a feedback update
                if "feedback" in data and "id" in data:
                    # Convert chat_id to integer if it's not None
                    try:
                        chat_id = int(data.get("id")) if data.get("id") is not None else None
                        feedback = data.get("feedback")
                        
                        if feedback in ("like", "dislike") and chat_id:
                            success = ChatRepository.update_feedback(chat_id, feedback)
                            if success:
                                await safe_send(websocket, {"status": "success", "message": f"Feedback '{feedback}' recorded for chat ID: {chat_id}"})
                            else:
                                await safe_send(websocket, {"status": "error", "message": f"Failed to record feedback for chat ID: {chat_id}"})
                            continue
                    except ValueError:
                        await safe_send(websocket, {"status": "error", "message": f"Invalid chat ID format: {data.get('id')}"})
                        continue
                    
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
                continue

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
                
                # Save the query-rewriting agent's response to the database
                try:
                    # Save to the database using our custom string ID
                    chat_id = ChatRepository.save_chat(question, agent_result["response"], [], chat_str_id)
                    logger.info(f"Query-rewriting agent response saved to database with numeric ID: {chat_id} and string ID: {chat_str_id}")
                except Exception as e:
                    logger.exception(f"Failed to save query-rewriting agent response to database: {e}")
                
                continue
                
            # Use the rewritten query for retrieval if available
            query_for_retrieval = agent_result.get("rewritten_query", question)
            
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
            
            # Retrieve documents using the retriever
            try:
                retrieved_docs = await asyncio.to_thread(retriever.invoke, query_for_retrieval)
            except Exception as e:
                error_details = str(e)
                logger.exception(f"Retrieval error: {error_details}")
                await safe_send(websocket, {
                    "response": "I encountered an issue while searching for information to answer your question. This might be because the question is outside my knowledge domain or due to a temporary system issue.", 
                    "error": "retrieval_error",
                    "error_details": error_details,
                    "sources": []
                })
                continue

            docs = [{
                "summary": ele.metadata.get("summary", ""),
                "chunk": ele.page_content,
                "page_source": ele.metadata.get("page_source", "")
            } for ele in retrieved_docs]

            if not docs:
                await safe_send(websocket, {"response": "No information found to answer your question.", "sources": []})
                continue

            # Rerank the documents (fallback to original docs if reranking fails)
            try:
                ranked_docs = await asyncio.to_thread(rerank_docs, query_for_retrieval, docs, pc)
            except Exception as e:
                logger.exception("Reranking error:")
                ranked_docs = docs

            # Prepare the conversation messages
            messages = [{"role": "system", "content": system_prompt}]
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

            # Process and map citations in the final answer
            try:
                updated_answer, citations = process_citations(complete_answer, ranked_docs)
            except Exception as e:
                logger.exception("Error processing citations:")
                updated_answer, citations = complete_answer, []

            # Generate a unique string ID for this chat
            chat_str_id = str(uuid.uuid4())
            
            # Send the response to the client with the string ID immediately
            await safe_send(websocket, {
                "response": updated_answer,
                "sources": citations,
                "id": chat_str_id  # Include the string ID with the response
            })
            
            # Now save the chat history to the database with our pre-generated ID
            try:
                # Save to the database using our custom string ID
                chat_id = ChatRepository.save_chat(question, updated_answer, citations, chat_str_id)
                logger.info(f"Chat saved to database with numeric ID: {chat_id} and string ID: {chat_str_id}")
            except Exception as e:
                logger.exception(f"Failed to save chat to database: {e}")

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
    query_for_retrieval = agent_result.get("rewritten_query", question)
    
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
    
    # Retrieve documents using the retriever.
    try:
        retrieved_docs = await asyncio.to_thread(retriever.invoke, query_for_retrieval)
    except Exception as e:
        logger.exception("Document retrieval error:")
        return {
            "response": "This question is out of my scope. Please try again with another question.",
            "sources": []
        }

    docs = [{
        "summary": ele.metadata.get("summary", ""),
        "chunk": ele.page_content,
        "page_source": ele.metadata.get("page_source", "")
    } for ele in retrieved_docs]

    if not docs:
        return {
            "response": "No information found to answer your question.",
            "sources": []
        }

    # Rerank the documents (fallback to original docs if reranking fails)
    try:
        ranked_docs = await asyncio.to_thread(rerank_docs, query_for_retrieval, docs, pc)
    except Exception as e:
        logger.exception("Reranking error:")
        ranked_docs = docs

    # Prepare the conversation messages.
    messages = [{"role": "system", "content": system_prompt}]
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

    # Process and map citations in the final answer.
    try:
        updated_answer, citations = process_citations(complete_answer, ranked_docs)
    except Exception as e:
        logger.exception("Error processing citations:")
        updated_answer, citations = complete_answer, []
        
    # Generate a unique string ID for this chat
    chat_str_id = str(uuid.uuid4())
    
    # Define a function to save chat in the background
    def save_chat_to_db(q, answer, cite, str_id):
        try:
            chat_id = ChatRepository.save_chat(q, answer, cite, str_id)
            logger.info(f"Chat saved to database with numeric ID: {chat_id} and string ID: {str_id}")
            return chat_id
        except Exception as e:
            logger.exception(f"Failed to save chat to database: {e}")
            return None
    
    # Add the save operation to background tasks
    # This will run after the response is sent to the client
    background_tasks.add_task(save_chat_to_db, question, updated_answer, citations, chat_str_id)

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
async def health():
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
                    "pinecone": "connected"
                }
            }
        )
    except Exception as e:
        logger.exception("Health check failed:")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )