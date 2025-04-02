import asyncio
import re

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from langchain_huggingface import HuggingFaceEmbeddings

# Import modules
from modules.config import logger, validate_env_vars, system_prompt
from modules.schemas import ChatRequest, CitationSource
from modules.utils import safe_send, format_query
from modules.citations import process_citations
from modules.retrieval import initialize_pinecone_with_embeddings, rerank_docs, tavily_search
from modules.query_rewriting import query_rewriting_agent, openai_client

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Initialize Pinecone retriever and client using global embedding model
# ------------------------------------------------------------------------------
retriever, pc = initialize_pinecone_with_embeddings(embed_model)

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
                chat_request = ChatRequest(**data)
            except ValidationError as ve:
                logger.exception("Validation error:")
                await safe_send(websocket, {"response": "Invalid request format.", "sources": []})
                continue

            question = chat_request.question
            language = chat_request.language
            previous_chats = chat_request.previous_chats

            # Apply query rewriting agent to analyze and possibly rewrite the query
            agent_result = await query_rewriting_agent(question, language, previous_chats)
            
            # Handle direct responses (out of scope or identity queries)
            if agent_result["action"] == "respond":
                await safe_send(websocket, {"response": agent_result["response"], "sources": []})
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
                logger.exception("Retrieval error:")
                await safe_send(websocket, {"response": "This question is out of my scope. Please try again with another question.", "sources": []})
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
                logger.exception("Error during streaming:")
                await safe_send(websocket, {"response": "Response generation failed. Please try again later.", "sources": []})
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

            await safe_send(websocket, {
                "response": updated_answer,
                "sources": citations
            })

        except WebSocketDisconnect:
            logger.info("Client disconnected")
            break
        except Exception as e:
            logger.exception("Unexpected error in websocket endpoint:")
            try:
                await safe_send(websocket, {"response": "An unexpected error occurred. Please try again.", "sources": []})
            except:
                logger.info("Could not send error message as websocket appears to be closed")
            break

# ------------------------------------------------------------------------------
# HTTP endpoint for Telegram chat
# ------------------------------------------------------------------------------
@app.post("/telegram-chat")
async def telegram_chat(chat_request: ChatRequest):
    # Extract the question and language from the validated request body.
    logger.info(f"Received telegram chat request: {chat_request}")
    
    question = chat_request.question
    language = chat_request.language
    previous_chats = chat_request.previous_chats

    # Apply query rewriting agent to analyze and possibly rewrite the query
    agent_result = await query_rewriting_agent(question, language, previous_chats)
    
    # Handle direct responses (out of scope or identity queries)
    if agent_result["action"] == "respond":
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
        "page_source": ele.metadata.get("source", "")
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
        async for chunk in completion:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                if "🛑" in delta_content:
                    isResponseAvailable = False
                    break
                complete_answer += delta_content
                # Remove inline citation markers from the streamed chunk.
                cleaned_content = re.sub(r'\[\d+\]', '', delta_content)
    except Exception as e:
        logger.exception("Error during streaming response:")
        return {
            "response": "Response generation failed. Please try again later.",
            "sources": []
        }

    # If the initial response indicates no answer, perform a fallback search.
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
                    cleaned_content = re.sub(r'\[\d+\]', '', delta_content)
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

    return {"response": updated_answer, "sources": citations}

# ------------------------------------------------------------------------------
# Simple health check endpoint
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return JSONResponse(content={"message": "working"})

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