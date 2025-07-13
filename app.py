import asyncio
import re
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# Import modules
from modules.config import logger, validate_env_vars, get_system_prompt, RAG_APP_NAME
from modules.schemas import ChatRequest
from modules.utils import safe_send, format_query, deduplicate_documents, format_docs
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
    logger.info("Application startup: Connecting to database...")
    pool = await connect_db()
    logger.info("Application startup: Initializing database schema...")
    await init_db(pool)
    app.state.pool = pool
    yield
    logger.info("Application shutdown: Disconnecting from database...")
    await disconnect_db(app.state.pool)

# ------------------------------------------------------------------------------
# Initialize FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Initialize retrieval components and Pinecone clients
# ------------------------------------------------------------------------------
embed_model, bm25_encoder = initialize_retrieval_components()
pinecone_summary_index, pinecone_text_index = initialize_pinecone_clients()
logger.info("Retrieval components and Pinecone clients initialized successfully.")

app.include_router(history_router, prefix="/api")

# ------------------------------------------------------------------------------
# WebSocket endpoint for chat functionality
# ------------------------------------------------------------------------------
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New client connected")
    
    while True:
        try:
            data = await websocket.receive_json()
            
            if "feedback" in data and "id" in data:
                # Handle feedback separately
                chat_id_str = data.get("id")
                feedback = data.get("feedback")
                if chat_id_str and feedback in ["like", "dislike"]:
                    logger.info(f"Received feedback '{feedback}' for chat ID: {chat_id_str}")
                    success = await ChatRepository.update_feedback(app.state.pool, chat_id_str, feedback)
                    await safe_send(websocket, {"status": f"feedback_{'updated' if success else 'failed'}", "id": chat_id_str})
                else:
                    logger.warning(f"Invalid feedback data received: id={chat_id_str}, feedback={feedback}")
                continue

            chat_request = ChatRequest(**data)
            question = chat_request.question
            language = chat_request.language
            previous_chats = chat_request.previous_chats

            agent_result = await query_rewriting_agent(question, language, previous_chats)

            if agent_result["action"] in ["respond", "clarify", "identity"]:
                chat_str_id = str(uuid.uuid4())
                await safe_send(websocket, {"response": agent_result["response"], "sources": [], "id": chat_str_id})
                asyncio.create_task(ChatRepository.save_chat(app.state.pool, question, agent_result["response"], [], chat_str_id, RAG_APP_NAME))
                continue

            # Extract the rewritten queries from the nested structure
            rewritten_queries = agent_result.get("rewritten_queries", {})
            natural_language_query = rewritten_queries.get("natural_language_query", question)
            metadata_query = rewritten_queries.get("metadata_query", natural_language_query)
            
            logger.info("Rewritten Query (WebSocket): %s", natural_language_query)
            logger.info("Metadata Query (WebSocket): %s", metadata_query)

            retrieved_docs = await fetch_balanced_documents(
                metadata_query=metadata_query,
                natural_language_query=natural_language_query,
                pinecone_summary_index=pinecone_summary_index,
                pinecone_text_index=pinecone_text_index,
                embed_model=embed_model,
                bm25_encoder=bm25_encoder
            )
            deduplicated_docs = await deduplicate_documents(retrieved_docs)
            
            retrieved_urls = [doc.metadata.get('page_source', 'N/A') for doc in deduplicated_docs]
            logger.info("Retrieved %d documents (WebSocket): %s", len(deduplicated_docs), retrieved_urls)

            ranked_docs = []
            if deduplicated_docs:
                docs_for_reranking = []
                for doc in deduplicated_docs:
                    new_metadata = doc.metadata.copy()
                    new_metadata['source'] = new_metadata.get('page_source', 'N/A')
                    docs_for_reranking.append({"page_content": doc.page_content, "metadata": new_metadata})

                ranked_docs = await rerank_docs(natural_language_query, docs_for_reranking, agent_result.get("is_time_sensitive", False))
                final_doc_urls = [doc.get('metadata', {}).get('source', 'N/A') for doc in ranked_docs]
                logger.info("Final %d documents for LLM (WebSocket): %s", len(ranked_docs), final_doc_urls)

            if not ranked_docs:
                await safe_send(websocket, {"response": "I couldn't find any relevant information to answer your question.", "sources": [], "id": str(uuid.uuid4())})
                continue

            formatted_docs = format_docs(ranked_docs)
            relevant_history = [previous_chats[i] for i in agent_result.get("relevant_history_indices", []) if 0 <= i < len(previous_chats)]
            messages = [{"role": "system", "content": get_system_prompt()}] + relevant_history + [{"role": "user", "content": format_query(question, language, formatted_docs)}]

            complete_answer = ""
            async for chunk in await openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.1, stream=True):
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    complete_answer += delta_content
                    await safe_send(websocket, {"response": delta_content})
            
            updated_answer, citations = process_citations(complete_answer, ranked_docs)
            chat_str_id = str(uuid.uuid4())
            # The frontend expects a final message with the complete, updated answer.
            await safe_send(websocket, {"response": updated_answer, "sources": citations, "id": chat_str_id, "is_final": True})
            asyncio.create_task(ChatRepository.save_chat(app.state.pool, question, updated_answer, citations, chat_str_id, RAG_APP_NAME))

        except WebSocketDisconnect:
            logger.info("Client disconnected")
            break
        except Exception as e:
            logger.exception(f"Error in websocket endpoint: {e}")
            await safe_send(websocket, {"response": "An error occurred.", "sources": []})
            break

# ------------------------------------------------------------------------------
# HTTP endpoint for Telegram chat
# ------------------------------------------------------------------------------
@app.post("/telegram-chat")
async def telegram_chat(chat_request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        question = chat_request.question
        language = chat_request.language
        previous_chats = chat_request.previous_chats

        agent_result = await query_rewriting_agent(question, language, previous_chats)

        if agent_result["action"] in ["respond", "clarify"]:
            return {"response": agent_result["response"], "sources": []}

        # Extract the rewritten queries from the nested structure
        rewritten_queries = agent_result.get("rewritten_queries", {})
        rewritten_query = rewritten_queries.get("natural_language_query", question)
        metadata_query = rewritten_queries.get("metadata_query", rewritten_query)
        
        logger.info(f"Rewritten Query (Telegram): '{rewritten_query}'")
        logger.info(f"Metadata Query (Telegram): '{metadata_query}'")

        retrieved_docs = await fetch_balanced_documents(
            metadata_query=metadata_query,
            natural_language_query=rewritten_query,
            pinecone_summary_index=pinecone_summary_index,
            pinecone_text_index=pinecone_text_index,
            embed_model=embed_model,
            bm25_encoder=bm25_encoder
        )
        deduplicated_docs = await deduplicate_documents(retrieved_docs)

        retrieved_urls = [doc.metadata.get('source', 'N/A') for doc in deduplicated_docs]
        logger.info(f"Retrieved {len(deduplicated_docs)} documents (Telegram): {retrieved_urls}")

        ranked_docs = []
        if deduplicated_docs:
            docs_for_reranking = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in deduplicated_docs]
            ranked_docs = await rerank_docs(rewritten_query, docs_for_reranking, agent_result.get("is_time_sensitive", False))
            final_doc_urls = [doc['metadata'].get('source', 'N/A') for doc in ranked_docs]
            logger.info(f"Final {len(ranked_docs)} documents for LLM (Telegram): {final_doc_urls}")

        if not ranked_docs:
            return {"response": "No relevant information found to answer your question.", "sources": []}

        relevant_history = [previous_chats[i] for i in agent_result.get("relevant_history_indices", []) if 0 <= i < len(previous_chats)]
        messages = [{"role": "system", "content": get_system_prompt()}] + relevant_history + [{"role": "user", "content": format_query(question, language, ranked_docs)}]

        completion = await openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.1, stream=False)
        complete_answer = completion.choices[0].message.content
        
        updated_answer, citations = process_citations(complete_answer, ranked_docs)
        chat_str_id = str(uuid.uuid4())

        background_tasks.add_task(ChatRepository.save_chat, app.state.pool, question, updated_answer, citations, chat_str_id, RAG_APP_NAME)

        return {"response": updated_answer, "sources": citations, "id": chat_str_id}

    except Exception as e:
        logger.exception(f"Error in telegram-chat endpoint: {e}")
        return {"response": "An error occurred.", "sources": []}

# ------------------------------------------------------------------------------
# Health check endpoint
# ------------------------------------------------------------------------------
@app.get("/", response_class=JSONResponse)
async def root():
    return JSONResponse(content={"status": "working"})