# src/lightrag_pdf_processor.py
import os
import asyncio
import glob
import logging
import textract
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Union, Any, Optional
import uuid
import time

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete # Example models
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Setup logger
setup_logger("lightrag_openai_proxy_service", level="INFO")
logger = logging.getLogger("lightrag_openai_proxy_service")

# Configuration
WORKING_DIR = "./rag_pdf_storage_openai_proxy"
PDF_INPUT_DIRECTORY = "./sample_pdfs" # User needs to set this
SERVICE_HOST = "0.0.0.0"
SERVICE_PORT = 8001 # Port for this RAG service

# Global LightRAG instance
rag_instance: LightRAG = None

# --- Pydantic Models for OpenAI API Mimicry ---

class MessageContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class RequestMessage(BaseModel):
    role: str
    content: Union[str, List[MessageContentItem]]
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[RequestMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ResponseMessage(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"
    # logprobs: Optional[Any] = None # Omitting for simplicity

class UsageInfo(BaseModel): # Mocked
    prompt_tokens: int = 0 # Actual token counting for RAG is complex
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[UsageInfo] = UsageInfo() # Provide mocked usage
    # system_fingerprint: Optional[str] = None # OpenAI specific

# --- LightRAG Initialization and PDF Processing ---
async def initialize_rag_and_load_pdfs():
    global rag_instance
    logger.info("Initializing LightRAG instance for OpenAI proxy service...")
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
        logger.info(f"Created working directory: {WORKING_DIR}")

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable is not set. Internal LLM calls by LightRAG might fail.")

    rag_instance = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed, # Used for PDF embedding
        llm_model_func=gpt_4o_mini_complete, # LLM used by RAG for generation
    )
    await rag_instance.initialize_storages()
    await initialize_pipeline_status()
    logger.info("LightRAG instance initialized.")

    if not os.path.exists(PDF_INPUT_DIRECTORY):
        os.makedirs(PDF_INPUT_DIRECTORY)
        logger.info(f"Created placeholder PDF directory: {PDF_INPUT_DIRECTORY}.")
        logger.info(f"Please add your PDF documents to this directory: {os.path.abspath(PDF_INPUT_DIRECTORY)}")
        try:
            from reportlab.pdfgen import canvas
            dummy_pdf_path = os.path.join(PDF_INPUT_DIRECTORY, "dummy_openai_proxy.pdf")
            if not os.path.exists(dummy_pdf_path):
                c = canvas.Canvas(dummy_pdf_path)
                c.drawString(100, 750, "Dummy PDF for LightRAG OpenAI Proxy Service.")
                c.save()
                logger.info(f"Created a dummy PDF for testing: {dummy_pdf_path}")
        except ImportError: logger.warning("reportlab not found, could not create dummy PDF.")
        except Exception as e_pdf: logger.error(f"Could not create dummy PDF: {e_pdf}")
            
    pdf_files = glob.glob(os.path.join(PDF_INPUT_DIRECTORY, "*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {PDF_INPUT_DIRECTORY}.")

    if pdf_files:
        # This part can be time-consuming for many/large PDFs.
        # Consider making it asynchronous or a separate management command if startup time is critical.
        all_docs_content = []
        all_doc_paths = []
        for pdf_path in pdf_files:
            try:
                logger.info(f"Processing PDF: {pdf_path}")
                text_bytes = textract.process(pdf_path)
                text_str = text_bytes.decode('utf-8').strip()
                if text_str:
                    all_docs_content.append(text_str)
                    all_doc_paths.append(pdf_path)
                    logger.info(f"Extracted text from {pdf_path} (Length: {len(text_str)})")
                else: logger.warning(f"No text extracted from {pdf_path}")
            except Exception as e: logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        if all_docs_content:
            logger.info(f"Inserting {len(all_docs_content)} documents into LightRAG...")
            await rag_instance.insert(all_docs_content, file_paths=all_doc_paths)
            logger.info("Documents inserted successfully.")
        else: logger.warning("No content extracted from PDFs to insert.")
    else: logger.warning(f"No PDFs found in '{PDF_INPUT_DIRECTORY}'. RAG knowledge base will be empty.")

# --- FastAPI Application ---
app = FastAPI(
    title="LightRAG OpenAI Proxy Service",
    description="Mimics OpenAI Chat Completions API, using LightRAG with PDF knowledge base.",
    version="0.1.1" # Incremented version
)

@app.on_event("startup")
async def startup_event():
    await initialize_rag_and_load_pdfs()

@app.on_event("shutdown")
async def shutdown_event():
    global rag_instance
    if rag_instance:
        await rag_instance.finalize_storages()
        logger.info("LightRAG instance finalized.")

# Mimic OpenAI's Chat Completions endpoint
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions_proxy(request: ChatCompletionRequest = Body(...)):
    global rag_instance
    if not rag_instance:
        logger.error("RAG instance not initialized for /v1/chat/completions.")
        raise HTTPException(status_code=503, detail="RAG service is not ready.")

    logger.info(f"Received OpenAI API request for model: {request.model}")
    
    # Extract the last user message as the primary query for RAG
    user_query_text = ""
    if request.messages:
        for msg in reversed(request.messages):
            if msg.role == "user":
                if isinstance(msg.content, str):
                    user_query_text = msg.content
                    break
                elif isinstance(msg.content, list):
                    # Concatenate text parts from multimodal content
                    text_parts = [item.text for item in msg.content if item.type == "text" and item.text]
                    user_query_text = "\n".join(text_parts)
                    # Note: Image data from msg.content is not directly passed to rag_instance.query here.
                    # The internal LLM of LightRAG (gpt_4o_mini_complete) might be vision-capable,
                    # but rag_instance.query itself primarily works with text queries for retrieval.
                    # For full multimodal RAG, LightRAG's query/LLM call would need specific handling.
                    break
    
    if not user_query_text:
        logger.warning("No user query text found in messages.")
        # Return a generic response or error, mimicking OpenAI
        return ChatCompletionResponse(
            id=f"chatcmpl-empty-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[Choice(message=ResponseMessage(role="assistant", content="I received an empty query."))]
        )

    logger.info(f"Performing RAG query with: '{user_query_text[:100]}...'")
    try:
        # Use LightRAG to get an answer based on PDFs + internal LLM
        # The `request.messages` (full history) is not directly passed to rag_instance.query.
        # LightRAG's llm_model_func (gpt_4o_mini_complete) will be called with the user_query_text + retrieved context.
        # If full conversation history is needed for the final LLM call within RAG,
        # LightRAG's query method or llm_model_func would need to be adapted.
        # For now, focusing on RAG for the latest query.
        rag_response_obj = await rag_instance.query(
            user_query_text,
            param=QueryParam(mode="hybrid") # Defaulting to hybrid
        )
        
        # Extract answer from RAG response
        # This depends on the structure of rag_response_obj
        if isinstance(rag_response_obj, str):
            final_answer = rag_response_obj
        elif hasattr(rag_response_obj, 'answer') and rag_response_obj.answer:
            final_answer = rag_response_obj.answer
        elif isinstance(rag_response_obj, dict) and rag_response_obj.get('answer'):
            final_answer = rag_response_obj['answer']
        else:
            logger.warning(f"Unexpected RAG response format: {type(rag_response_obj)}. Using as string.")
            final_answer = str(rag_response_obj)

        logger.info("RAG query processed successfully.")
        
        # Construct OpenAI-like response
        response_payload = ChatCompletionResponse(
            id=f"chatcmpl-rag-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model, # Echoing requested model
            choices=[
                Choice(
                    message=ResponseMessage(role="assistant", content=final_answer)
                )
            ]
        )
        return response_payload
        
    except Exception as e:
        logger.error(f"Error during RAG query or OpenAI response construction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing RAG query: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "rag_initialized": rag_instance is not None, "pdf_dir": PDF_INPUT_DIRECTORY}

if __name__ == "__main__":
    print(f"Starting LightRAG OpenAI Proxy Service on http://{SERVICE_HOST}:{SERVICE_PORT}")
    print(f"PDFs will be loaded from: {os.path.abspath(PDF_INPUT_DIRECTORY)}")
    print("Ensure 'textract', 'fastapi', 'uvicorn', 'python-multipart' (for form data if ever needed by OpenAI client), and LightRAG dependencies are installed.")
    print("If LightRAG's internal LLM is OpenAI, ensure OPENAI_API_KEY environment variable is set.")
    
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, log_level="info")