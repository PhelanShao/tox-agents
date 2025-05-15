# src/lightrag_pdf_processor.py
import os
import asyncio
import glob # For finding PDF files
import logging
import textract # For PDF text extraction

from lightrag import LightRAG, QueryParam
# Example: using OpenAI models. Replace if using different providers.
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Setup logger for this module
setup_logger("lightrag_processor", level="INFO")
logger = logging.getLogger("lightrag_processor")

# Configuration
WORKING_DIR = "./rag_pdf_storage"  # Directory to store LightRAG data
# !!! IMPORTANT: User needs to set this path to their PDF directory !!!
PDF_INPUT_DIRECTORY = "./sample_pdfs" # Placeholder directory for your PDF documents

async def initialize_rag_instance() -> LightRAG:
    """Initializes and returns a LightRAG instance."""
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
        logger.info(f"Created working directory: {WORKING_DIR}")

    # Ensure OPENAI_API_KEY is set if using OpenAI models
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable is not set. OpenAI calls will likely fail.")
        # Consider raising an error or using a fallback if the key is essential

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,  # Replace with your preferred embedding function
        llm_model_func=gpt_4o_mini_complete,  # Replace with your preferred LLM
    )
    await rag.initialize_storages()
    # Initialize pipeline status if you plan to use asynchronous document processing pipeline
    await initialize_pipeline_status()
    logger.info("LightRAG instance initialized.")
    return rag

def find_pdf_files(pdf_directory: str) -> list[str]:
    """Finds all PDF files in the specified directory."""
    if not os.path.isdir(pdf_directory):
        logger.error(f"PDF directory not found: {pdf_directory}")
        return []
    
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_directory}.")
    return pdf_files

async def process_pdfs_and_insert(rag: LightRAG, pdf_files: list[str]):
    """Extracts text from PDF files and inserts them into LightRAG."""
    if not pdf_files:
        logger.warning("No PDF files to process.")
        return

    documents_content = []
    document_paths = []

    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            # textract.process returns bytes, so decode to utf-8
            text_content_bytes = textract.process(pdf_path)
            text_content_str = text_content_bytes.decode('utf-8').strip()
            
            if text_content_str: # Ensure there's content to add
                documents_content.append(text_content_str)
                document_paths.append(pdf_path)
                logger.info(f"Successfully extracted text from {pdf_path} (Length: {len(text_content_str)})")
            else:
                logger.warning(f"No text content extracted from {pdf_path}")
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            # Optionally, skip this file or handle error differently

    if documents_content:
        logger.info(f"Inserting {len(documents_content)} documents into LightRAG...")
        # Using rag.insert for simplicity.
        # For very large numbers of documents, consider apipeline_enqueue_documents
        await rag.insert(documents_content, file_paths=document_paths)
        logger.info("Documents inserted successfully.")
    else:
        logger.warning("No content extracted from PDFs to insert.")

async def perform_query(rag: LightRAG, query_text: str):
    """Performs a query against the RAG and prints the result."""
    if not query_text:
        logger.warning("Query text is empty. Skipping query.")
        return

    logger.info(f"Performing query: '{query_text}'")
    try:
        response = await rag.query(
            query_text,
            param=QueryParam(mode="hybrid")  # Example mode, can be "local", "global", "mix", "naive"
        )
        logger.info("Query response received.")
        # The response object structure depends on LightRAG version and query settings.
        # Typically, it might have attributes like response.answer or similar.
        print("\n--- Query Result ---")
        print(response) 
        print("--- End of Query Result ---\n")

    except Exception as e:
        logger.error(f"Error during query: {e}")

async def main_workflow():
    """Main workflow for the PDF RAG processor."""
    rag_instance = None
    try:
        # Create placeholder PDF directory if it doesn't exist
        if not os.path.exists(PDF_INPUT_DIRECTORY):
            os.makedirs(PDF_INPUT_DIRECTORY)
            logger.info(f"Created placeholder PDF directory: {PDF_INPUT_DIRECTORY}.")
            logger.info(f"Please add your PDF documents to this directory: {os.path.abspath(PDF_INPUT_DIRECTORY)}")
            # Optionally, create a dummy PDF for testing if reportlab is available
            try:
                from reportlab.pdfgen import canvas
                dummy_pdf_path = os.path.join(PDF_INPUT_DIRECTORY, "dummy_document.pdf")
                if not os.path.exists(dummy_pdf_path):
                    c = canvas.Canvas(dummy_pdf_path)
                    c.drawString(100, 750, "This is a dummy PDF document for LightRAG testing.")
                    c.drawString(100, 730, "LightRAG helps with retrieval-augmented generation from multiple documents.")
                    c.drawString(100, 710, "This framework processes PDFs and allows querying their content.")
                    c.save()
                    logger.info(f"Created a dummy PDF for testing: {dummy_pdf_path}")
            except ImportError:
                logger.warning("reportlab not found, could not create dummy PDF. Please add PDFs manually to the directory.")
            except Exception as e_pdf:
                 logger.error(f"Could not create dummy PDF: {e_pdf}")
        
        rag_instance = await initialize_rag_instance()
        pdf_files_to_process = find_pdf_files(PDF_INPUT_DIRECTORY)

        if pdf_files_to_process:
            await process_pdfs_and_insert(rag_instance, pdf_files_to_process)

            # --- Example Query ---
            # !!! IMPORTANT: User should change this to a relevant query based on their PDFs !!!
            sample_query = "What is retrieval-augmented generation?" # Placeholder query
            await perform_query(rag_instance, sample_query)
        else:
            logger.warning(f"No PDFs found in '{PDF_INPUT_DIRECTORY}'. Skipping document insertion and query.")
            logger.warning(f"Please add PDF files to {os.path.abspath(PDF_INPUT_DIRECTORY)} and re-run.")

    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}", exc_info=True)
    finally:
        if rag_instance:
            await rag_instance.finalize_storages()
            logger.info("LightRAG instance finalized.")

if __name__ == "__main__":
    print("Starting LightRAG PDF Processor...")
    print(f"Make sure your PDF documents are in: {os.path.abspath(PDF_INPUT_DIRECTORY)}")
    print("Ensure 'textract' and its dependencies (e.g., pdftotext) are installed.")
    print("If using OpenAI, ensure OPENAI_API_KEY environment variable is set.")
    
    asyncio.run(main_workflow())
    print("LightRAG PDF Processor finished.")