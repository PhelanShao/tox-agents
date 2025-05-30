# ToxPred: Molecular Toxicity Prediction and Analysis Platform

ToxPred is an interactive web application built with Gradio, designed to provide toxicity prediction, property prediction, and visualization analysis for molecular data. It integrates a chatbot, allowing users to interact with and analyze prediction results through natural language.

Deployed version -tox agents (web app): https://bohrium.dp.tech/apps/tox-agents

Molreac (Nano reactor app): https://bohrium.dp.tech/apps/molreac



## Key Features

1.  **XYZ to NPZ File Conversion**:
    *   Users can upload molecular structure files in `.xyz` format.
    *   The application converts them to `.npz` format, which is required for subsequent prediction and analysis steps.

2.  **Toxicity and Property Prediction**:
    *   **Property Prediction**: After uploading an `.npz` file, various physicochemical properties of the molecule can be predicted. Requires specifying model directory path and reference NPZ file path.
    *   **Binary Toxicity Prediction**: Predicts whether a molecule is toxic (binary classification). Requires specifying the model directory path.
    *   Prediction results (probability and class) are provided as a CSV file, and a probability analysis plot is displayed.

3.  **Molecular Visualization**:
    *   `.npz` files can be converted back to `.xyz` format for visualization.
    *   Offers multiple molecular representation styles (e.g., sticks, ball_and_stick, spacefill).
    *   Users can interactively rotate, zoom, and pan the molecular structure.
    *   Supports selecting and viewing different frames in a molecular trajectory.
    *   The current view can be exported as a PNG or JPG image.

4.  **Chatbot Integration & RAG Enhancement**:
    *   Features a built-in chatbot interface for user interaction.
    *   The chatbot can be configured to use a backend RAG (Retrieval-Augmented Generation) service (`lightrag_pdf_processor.py`) which leverages a knowledge base built from PDF documents.
    *   This RAG service (mimicking an OpenAI API) enhances queries with information extracted from the PDF knowledge base before sending them to a configured Large Language Model.
    *   Supports various large language models (configurable via the RAG service or directly if RAG is not used).
    *   **Analysis Function**: Users can send prediction results (image and data) of a specific frame to the chatbot for in-depth analysis.
    *   Supports multimodal input (text and images) if the underlying LLM and RAG setup supports it.

5.  **Nano Reactor**:
    *   A dedicated module for processing and analyzing molecular reaction simulation data.
    *   **File Upload**: Users can upload a `parameters3.dat` file and associated job files for a specific Job ID.
    *   **Run Analysis**:
        *   `Run Extra`: Executes an additional module for data extraction and plotting.
        *   `Analyze Fragments`: Analyzes molecular fragments during the reaction process and generates a timeline plot.
    *   **Fragment Extraction**: Users can select specific timesteps (or all timesteps) to extract coordinate data for reaction fragments.

## How to Use

### 1. Environment Setup

*   **Python**: Ensure you have a Python environment installed (e.g., Python 3.9+).
*   **Core Dependencies**:
    ```bash
    pip install gradio openai numpy pandas Pillow textract fastapi "uvicorn[standard]" python-multipart httpx reportlab plotly
    ```
    *(Note: `httpx` is added for `chatbot.py` to call the RAG service, `reportlab` is optional for dummy PDF generation in the RAG service, `plotly` is used by `extract_module.py` and `fragment_module.py`)*

*   **UniMol Tools (`unimol_tools`)**: This is crucial for molecular predictions.
    *   **PyTorch**: Install PyTorch according to your environment (CPU or CUDA). More details: [PyTorch Get Started](https://pytorch.org/get-started/locally/)
    *   **RDKit**: `unimol_tools` currently requires `numpy<2.0.0`. Install RDKit (which includes a compatible NumPy version):
        ```bash
        pip install rdkit
        # Or, if you need a specific numpy version first:
        # pip install "numpy<2.0.0"
        # pip install rdkit
        ```
    *   **Install `unimol_tools`**:
        *   **Option 1: From PyPI (Recommended)**
            ```bash
            pip install unimol_tools --upgrade
            pip install huggingface_hub  # Recommended for automatic model downloads
            ```
            You can set `export HF_ENDPOINT=https://hf-mirror.com` if Hugging Face Hub downloads are slow.
        *   **Option 2: From Source (for latest version)**
            ```bash
            # Ensure dependencies like Cython, etc., are met as per Uni-Mol's requirements.txt
            # pip install -r requirements.txt # (from Uni-Mol repo if needed)
            git clone https://github.com/deepmodeling/Uni-Mol.git
            cd Uni-Mol/unimol_tools
            python setup.py install
            cd ../.. # Return to project root
            ```
    *   **Uni-Mol Model Weights**:
        *   Models can be automatically downloaded via `huggingface_hub` if installed.
        *   Alternatively, set the `UNIMOL_WEIGHT_DIR` environment variable if you have downloaded weights manually:
            ```bash
            export UNIMOL_WEIGHT_DIR=/path/to/your/unimol_weights_dir/
            ```

*   **LightRAG (for PDF RAG service)**:
    ```bash
    pip install "lightrag-hku[openai]" # Or other extras depending on your LLM/embedding choice for LightRAG
    ```

### 2. Launching the Applications

This project now potentially involves two services: the main ToxPred application and the LightRAG PDF service.

*   **A. Launch the LightRAG PDF Service (Optional, for RAG-enhanced chat)**:
    1.  Navigate to the project root directory (`f:/develop/toxchat/toxchat`).
    2.  Ensure your PDF documents are placed in the directory specified by `PDF_INPUT_DIRECTORY` in `src/lightrag_pdf_processor.py` (default: `./sample_pdfs`).
    3.  If the LightRAG service's internal LLM (e.g., OpenAI) requires an API key, set the `OPENAI_API_KEY` environment variable.
    4.  Run the service:
        ```bash
        python src/lightrag_pdf_processor.py
        ```
        This service will typically run on `http://0.0.0.0:8001`.

*   **B. Launch the Main ToxPred Application (`toxpre.py`)**:
    1.  Navigate to the project root directory.
    2.  Run the main program script:
        ```bash
        python src/toxpre.py
        ```
    3.  The application will start a local web server (e.g., `http://0.0.0.0:50007`).
    4.  Open this address in your web browser.

### 3. Interface Operation

*   **XYZ to NPZ Converter**: (As previously described)
*   **Toxicity Prediction & Visualization**: (As previously described)
    *   **API Configuration for Chatbot**:
        *   If using the LightRAG PDF service:
            *   Set "Base URL" to the LightRAG service address (e.g., `http://localhost:8001`).
            *   "API Key" can be a dummy value (e.g., "rag_service_key") as the RAG service currently doesn't validate it from `chatbot.py`.
            *   The "Model" selected here will be passed in the request but the RAG service will use its internally configured LLM.
        *   If NOT using the LightRAG service (direct LLM call):
            *   Set "Base URL" and "API Key" for your chosen LLM provider.
*   **Nano Reactor**: (As previously described)

## File Structure (Main Scripts)

*   `src/toxpre.py`: Main Gradio UI for ToxPred.
*   `src/interface.py`: Core backend logic for ToxPred features.
*   `src/chatbot.py`: Chatbot logic, now capable of querying the LightRAG service or a direct LLM.
*   `src/lightrag_pdf_processor.py`: FastAPI service for RAG using PDF documents, mimicking an OpenAI API.
*   `src/predictor.py`: `BinaryPredictor` for toxicity.
*   `src/MoleculePredictor.py`: `MoleculePredictor` for properties.
*   (Other utility and module scripts as previously listed)

## Important Notes

*   **Model Paths**:
    *   For `toxpre.py` predictions: The code contains some hardcoded model paths (e.g., `/mnt/backup2/ai4s/...`). Modify these in the Gradio UI or code to point to your actual Uni-Mol model directories.
    *   For `unimol_tools`: Ensure models are downloadable via `huggingface_hub` or `UNIMOL_WEIGHT_DIR` is set.
*   **API Keys**:
    *   If `chatbot.py` is configured to call an LLM API directly (not through the RAG service), ensure the Base URL and API Key are correctly set in the Gradio UI.
    *   If the LightRAG service (`src/lightrag_pdf_processor.py`) uses an LLM that requires an API key (e.g., OpenAI for its internal `llm_model_func`), ensure the `OPENAI_API_KEY` (or equivalent for other providers) environment variable is set when running the RAG service.
*   **Dependencies**: This project has several key dependencies. Pay close attention to the "Environment Setup" section, especially for PyTorch, RDKit (with `numpy<2.0.0`), `unimol_tools`, and `lightrag-hku`.
*   **PyMol**: Molecular visualization in `visualizer.py` might depend on PyMol installation and configuration.
*   **`textract` Dependencies**: `textract` (used by the RAG service) may require system-level packages like `pdftotext` to process PDFs. Check `textract` documentation for OS-specific requirements.
