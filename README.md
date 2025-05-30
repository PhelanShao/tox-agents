# ToxPred React Interface

## Overview

This version of the ToxPred application uses a modern React frontend with a Python FastAPI backend. It provides functionalities for toxicity prediction, molecule visualization, file conversion, and an AI chat interface for analysis.

## Prerequisites

*   **Node.js and npm:** Required for frontend development or if you need to rebuild the frontend. You can download them from [https://nodejs.org/](https://nodejs.org/).
*   **Python:** Version 3.7+ is recommended. You can download Python from [https://www.python.org/](https://www.python.org/).

## Backend Setup

1.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    The `requirements.txt` file includes the core dependencies for running the backend server and basic functionalities.
    ```bash
    pip install -r requirements.txt
    ```

## Frontend Setup

The frontend is pre-built and located in the `frontend/dist/` directory. The FastAPI backend is configured to serve these static files.

**For Developers (Modifying or Rebuilding the Frontend):**

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```

2.  Install npm dependencies:
    ```bash
    npm install
    ```
    If you encounter peer dependency issues, you might need to use:
    ```bash
    npm install --legacy-peer-deps
    ```

3.  Build the frontend for production:
    ```bash
    npm run build
    ```
    This will regenerate the files in the `frontend/dist/` directory.

## Running the Application

1.  **Start the FastAPI Backend:**
    Navigate to the project's root directory (where the `src/` folder is located).
    Run the API server:
    ```bash
    python src/api.py
    ```
    For development with auto-reload, you can use Uvicorn directly from the project root:
    ```bash
    python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
    ```
    The backend server will typically start on `http://localhost:8000`.

2.  **Access the Application:**
    Open your web browser and navigate to:
    [http://localhost:8000](http://localhost:8000)

## Optional: Enabling Full Prediction/RAG Capabilities

The default `requirements.txt` installs dependencies for the core application, excluding heavy AI models to keep the setup lightweight. For full features, including molecule predictions with UniMol models and advanced RAG (Retrieval Augmented Generation) chat capabilities with LightRAG, you need to install additional dependencies.

1.  **Install Full Requirements:**
    Use the `requirements_full.txt` file for this purpose:
    ```bash
    pip install -r requirements_full.txt
    ```
    This file includes the core dependencies plus `torch`, `unimol_tools` (placeholder for UniMol related tools), and `lightrag-hku[openai]` (placeholder for LightRAG).
    *Note: `unimol_tools` and `lightrag-hku[openai]` are placeholders. You may need to find the correct package names or follow specific installation instructions for UniMol and LightRAG if these names are not exact.*

2.  **Model and Service Setup:**
    *   **UniMol Models:** For molecule predictions, you will need to download and set up the UniMol model files. Refer to the original project documentation (now available in `README_original_gradio.md` or the relevant model repositories) for instructions on obtaining and placing these models. The application may look for these models in specific paths (e.g., `/mnt/backup2/ai4s/...` as seen in some component defaults, or configurable paths).
    *   **LightRAG Service:** For the RAG chat features, ensure your LightRAG service or any other specified AI model provider (e.g., OpenRouter.ai, OpenAI) is correctly configured with API keys and base URLs in the Chat Interface's configuration section.

    For detailed setup of these advanced features, please consult the `README_original_gradio.md` file or the documentation associated with UniMol and LightRAG.

## File Structure (Brief)

*   `frontend/`: Contains the React frontend source code.
    *   `frontend/src/`: Main source files for the React application.
    *   `frontend/dist/`: Pre-built static assets of the frontend (served by FastAPI).
*   `src/`: Contains the Python backend source code.
    *   `src/api.py`: The main FastAPI application defining API endpoints and serving the frontend.
    *   `src/interface.py`: Core logic functions previously used by Gradio, now refactored for FastAPI.
    *   `src/chatbot.py`: Chat interface logic.
    *   `src/predictor.py`, `src/MoleculePredictor.py`: Prediction model interfaces.
    *   `src/reactor.py`: Nano Reactor simulation logic.
    *   `src/visualizer.py`: Molecule visualization logic.
*   `requirements.txt`: Core Python dependencies for the backend.
*   `requirements_full.txt`: Python dependencies for full backend capabilities, including AI models.
*   `README.md`: This file.
*   `README_original_gradio.md`: The README file from the original Gradio-based version of the application.

---

This setup allows for a decoupled frontend and backend, with the React application providing a dynamic user experience and FastAPI handling the core processing and API services.
