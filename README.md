# ToxPred: Molecular Toxicity Prediction and Analysis Platform

ToxPred is an interactive web application built with Gradio, designed to provide toxicity prediction, property prediction, and visualization analysis for molecular data. It integrates a chatbot, allowing users to interact with and analyze prediction results through natural language.

## Key Features
Deployed version(web app): https://bohrium.dp.tech/apps/tox-agents
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

4.  **Chatbot Integration**:
    *   Features a built-in chatbot interface for user interaction.
    *   Supports various large language models (configurable via API).
    *   **Analysis Function**: Users can send prediction results (image and data) of a specific frame to the chatbot for in-depth analysis. The bot generates a detailed chemical and biological analysis report based on the provided molecular data (toxicity probability, other properties).
    *   Supports multimodal input (text and images).

5.  **Nano Reactor**:
    *   A dedicated module for processing and analyzing molecular reaction simulation data.
    *   **File Upload**: Users can upload a `parameters3.dat` file and associated job files for a specific Job ID.
    *   **Run Analysis**:
        *   `Run Extra`: Executes an additional module for data extraction and plotting.
        *   `Analyze Fragments`: Analyzes molecular fragments during the reaction process and generates a timeline plot.
    *   **Fragment Extraction**: Users can select specific timesteps (or all timesteps) to extract coordinate data for reaction fragments.

## How to Use

### 1. Environment Setup

*   Ensure you have a Python environment installed.
*   Install the required dependencies, primarily `gradio`, `openai`, `numpy`, `pandas`, `Pillow`. (Refer to the `import` statements in the code for a complete list).

### 2. Launching the Application

*   Run the main program script from the project root directory:
    ```bash
    python toxpre.py
    ```
*   The application will start a local web server (the address and port, e.g., `http://0.0.0.0:50007`, will typically be displayed in the console).
*   Open this address in your web browser to access the application interface.

### 3. Interface Operation

The application interface consists of several main tabs:

*   **XYZ to NPZ Converter**:
    1.  Click "Upload XYZ file" to upload your `.xyz` file.
    2.  Click the "Convert to NPZ" button.
    3.  Once conversion is complete, status information will be displayed, and the generated `.npz` file will be available for download.

*   **Toxicity Prediction & Visualization**:
    1.  **API Configuration (Optional but Recommended)**:
        *   Expand the "API Configuration" section.
        *   Enter your API Base URL and API Key.
        *   Select a model or enter a custom model identifier.
        *   Click "Configure API". This is required for the chatbot functionality.
    2.  **Prediction & Visualization**:
        *   In the "Left Column - Toxicity Prediction" section:
            *   Click "Upload NPZ file" to upload the previously converted or an existing `.npz` file. Prediction will start automatically upon upload.
            *   Modify the model paths and reference file path in the "Property Prediction" and "Binary Prediction" sections as needed.
            *   Prediction result files (Property Predictions, Binary Predictions) and logs will be displayed.
            *   The "Probability Analysis" plot will show the probability distribution for binary predictions.
        *   Click "Convert to XYZ" to convert the uploaded NPZ file to XYZ format and display the molecular structure in the "Structure View" on the right.
        *   In the "Right Column - Visualization" section:
            *   Use the sliders and dropdowns in "Visualization Controls" to adjust the molecule display:
                *   `Select Frame`: Choose different frames from the molecular trajectory.
                *   `Representation Style`: Change the display style of the molecule.
                *   `X/Y/Z Rotation`, `Zoom`: Adjust the viewpoint.
            *   Click the "Export" button to export the current frame's image and corresponding prediction data (binary and property predictions). The exported image file and a JSON file containing the data will be provided for download.
            *   Click the "Analysis" button to send the currently exported image and data to the chatbot for analysis.
    3.  **Chat Interaction**:
        *   In the chat interface at the top, type your questions or commands.
        *   If "Enable Image Input" is checked and an image is uploaded, it will be sent with the message.
        *   Click "Send" or press Enter to send the message.
        *   The chat history will be displayed above.

*   **Nano Reactor**:
    1.  Enter a "Job ID".
    2.  Upload the `parameters3.dat` file and relevant job files ("Select files for Job ID").
    3.  Click "Upload Files".
    4.  Click "Run Extra" to perform additional data extraction and plotting. Result logs, plots, and download links will be displayed.
    5.  Click "Analyze Fragments" to analyze molecular fragments. Analysis output and a timeline plot will be shown.
    6.  Enter the timesteps for fragment coordinate extraction in "Select Timestep" (single, comma-separated multiple, or 'all').
    7.  Click "Extract Fragment Coordinates". Extraction status and fragment file download links will be displayed.

## File Structure (Main Scripts)

*   `toxpre.py`: The main entry point for the Gradio interface, integrating various functional modules.
*   `interface.py`: Defines core functions for conversion, prediction, visualization, and export, as well as some Gradio UI element handling logic.
*   `chatbot.py`: Implements the core logic for the chatbot, including interaction with large language model APIs, message memory management, and construction of the Gradio chat interface.
*   `predictor.py`: Contains the `BinaryPredictor` class for performing binary toxicity predictions.
*   `MoleculePredictor.py`: Contains the `MoleculePredictor` class for performing molecular property predictions.
*   `file_converter.py`: Contains functions for reading and writing XYZ and NPZ files.
*   `converter.py`: Contains the `NPZToXYZ` class for converting NPZ files back to XYZ format.
*   `visualizer.py`: Uses PyMol (via command-line calls or as a library) for molecular visualization and image export.
*   `probability_plot.py`: Creates the toxicity probability analysis plot.
*   `reactor.py`: Contains functions related to the Nano Reactor module.

## Important Notes

*   **Model Paths**: The code contains some hardcoded model paths (e.g., `/mnt/backup2/ai4s/...`). You will need to modify these paths according to your environment to point to the correct model files or directories.
*   **API Keys**: The chatbot functionality relies on external large language model APIs. Ensure you have correctly configured the Base URL and a valid API Key in the "API Configuration" section.
*   **Dependencies**: Make sure all Python dependencies are correctly installed.
*   **PyMol**: The molecular visualization feature may depend on PyMol being installed and configured.
