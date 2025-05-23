import os
import io
import json
import logging
# import gradio as gr # Removed Gradio
import tempfile
import shutil
import zipfile
import base64 # Added for process_export_for_chat if we keep it
import numpy as np
import pandas as pd
from chatbot import ChatInterface
from predictor import BinaryPredictor
from MoleculePredictor import MoleculePredictor
from reactor import (
    handle_uploaded_files, run_extra,
    run_net, on_timestep_select
)
from file_converter import read_xyz_file, save_npz
from converter import NPZToXYZ
from visualizer import (
    display_molecule_pymol, export_current_image
)
from probability_plot import create_probability_plot

log_stream = io.StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(log_stream),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CombinedInterface:
    def __init__(self):
        self.chat_interface = ChatInterface()
        
    # This method seems specific to a Gradio-based chat interaction flow
    # and is not directly used by the FastAPI backend.
    # It might need to be adapted or removed if not used by any backend logic.
    # For now, commenting it out to ensure no Gradio dependencies are hidden here.
    # def process_export_for_chat(self, export_path, export_data):
    #     # ... (implementation uses self.chat_interface.memory which might be Gradio specific)
    #     pass

# xyz_file_path is a string path to the file saved by FastAPI
def convert_xyz_to_npz(xyz_file_path: str):
    if not xyz_file_path or not os.path.exists(xyz_file_path):
        # Return format for FastAPI: tuple (error_message_or_None, result_path_or_None)
        # Consistent with process_property_prediction's success return.
        # However, the API endpoint for this expects FileResponse or HTTPException.
        # The functions here should probably raise exceptions on error, or return paths for success.
        # Let's make it return just the output path, and raise Exception on failure.
        raise ValueError("XYZ file path is missing or does not exist.")
        
    try:
        # The original function saved the file again, but FastAPI already saved it.
        # We can use xyz_file_path directly.
        # Create a temporary directory for any intermediate files if needed by read_xyz_file or save_npz
        with tempfile.TemporaryDirectory() as temp_dir:
            # If read_xyz_file or save_npz need the file in temp_dir with a specific name, copy it.
            # For now, assume they can work with the direct path or handle paths appropriately.
            
            # Original code used xyz_file.name, now we use xyz_file_path
            # temp_xyz = os.path.join(temp_dir, "temp.xyz") # This was a copy of the input
            # For now, directly use xyz_file_path assuming read_xyz_file can handle it.
            data_dict, invalid_frames = read_xyz_file(xyz_file_path)
            
            # Output file should be in a place accessible for FileResponse, or UPLOAD_DIR
            # Let's make it save to a temp file and return its path. The API endpoint will move/serve it.
            # Or, better, let it save to UPLOAD_DIR if that's the convention.
            # For now, save it in temp_dir and return path. API will copy it to UPLOAD_DIR if needed or serve from there.
            # The API currently expects this function to return the final path directly.
            # So this function should probably take an output_dir argument or save to a predictable place.
            # Let's have it create a file in a temp dir, and the API will handle it.
            # The API endpoint for xyz_to_npz doesn't specify an output_dir for this function.
            # It expects the function to return a path that it can then serve.
            
            # Let's save it with a unique name in a temporary fashion that the calling API can then handle.
            base_name = os.path.splitext(os.path.basename(xyz_file_path))[0]
            output_file_name = f"{base_name}_converted.npz"
            # Saving to current working directory, as original function did with shutil.copy2
            # This is not ideal for an API. It should save to UPLOAD_DIR or a dedicated output dir.
            # For now, let's stick to the function's apparent original behavior of creating in CWD,
            # and the API layer is responsible for cleanup or moving.
            # The FastAPI endpoint does not cleanup this returned path, it serves it then forgets it.
            # The caller of the API (frontend) downloads it.
            # The frontend XYZtoNPZConverter expects a file named based on original, e.g. "input.npz"
            
            # The original code saved to temp_dir then copied to CWD.
            # Let's simplify: save directly to a temporary file, return its path.
            # The API endpoint should then ensure this file is cleaned up after response.
            # For now, this function will create it, and it's up to API to manage.
            
            # Let's create the output file in a temporary manner.
            # The API endpoint is responsible for sending it and then cleaning it up.
            # This function should just return the path to the generated NPZ.
            # The original function returned a path to CWD.
            final_output_path = os.path.join(os.getcwd(), output_file_name) # Save to CWD

            info = save_npz(data_dict, final_output_path) # save_npz returns a string summary
            
            # The API endpoint expects to return this path using FileResponse.
            # The summary/info string is lost in current API design.
            # We should return only the path.
            return final_output_path
            
    except Exception as e:
        # Log the error for server-side records
        logger.error(f"Error during XYZ to NPZ conversion: {str(e)}")
        # Re-raise for the API to catch and turn into HTTPException
        raise RuntimeError(f"Error during conversion: {str(e)}")


# npz_file_path is a string path
def convert_npz_to_xyz(npz_file_path: str):
    if not npz_file_path or not os.path.exists(npz_file_path):
        raise ValueError("NPZ file path is missing or does not exist.")
        
    try:
        # Output file name based on the input, saved in CWD.
        output_file_name = os.path.splitext(os.path.basename(npz_file_path))[0] + "_converted.xyz"
        output_file_path = os.path.join(os.getcwd(), output_file_name)

        converter = NPZToXYZ(npz_file_path) # NPZToXYZ expects a path string
        result_message = converter.convert(output_file_path) # convert writes to output_file_path
        
        # result_message from converter.convert is a success/info string.
        # The API endpoint needs only the path.
        return output_file_path
    except Exception as e:
        logger.error(f"Error during NPZ to XYZ conversion: {str(e)}")
        raise RuntimeError(f"Error during conversion: {str(e)}")

# file_path is a string path. output_dir is where results (image, legend) should be saved.
def process_molecule_visualization(file_path: str, frame_index: int, representation: str, rotations: list, zoom: float, output_dir: str):
    if not file_path or not os.path.exists(file_path):
        # API expects (image_path, legend_path) or raises HTTPException
        # This function should raise error if file is missing.
        raise ValueError("Visualization file path is missing or does not exist.")
    
    # Ensure output_dir exists (API layer does this in current code)
    # os.makedirs(output_dir, exist_ok=True) # Not needed if API creates it

    # display_molecule_pymol is from visualizer.py
    # It's expected to save image and legend to output_dir and return their paths and total_frames
    img_path, legend_text_path, total_frames = display_molecule_pymol(
        file_path=file_path, # visualizer.py expects a path string
        frame_index=frame_index,
        representation=representation,
        rotations=rotations, # expects a list/tuple of 3 floats
        zoom=zoom,
        output_dir=output_dir # visualizer.py needs this to save files
    )
    
    # The API endpoint /api/v1/visualize/molecule expects this function to return (image_path, legend_path)
    # total_frames is not used by that specific API endpoint directly, but good to have.
    # The Gradio UI used total_frames to update a slider. React frontend handles frame input differently.
    return img_path, legend_text_path # total_frames is implicitly ignored by API caller

# file_path is a string path. model_path and reference_path are also paths.
# output_dir is where prediction CSV should be saved.
def process_binary_prediction(file_path: str, model_path: str, output_dir: str):
    if not file_path or not os.path.exists(file_path):
        raise ValueError("Binary prediction input file path is missing or does not exist.")
    
    try:
        predictor = BinaryPredictor(model_base_dir=model_path) # Assumes model_path is base dir for model files
        # Output path for the CSV, should be in output_dir
        output_csv_filename = os.path.splitext(os.path.basename(file_path))[0] + '_binary_predictions.csv'
        output_csv_path = os.path.join(output_dir, output_csv_filename)

        predictions_df = predictor.predict(file_path, output_csv_path) # predict should save to output_csv_path and return df
        
        # Post-processing of probabilities (already in your Gradio version)
        probabilities = predictions_df['probability'].values
        if np.max(probabilities) > 1 or np.min(probabilities) < 0: # Normalize if not in [0,1]
            probabilities = (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities))
            predictions_df['probability'] = probabilities
            predictions_df['prediction'] = (probabilities > 0.5).astype(int)
            predictions_df.to_csv(output_csv_path, index=False) # Save updated DataFrame
        
        # The API endpoint /api/v1/predict/binary expects (fig, result_text, output_file_path)
        # This function currently returns (output_path, log_messages string)
        # The fig is None, result_text can be the log_messages.
        # Let's align: return (None, log_messages_string, output_csv_path)
        log_messages = [
            f"Predictions: +{(predictions_df['prediction'] == 1).sum()}, -{(predictions_df['prediction'] == 0).sum()}",
            f"Probability range: {probabilities.min():.3f} - {probabilities.max():.3f}"
        ]
        
        return None, "\n".join(log_messages), output_csv_path
    except Exception as e:
        logger.error(f"Error in process_binary_prediction: {str(e)}")
        raise RuntimeError(f"Error during binary prediction: {str(e)}")

# file_path, reference_path, model_path are string paths.
# output_dir is where prediction CSV should be saved.
def process_property_prediction(file_path: str, model_path: str, reference_path: str, output_dir: str):
    if not all([file_path, model_path, reference_path]): # Add os.path.exists checks if necessary
        raise ValueError("Missing inputs for property prediction.")
    if not os.path.exists(file_path) or not os.path.exists(reference_path): # Model path might be a dir, harder to check simply
        raise ValueError("Input file or reference NPZ file does not exist.")

    try:
        predictor = MoleculePredictor() # Assumes this class is defined and works
        # The predict method of MoleculePredictor should save its output CSV into output_dir
        # and return the path to it, along with any message.
        output_csv_path, message = predictor.predict(file_path, reference_path, model_path, output_dir=output_dir)
        
        # The API endpoint /api/v1/predict/property expects (fig, result_text, output_file_path)
        # This function returns (output_file, message)
        # Let's align: return (None, message, output_csv_path)
        return None, message, output_csv_path
    except Exception as e:
        logger.error(f"Error in process_property_prediction: {str(e)}")
        raise RuntimeError(f"Error during property prediction: {str(e)}")

# current_image_path is a string path. other pred_file_paths are also paths.
# output_dir is where the new image and json should be saved.
def export_frame_data(current_image_path: str, output_format: str, frame_index: int, 
                        binary_pred_file_path: Optional[str], property_pred_file_path: Optional[str], 
                        output_dir: str):
    if not current_image_path or not os.path.exists(current_image_path):
        raise ValueError("Current image path for export is missing or does not exist.")

    try:
        # Ensure output_dir exists (API layer does this)
        # os.makedirs(output_dir, exist_ok=True)

        # export_current_image is from visualizer.py
        # It should save the new image (e.g., different format, or just copy) into output_dir
        # and return its path.
        # The first argument to export_current_image was 'current_image' (blob from Gradio),
        # now it's 'current_image_path' (string path from FastAPI).
        # visualizer.export_current_image needs to be adapted if it expected a blob.
        # Assuming visualizer.export_current_image can handle a path.
        exported_image_path = export_current_image(current_image_path, output_format, output_dir=output_dir)
        
        data = {}
        if binary_pred_file_path and os.path.exists(binary_pred_file_path):
            binary_df = pd.read_csv(binary_pred_file_path)
            if len(binary_df) > frame_index:
                row = binary_df.iloc[frame_index]
                binary_data = {col: str(row[col]) for col in binary_df.columns} # Ensure serializable
                data['binary_prediction'] = binary_data
        
        if property_pred_file_path and os.path.exists(property_pred_file_path):
            property_df = pd.read_csv(property_pred_file_path)
            if len(property_df) > frame_index:
                row = property_df.iloc[frame_index]
                property_data = {col: str(row[col]) for col in property_df.columns} # Ensure serializable
                data['property_prediction'] = property_data
        
        # Save JSON data to output_dir
        json_filename = f'frame_{frame_index}_data.json'
        exported_json_path = os.path.join(output_dir, json_filename)
        with open(exported_json_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return exported_image_path, exported_json_path
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        # Don't rmtree temp_dir here, output_dir is managed by API layer or might contain results.
        raise RuntimeError(f"Error during frame data export: {str(e)}")

# Removed Gradio specific create_interface() and demo.launch()
# if __name__ == "__main__":
#     demo = create_interface()
#     demo.launch(
#         server_name='0.0.0.0',
#         server_port=50001,
#         share=False,
#         show_api=False
#     )
# Content of these functions now assumes inputs are file paths,
# and they are responsible for creating output files in specified output_dir (for predictions/visualizations)
# or returning paths to temporary files (for conversions) that the API layer will handle.
# Error handling is done by raising exceptions, which FastAPI will convert to HTTPExceptions.


