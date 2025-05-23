from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import shutil
import json
from typing import List, Optional, Dict, Any

# Placeholder for imports from the existing codebase
from interface import (
    convert_xyz_to_npz,
    convert_npz_to_xyz,
    process_property_prediction,
    process_binary_prediction,
    process_molecule_visualization,
    export_frame_data,
)

# Conditional imports for modules that might not exist or have all functions
try:
    from probability_plot import create_probability_plot
except ImportError:
    def create_probability_plot(*args, **kwargs):
        raise NotImplementedError("probability_plot.create_probability_plot is not available")

try:
    import reactor
except ImportError:
    class DummyReactor:
        def handle_uploaded_files(self, *args, **kwargs): raise NotImplementedError("reactor.handle_uploaded_files is not available")
        def run_extra(self, *args, **kwargs): raise NotImplementedError("reactor.run_extra is not available")
        def run_net(self, *args, **kwargs): raise NotImplementedError("reactor.run_net is not available")
        def on_timestep_select(self, *args, **kwargs): raise NotImplementedError("reactor.on_timestep_select is not available")
    reactor = DummyReactor()

try:
    from chatbot import ChatInterface 
except ImportError:
    class DummyChatInterface:
        def __init__(self): self.client = None
        def initialize_client(self, *args, **kwargs):
            print("Warning: ChatInterface.initialize_client called but not fully implemented.")
            return "Client initialized (dummy)"
        def process_message(self, *args, **kwargs):
            print("Warning: ChatInterface.process_message called but not fully implemented.")
            return {"chat_history": [], "error": "ChatInterface not fully implemented"}
    ChatInterface = DummyChatInterface

# Main FastAPI app
app = FastAPI()

# API Router with /api/v1 prefix
api_router = APIRouter(prefix="/api/v1")

# Initialize ChatInterface if it's a class
chat_interface = ChatInterface()

# --- Helper Functions ---
# UPLOAD_DIR is relative to where the script is run. If run from `src/`, it's `src/temp_uploads`.
# If run from project root `python -m src.api`, it's `temp_uploads` in root.
# For consistency, let's define it from the project root perspective if possible, or ensure it's created correctly.
# Assuming script is run from `src` or project root such that UPLOAD_DIR is correctly located.
# For serving static files from UPLOAD_DIR via an API endpoint, the path needs to be stable.
# Let's assume UPLOAD_DIR is at the project root for clarity with ../frontend/dist
# Or, more robustly, define UPLOAD_DIR based on the script's location.
# For simplicity, keeping UPLOAD_DIR as "temp_uploads". It will be created relative to CWD.
# The /api/v1/static_files endpoint will use this.
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of api.py (src/)
UPLOAD_DIR = os.path.join(APP_ROOT_DIR, "..", "temp_uploads_api") # Place it in project root, sibling to src/ and frontend/
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_upload_file(upload_file: UploadFile) -> str:
    try:
        # Ensure filename is safe
        safe_filename = os.path.basename(upload_file.filename)
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return file_path
    finally:
        upload_file.file.close()

def cleanup_file(file_path: str):
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

# --- API Endpoints (now on api_router) ---

@api_router.get("/")
async def api_root(): # Renamed from root to avoid conflict if any
    return {"message": "Welcome to the Molecule API v1"}

# 1. File Conversions
@api_router.post("/convert/xyz-to-npz")
async def api_convert_xyz_to_npz_v1(xyz_file: UploadFile = File(...)): # Renamed for clarity
    xyz_file_path = save_upload_file(xyz_file)
    try:
        output_npz_path = convert_xyz_to_npz(xyz_file_path)
        if not output_npz_path or not os.path.exists(output_npz_path):
            raise HTTPException(status_code=500, detail="Failed to convert XYZ to NPZ.")
        return FileResponse(output_npz_path, media_type="application/octet-stream", filename=os.path.basename(output_npz_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(xyz_file_path)

@api_router.post("/convert/npz-to-xyz")
async def api_convert_npz_to_xyz_v1(npz_file: UploadFile = File(...)): # Renamed
    npz_file_path = save_upload_file(npz_file)
    try:
        output_xyz_path = convert_npz_to_xyz(npz_file_path)
        if not output_xyz_path or not os.path.exists(output_xyz_path):
            raise HTTPException(status_code=500, detail="Failed to convert NPZ to XYZ.")
        return FileResponse(output_xyz_path, media_type="text/plain", filename=os.path.basename(output_xyz_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(npz_file_path)

# 2. Predictions
@api_router.post("/predict/property")
async def api_process_property_prediction_v1(npz_file: UploadFile = File(...), model_path: str = Form(...), reference_path: Optional[str] = Form(None)):
    npz_file_path = save_upload_file(npz_file)
    temp_output_dir = os.path.join(UPLOAD_DIR, f"prop_pred_output_{os.path.basename(npz_file.filename)}") # Unique output dir
    os.makedirs(temp_output_dir, exist_ok=True)
    try:
        fig, result_text, output_file_path = process_property_prediction(npz_file_path, model_path, reference_path, temp_output_dir)
        if not output_file_path or not os.path.exists(output_file_path):
            potential_outputs = [f for f in os.listdir(temp_output_dir) if f.endswith(".csv")]
            if potential_outputs: output_file_path = os.path.join(temp_output_dir, potential_outputs[0])
            else: raise HTTPException(status_code=500, detail="Property prediction did not produce an identifiable output file.")
        return FileResponse(output_file_path, media_type="application/octet-stream", filename=os.path.basename(output_file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(npz_file_path)
        # shutil.rmtree(temp_output_dir) # Cleanup temp_output_dir after FileResponse (tricky for async) - usually handled by OS temp cleaning or dedicated jobs

@api_router.post("/predict/binary")
async def api_process_binary_prediction_v1(npz_file: UploadFile = File(...), model_path: str = Form(...)):
    npz_file_path = save_upload_file(npz_file)
    temp_output_dir = os.path.join(UPLOAD_DIR, f"bin_pred_output_{os.path.basename(npz_file.filename)}")
    os.makedirs(temp_output_dir, exist_ok=True)
    try:
        fig, result_text, output_file_path = process_binary_prediction(npz_file_path, model_path, temp_output_dir)
        if not output_file_path or not os.path.exists(output_file_path):
            potential_outputs = [f for f in os.listdir(temp_output_dir) if f.endswith(".csv")]
            if potential_outputs: output_file_path = os.path.join(temp_output_dir, potential_outputs[0])
            else: raise HTTPException(status_code=500, detail="Binary prediction did not produce an identifiable output file.")
        return FileResponse(output_file_path, media_type="application/octet-stream", filename=os.path.basename(output_file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(npz_file_path)

@api_router.post("/predict/probability_plot")
async def api_create_probability_plot_v1(results_file: UploadFile = File(None), results_data: Optional[str] = Form(None), true_label: Optional[str] = Form("1")):
    results_file_path = None
    plot_output_dir = os.path.join(UPLOAD_DIR, "prob_plot_output")
    os.makedirs(plot_output_dir, exist_ok=True)
    try:
        binary_prediction_results = []
        if results_file:
            results_file_path = save_upload_file(results_file)
            if results_file.filename.endswith(".json"):
                with open(results_file_path, 'r') as f: binary_prediction_results = json.load(f)
            elif results_file.filename.endswith(".csv"):
                with open(results_file_path, 'r') as f:
                    next(f) 
                    for line in f:
                        parts = line.strip().split(',');
                        if len(parts) == 2: binary_prediction_results.append([parts[0], float(parts[1])])
            else: raise HTTPException(status_code=400, detail="Unsupported results file format.")
        elif results_data:
            try: binary_prediction_results = json.loads(results_data)
            except json.JSONDecodeError: raise HTTPException(status_code=400, detail="Invalid JSON data.")
        else: raise HTTPException(status_code=400, detail="Either results_file or results_data must be provided.")
        if not binary_prediction_results: raise HTTPException(status_code=400, detail="No prediction results.")
        
        plot_image_path = create_probability_plot(binary_prediction_results, true_label, plot_output_dir)
        if not plot_image_path or not os.path.exists(plot_image_path):
            raise HTTPException(status_code=500, detail="Failed to create probability plot image.")
        return FileResponse(plot_image_path, media_type="image/png", filename=os.path.basename(plot_image_path))
    except NotImplementedError as e: raise HTTPException(status_code=501, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: cleanup_file(results_file_path)

# 3. Visualization
@api_router.post("/visualize/molecule")
async def api_process_molecule_visualization_v1(xyz_file: UploadFile = File(...), frame_index: int = Form(0), representation_style: str = Form("ball_and_stick"), rotation_x: float = Form(0.0), rotation_y: float = Form(0.0), rotation_z: float = Form(0.0), zoom_level: float = Form(1.0)):
    xyz_file_path = save_upload_file(xyz_file)
    # Create a unique output directory for this visualization call based on filename and timestamp/random
    viz_session_id = f"mol_viz_{os.path.splitext(os.path.basename(xyz_file.filename))[0]}_{int(frame_index)}_{representation_style}"
    output_dir = os.path.join(UPLOAD_DIR, viz_session_id)
    os.makedirs(output_dir, exist_ok=True)
    try:
        image_path, legend_path = process_molecule_visualization(xyz_file_path, frame_index, representation_style, (rotation_x, rotation_y, rotation_z), zoom_level, output_dir)
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=500, detail="Molecule visualization did not produce an image file.")
        
        # Return paths relative to UPLOAD_DIR for use with /api/v1/static_files/
        return JSONResponse(content={
            "status": "fallback_image",
            "message": "File paths are relative to a base static files directory.",
            "image_file": os.path.relpath(image_path, UPLOAD_DIR).replace(os.sep, '/'), # Path relative to UPLOAD_DIR
            "legend_file": os.path.relpath(legend_path, UPLOAD_DIR).replace(os.sep, '/') if legend_path and os.path.exists(legend_path) else None,
            # "output_directory_hint" is no longer needed if paths are correctly relative to UPLOAD_DIR for static serving
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(xyz_file_path)
        # Consider cleanup of output_dir if files are served and then no longer needed, or a TTL policy.

@api_router.post("/visualize/export_frame")
async def api_export_frame_data_v1(current_image_file: Optional[UploadFile] = File(None), current_image_server_path: Optional[str] = Form(None), output_format: str = Form(...), frame_index: int = Form(...), property_prediction_file_path: Optional[str] = Form(None), binary_prediction_file_path: Optional[str] = Form(None)):
    image_to_process_path = None
    temp_image_path = None
    if current_image_file:
        temp_image_path = save_upload_file(current_image_file)
        image_to_process_path = temp_image_path
    elif current_image_server_path:
        # Ensure current_image_server_path is within UPLOAD_DIR for security
        abs_image_server_path = os.path.abspath(os.path.join(UPLOAD_DIR, current_image_server_path.strip('/\\')))
        if not os.path.commonpath([abs_image_server_path, UPLOAD_DIR]) == UPLOAD_DIR or not os.path.exists(abs_image_server_path):
             raise HTTPException(status_code=400, detail="Invalid current_image_server_path.")
        image_to_process_path = abs_image_server_path
    else: raise HTTPException(status_code=400, detail="Either current_image_file or current_image_server_path must be provided.")

    export_session_id = f"frame_export_{frame_index}_{output_format}"
    export_output_dir = os.path.join(UPLOAD_DIR, export_session_id)
    os.makedirs(export_output_dir, exist_ok=True)
    try:
        # Ensure prediction file paths are also within UPLOAD_DIR if provided
        # This part needs careful security handling if paths are user-supplied.
        # For now, assuming they are opaque identifiers or already validated if used.
        # The subtask omitted sending these from frontend, so these will be None.
        
        output_image_path, output_json_path = export_frame_data(image_to_process_path, output_format, frame_index, property_prediction_file_path, binary_prediction_file_path, export_output_dir)
        if not output_image_path or not os.path.exists(output_image_path): raise HTTPException(status_code=500, detail="Frame export did not produce an image file.")
        if not output_json_path or not os.path.exists(output_json_path): raise HTTPException(status_code=500, detail="Frame export did not produce a JSON data file.")
        
        return JSONResponse(content={
            "exported_image_file": os.path.relpath(output_image_path, UPLOAD_DIR).replace(os.sep, '/'),
            "exported_json_file": os.path.relpath(output_json_path, UPLOAD_DIR).replace(os.sep, '/'),
            "message": "Frame exported. File paths are relative to a base static files directory."
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_image_path)

# Moved /static_files endpoint to be under /api/v1/
@api_router.get("/static_files/{file_path:path}")
async def serve_api_static_file(file_path: str): # Renamed
    # This serves files from UPLOAD_DIR (e.g., visualization outputs, reactor files)
    # Security: Ensure file_path does not allow directory traversal outside UPLOAD_DIR
    # os.path.join will handle this if file_path is relative.
    # For absolute paths or ".." in file_path, more checks are needed.
    # FastAPI's StaticFiles is generally safer for serving directories.
    # Here, we construct the path manually, so care is needed.
    
    # Normalize file_path to prevent directory traversal
    # e.g. path = PurePath(file_path).name # only allows filename
    # or more complex normalization if subdirectories within UPLOAD_DIR are needed.
    
    # A safer way to join and check:
    full_path = os.path.abspath(os.path.join(UPLOAD_DIR, file_path.strip('/\\')))
    if not os.path.commonpath([full_path, UPLOAD_DIR]) == UPLOAD_DIR: # Check if path is still within UPLOAD_DIR
        raise HTTPException(status_code=404, detail="File not found (invalid path).")

    if not os.path.exists(full_path) or not os.path.isfile(full_path): # Check if it's a file
        raise HTTPException(status_code=404, detail="File not found.")
        
    media_type = "application/octet-stream"
    if "." in file_path:
        ext = file_path.split(".")[-1].lower()
        if ext == "png": media_type = "image/png"
        elif ext in ["jpg", "jpeg"]: media_type = "image/jpeg"
        elif ext in ["txt", "log"]: media_type = "text/plain"
        elif ext == "json": media_type = "application/json"
        elif ext == "csv": media_type = "text/csv"
        elif ext == "zip": media_type = "application/zip"
    return FileResponse(full_path, media_type=media_type)

# 4. Chat Functionality
@api_router.post("/chat/configure")
async def api_chat_configure_v1(base_url: Optional[str] = Form(None), api_key: Optional[str] = Form(None)):
    try:
        init_status = chat_interface.initialize_client(base_url=base_url, api_key=api_key)
        return {"status": "success", "message": f"Chat client initialized: {init_status}"}
    except NotImplementedError as e: raise HTTPException(status_code=501, detail=str(e))
    except Exception as e: return JSONResponse(status_code=500, content={"error": f"Failed to configure chat: {str(e)}"})

@api_router.post("/chat/message")
async def api_chat_message_v1(user_message: str = Form(...), image_file: Optional[UploadFile] = File(None), multimodal_flag: bool = Form(False), chat_history_json: str = Form("[]"), model_selection: Optional[str] = Form(None)):
    image_path = None
    try:
        if image_file: image_path = save_upload_file(image_file)
        try: chat_history = json.loads(chat_history_json)
        except ValueError: raise HTTPException(status_code=400, detail="Invalid chat_history_json.")
        
        result = chat_interface.process_message(user_message, image_path, multimodal_flag, chat_history, model_selection)
        if isinstance(result, tuple) and len(result) == 2:
            updated_history, error_message = result
            if error_message: return JSONResponse(status_code=200, content={"chat_history": updated_history, "error": error_message, "detail": "Message processed with error/warning."})
            return {"chat_history": updated_history, "error": None}
        elif isinstance(result, dict) and "chat_history" in result: return result
        else: raise HTTPException(status_code=500, detail="Chat processing returned unexpected data.")
    except NotImplementedError as e: raise HTTPException(status_code=501, detail=str(e))
    except Exception as e: return JSONResponse(status_code=500, content={"error": f"Failed to process message: {str(e)}"})
    finally: cleanup_file(image_path)

# 5. Nano Reactor
@api_router.post("/reactor/upload_files")
async def api_reactor_upload_files_v1(job_files: List[UploadFile] = File(...), job_id: str = Form(...)):
    saved_file_paths = []
    try:
        for job_file in job_files: saved_file_paths.append(save_upload_file(job_file))
        status = reactor.handle_uploaded_files(job_id, saved_file_paths, UPLOAD_DIR) 
        return {"status": "success", "message": f"Files uploaded for job {job_id}. Handler status: {status}", "uploaded_files": [f.filename for f in job_files]}
    except NotImplementedError as e: raise HTTPException(status_code=501, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/reactor/run_extra")
async def api_reactor_run_extra_v1(param3_file: UploadFile = File(...), job_id: str = Form(...)):
    param3_file_path = save_upload_file(param3_file)
    reactor_output_dir = os.path.join(UPLOAD_DIR, f"reactor_output_{job_id}")
    os.makedirs(reactor_output_dir, exist_ok=True)
    try:
        # Assuming reactor.run_extra adapted for API or is dummy
        result = reactor.run_extra(param3_file_path, job_id, None, reactor_output_dir) 
        # Ideal response structure for real implementation:
        if isinstance(result, tuple) and len(result) == 3:
            log_content, plot_path, zip_path = result
            return {
                "status": "success", "log_content": log_content,
                "plot_file": os.path.relpath(plot_path, UPLOAD_DIR).replace(os.sep, '/') if plot_path else None,
                "results_file": os.path.relpath(zip_path, UPLOAD_DIR).replace(os.sep, '/') if zip_path else None,
                "message": f"Run extra for job {job_id} processed."
            }
        # Fallback for current dummy which returns string in result
        return {"status": "success", "message": f"Run extra for job {job_id} processed. Result: {result}"}
    except NotImplementedError as e: raise HTTPException(status_code=501, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: cleanup_file(param3_file_path)

@api_router.post("/reactor/run_net")
async def api_reactor_run_net_v1(job_id: str = Form(...)):
    reactor_output_dir = os.path.join(UPLOAD_DIR, f"reactor_output_{job_id}")
    os.makedirs(reactor_output_dir, exist_ok=True)
    try:
        result = reactor.run_net(job_id, None, reactor_output_dir) 
        if isinstance(result, tuple) and len(result) == 3:
            html_content, timeline_plot, species_json = result
            return {
                "status": "success", "analysis_html": html_content,
                "timeline_plot_file": os.path.relpath(timeline_plot, UPLOAD_DIR).replace(os.sep, '/') if timeline_plot else None,
                "species_data_file": os.path.relpath(species_json, UPLOAD_DIR).replace(os.sep, '/') if species_json else None, # Path for next step
                "message": f"Run net for job {job_id} processed."
            }
        return {"status": "success", "message": f"Run net for job {job_id} processed. Result: {result}"}
    except NotImplementedError as e: raise HTTPException(status_code=501, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/reactor/extract_fragments")
async def api_reactor_extract_fragments_v1(job_id: str = Form(...), timestep_input: str = Form(...), species_data_json_path: str = Form(...)):
    reactor_output_dir = os.path.join(UPLOAD_DIR, f"reactor_output_{job_id}")
    os.makedirs(reactor_output_dir, exist_ok=True)
    
    # species_data_json_path is expected to be relative to UPLOAD_DIR, e.g., "reactor_output_JOBID/species_data.json"
    abs_species_path = os.path.abspath(os.path.join(UPLOAD_DIR, species_data_json_path.strip('/\\')))
    if not os.path.commonpath([abs_species_path, UPLOAD_DIR]) == UPLOAD_DIR or not os.path.exists(abs_species_path):
        raise HTTPException(status_code=400, detail=f"Invalid species_data_json_path: {species_data_json_path}")

    try:
        result = reactor.on_timestep_select(job_id, timestep_input, abs_species_path, reactor_output_dir)
        if isinstance(result, tuple) and len(result) == 2:
            status_msg, fragments_zip = result
            return {
                "status": "success", "status_msg": status_msg,
                "fragment_archive_file": os.path.relpath(fragments_zip, UPLOAD_DIR).replace(os.sep, '/') if fragments_zip else None,
                "message": f"Fragment extraction for job {job_id} processed."
            }
        return {"status": "success", "message": f"Fragment extraction for job {job_id} processed. Result: {result}"}
    except NotImplementedError as e: raise HTTPException(status_code=501, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# Include the API router in the main app
app.include_router(api_router)

# --- Static Frontend Serving ---
# Path to the directory where api.py is located (src/)
# Path to the frontend build directory (assumed to be ../frontend/dist relative to src/)
FRONTEND_DIST_DIR = os.path.join(APP_ROOT_DIR, "..", "frontend", "dist")

# Mount the 'assets' directory from the frontend build first
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST_DIR, "assets")), name="static_assets")

# Catch-all route to serve index.html for SPA, or other static files from root of dist
@app.get("/{full_path:path}", response_class=FileResponse, include_in_schema=False)
async def serve_react_app(full_path: str):
    file_path = os.path.join(FRONTEND_DIST_DIR, full_path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    index_html_path = os.path.join(FRONTEND_DIST_DIR, "index.html")
    if os.path.exists(index_html_path):
        return FileResponse(index_html_path)
    # Fallback if index.html is also missing for some reason (should not happen in a built app)
    raise HTTPException(status_code=404, detail="SPA Index.html not found.")


if __name__ == "__main__":
    # UPLOAD_DIR will be created relative to where this script is executed.
    # If `python src/api.py` is run, UPLOAD_DIR will be `src/../temp_uploads_api` -> `../temp_uploads_api` from project root.
    # This is fine.
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # To run for development with reload, from project root:
    # python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
    # Or from src/:
    # python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
