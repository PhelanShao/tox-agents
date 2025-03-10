#utils.py
import os
import zipfile

def create_job_zip(job_id, base_dir):
    """Create a zip file containing all job-related files"""
    try:
        job_dir = os.path.join(base_dir, job_id)
        if not os.path.exists(job_dir):
            return None
            
        zip_path = os.path.join(job_dir, f"{job_id}_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for root, dirs, files in os.walk(job_dir):
                for file in files:
                    if file.endswith('.zip'):
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, job_dir)
                    zf.write(file_path, arcname)
                    
        return zip_path
    except Exception as e:
        print(f"Error creating zip file: {str(e)}")
        return None