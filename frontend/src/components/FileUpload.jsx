import React, { useState } from 'react';

// Note: Global styles from index.css will apply to input and button.
// This component itself doesn't need much specific styling if global styles are comprehensive.

function FileUpload({ onFileUpload, label = "Upload File", disabled = false, accept }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setFileName(file.name);
      if (onFileUpload) {
        onFileUpload(file); // Pass the file object up immediately on selection
      }
    } else {
      setSelectedFile(null);
      setFileName('');
      if (onFileUpload) {
        onFileUpload(null); // Pass null if selection is cleared
      }
    }
  };

  // The actual "upload" action is now typically handled by a parent component's button.
  // This component focuses on file selection.
  // If a dedicated button for THIS component to trigger something is needed, it can be added.
  // For now, it's primarily a styled file input.

  return (
    <div className="file-upload-container">
      <label htmlFor={`file-upload-input-${label.replace(/\s+/g, '-')}`} className="file-upload-label">
        {/* The button styling is applied to the label which acts as a button */}
        <span className="button primary">{label}</span>
        <input 
          type="file" 
          id={`file-upload-input-${label.replace(/\s+/g, '-')}`} 
          onChange={handleFileChange} 
          disabled={disabled}
          accept={accept}
          style={{ display: 'none' }} // Hide the default input
        />
      </label>
      {fileName && <span className="file-name-display">Selected: {fileName}</span>}
    </div>
  );
}

export default FileUpload;

// Add some specific styles for FileUpload if needed, or rely on index.css
// Example: Add to a new file FileUpload.css and import it, or add to index.css:
/*
.file-upload-container {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-md); / Ensure it has some bottom margin /
}

.file-upload-label .button {
  padding: var(--spacing-sm) var(--spacing-md); / Slightly smaller padding for this button context /
}

.file-name-display {
  font-size: 0.9em;
  color: var(--text-muted-color);
  background-color: #e9ecef; / Light background for the filename /
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-color);
}
*/
