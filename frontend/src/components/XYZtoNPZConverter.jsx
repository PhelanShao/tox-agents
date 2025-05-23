import React, { useState } from 'react';
import FileUpload from './FileUpload'; 

const API_V1_BASE_URL = '/api/v1'; // Use relative path for API calls

function XYZtoNPZConverter() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [downloadFilename, setDownloadFilename] = useState('');

  const handleFileSelected = (file) => {
    setSelectedFile(file);
    setMessage('');
    setError('');
    setDownloadUrl('');
    setDownloadFilename('');
  };

  const handleConvert = async () => {
    if (!selectedFile) {
      setError('Please select a file first.');
      return;
    }

    setIsLoading(true);
    setMessage('');
    setError('');
    setDownloadUrl('');

    const formData = new FormData();
    formData.append('xyz_file', selectedFile); 

    try {
      // Updated API endpoint
      const response = await fetch(`${API_V1_BASE_URL}/convert/xyz-to-npz`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const contentDisposition = response.headers.get('content-disposition');
        let filename = 'converted.npz'; 
        if (contentDisposition) {
          const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
          if (filenameMatch && filenameMatch[1]) {
            filename = filenameMatch[1];
          }
        }
        setDownloadFilename(filename);

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        setDownloadUrl(url);
        setMessage(`Conversion successful! File "${filename}" is ready for download.`);
        setSelectedFile(null); 
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error occurred during conversion.' }));
        setError(errorData.detail || 'Conversion failed. Please try again.');
        console.error('Conversion failed:', errorData);
      }
    } catch (err) {
      console.error('Network or other error:', err);
      setError(`Network error: ${err.message || 'Could not connect to the server.'}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="component-section">
      <h3>XYZ to NPZ Converter</h3>
      <p>Upload an XYZ file to convert it to NPZ format.</p>
      
      <FileUpload 
        onFileUpload={handleFileSelected} 
        label="Select .xyz File" 
        accept=".xyz"
        disabled={isLoading} 
      />

      {selectedFile && !isLoading && (
        <div style={{marginTop: 'var(--spacing-md)'}}>
          <button onClick={handleConvert} disabled={isLoading}>
            {isLoading ? <><span className="loader-button"></span> Converting...</> : 'Convert to NPZ'}
          </button>
        </div>
      )}

      {isLoading && selectedFile && (
         <div className="mt-2">
          <div className="loader"></div>
          <p style={{textAlign: 'center'}}>Converting file...</p>
        </div>
      )}
      
      {error && <p className="message error mt-2">{error}</p>}
      {message && <p className="message success mt-2">{message}</p>}

      {downloadUrl && (
        <div className="mt-2">
          <a href={downloadUrl} download={downloadFilename} className="button primary">
            Download {downloadFilename}
          </a>
        </div>
      )}
    </div>
  );
}

export default XYZtoNPZConverter;
