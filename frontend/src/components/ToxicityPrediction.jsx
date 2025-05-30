import React, { useState, useEffect, useCallback } from 'react';
import FileUpload from './FileUpload';
import ChatInterface from './ChatInterface'; 

const API_V1_BASE_URL = '/api/v1'; // Use relative path for API calls
const API_V1_STATIC_FILES_URL = '/api/v1/static_files'; // For files served from UPLOAD_DIR

function ToxicityPrediction() {
  const [npzFile, setNpzFile] = useState(null);
  const [convertedXyzFile, setConvertedXyzFile] = useState(null); 
  
  const [propertyModelPath, setPropertyModelPath] = useState('/mnt/backup2/ai4s/backupunimolpy/MD_model');
  const [propertyRefPath, setPropertyRefPath] = useState('/mnt/backup2/ai4s/unimolpy/refscale.npz');
  const [binaryModelPath, setBinaryModelPath] = useState('/mnt/backup2/ai4s/backupunimolpy/ToxPred_modelmini');

  const [isLoadingProperty, setIsLoadingProperty] = useState(false);
  const [isLoadingBinary, setIsLoadingBinary] = useState(false);
  const [isLoadingConversion, setIsLoadingConversion] = useState(false);
  const [isLoadingVisualization, setIsLoadingVisualization] = useState(false);
  const [isLoadingProbabilityPlot, setIsLoadingProbabilityPlot] = useState(false);
  const [isLoadingExportFrame, setIsLoadingExportFrame] = useState(false);
  
  const [propertyResult, setPropertyResult] = useState(null); 
  const [binaryResult, setBinaryResult] = useState(null); 
  const [binaryPredictionCsvFile, setBinaryPredictionCsvFile] = useState(null); 
  const [conversionMessage, setConversionMessage] = useState('');
  
  const [error, setError] = useState(''); 
  const [probabilityPlotError, setProbabilityPlotError] = useState('');

  const [vizFrameIndex, setVizFrameIndex] = useState(0);
  const [vizRepresentation, setVizRepresentation] = useState('sticks');
  const [vizRotationX, setVizRotationX] = useState(0);
  const [vizRotationY, setVizRotationY] = useState(0);
  const [vizRotationZ, setVizRotationZ] = useState(0);
  const [vizZoom, setVizZoom] = useState(1.0);
  const [vizImageUrl, setVizImageUrl] = useState(''); 
  const [vizLegendContent, setVizLegendContent] = useState('');
  const [probabilityPlotUrl, setProbabilityPlotUrl] = useState('');

  const [exportedImageForAnalysis, setExportedImageForAnalysis] = useState(null); 
  const [exportedJsonForAnalysis, setExportedJsonForAnalysis] = useState(null); 
  const [exportFrameMessage, setExportFrameMessage] = useState('');

  const handleFileSelected = (file) => {
    setNpzFile(file); setConvertedXyzFile(null); setError(''); 
    setPropertyResult(null); setBinaryResult(null); setBinaryPredictionCsvFile(null); 
    setConversionMessage(''); setVizImageUrl(''); setVizLegendContent('');
    setProbabilityPlotUrl(''); setProbabilityPlotError('');
    setExportedImageForAnalysis(null); setExportedJsonForAnalysis(null); setExportFrameMessage('');
  };

  const getStaticFileUrl = (relativePathFromUploadDir) => {
    if (!relativePathFromUploadDir) return '';
    // API now returns paths like "mol_viz_output_filename/image.png"
    // which are already relative to UPLOAD_DIR.
    return `${API_V1_STATIC_FILES_URL}/${relativePathFromUploadDir}`;
  };

  const runPrediction = async (type) => {
    if (!npzFile) { setError('Please select an NPZ file first.'); return; }
    setError(''); setProbabilityPlotUrl(''); setProbabilityPlotError('');

    let url = '';
    let formData = new FormData();
    formData.append('npz_file', npzFile);

    if (type === 'property') {
      setIsLoadingProperty(true); setPropertyResult(null);
      url = `${API_V1_BASE_URL}/predict/property`;
      formData.append('model_path', propertyModelPath);
      formData.append('reference_path', propertyRefPath);
    } else if (type === 'binary') {
      setIsLoadingBinary(true); setBinaryResult(null); setBinaryPredictionCsvFile(null); 
      url = `${API_V1_BASE_URL}/predict/binary`;
      formData.append('model_path', binaryModelPath);
    } else { return; }

    try {
      const response = await fetch(url, { method: 'POST', body: formData });
      if (response.ok) {
        const contentDisposition = response.headers.get('content-disposition');
        let filename = type === 'property' ? 'property_prediction.csv' : 'binary_prediction.csv'; 
        if (contentDisposition) {
          const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
          if (filenameMatch && filenameMatch[1]) filename = filenameMatch[1];
        }
        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const fileObject = new File([blob], filename, { type: 'text/csv' });
        if (type === 'property') {
          setPropertyResult({ message: `Property prediction successful! File "${filename}" ready.`, downloadUrl, filename, fileObject });
        } else {
          setBinaryPredictionCsvFile(fileObject);
          setBinaryResult({ message: `Binary prediction successful! File "${filename}" ready.`, downloadUrl, filename, fileObject });
        }
      } else {
        const errorData = await response.json().catch(() => ({ detail: `Unknown error during ${type} prediction.` }));
        const errorMessage = errorData.detail || `Prediction failed for ${type}.`;
        if (type === 'property') setPropertyResult({ message: errorMessage, fileObject: null });
        else setBinaryResult({ message: errorMessage, fileObject: null }); 
        setError(prev => `${prev}\n${type} prediction error: ${errorMessage}`); 
      }
    } catch (err) {
      const errorMessage = `Network error: ${err.message || 'Could not connect to the server.'}`;
      if (type === 'property') setPropertyResult({ message: errorMessage, fileObject: null });
      else setBinaryResult({ message: errorMessage, fileObject: null }); 
      setError(prev => `${prev}\n${type} prediction error: ${errorMessage}`);
    } finally {
      if (type === 'property') setIsLoadingProperty(false);
      else setIsLoadingBinary(false);
    }
  };
  
  const handleRunAllPredictions = () => {
    if (!npzFile) { setError('Please select an NPZ file first.'); return; }
    runPrediction('property').then(() => { if(npzFile) { runPrediction('binary'); }});
  };

  const handleConvertAndPrepareViz = async () => {
    if (!npzFile) { setError('Please select an NPZ file first for conversion.'); return; }
    setIsLoadingConversion(true); setConversionMessage(''); setError('');
    setConvertedXyzFile(null); setVizImageUrl(''); setVizLegendContent('');
    const formData = new FormData();
    formData.append('npz_file', npzFile);
    try {
      const response = await fetch(`${API_V1_BASE_URL}/convert/npz-to-xyz`, { method: 'POST', body: formData });
      if (response.ok) {
        const contentDisposition = response.headers.get('content-disposition');
        let xyzFilename = 'converted.xyz'; 
        if (contentDisposition) {
          const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
          if (filenameMatch && filenameMatch[1]) xyzFilename = filenameMatch[1];
        }
        const blob = await response.blob();
        const xyzFileObject = new File([blob], xyzFilename, { type: 'text/plain' });
        setConvertedXyzFile(xyzFileObject);
        setConversionMessage(`NPZ successfully converted to ${xyzFilename}. Ready for visualization.`);
        await fetchVisualization(xyzFileObject); 
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error during NPZ to XYZ conversion.' }));
        setError(`NPZ to XYZ Conversion failed: ${errorData.detail || 'Please try again.'}`);
        setConversionMessage('');
      }
    } catch (err) {
      setError(`Network error during conversion: ${err.message || 'Could not connect.'}`);
      setConversionMessage('');
    } finally {
      setIsLoadingConversion(false);
    }
  };

  const fetchVisualization = async (xyzFileToVisualize) => {
    if (!xyzFileToVisualize) { setError('No XYZ file. Convert NPZ to XYZ first.'); return; }
    setIsLoadingVisualization(true); setError(''); setVizImageUrl(''); setVizLegendContent('');
    const formData = new FormData();
    formData.append('xyz_file', xyzFileToVisualize);
    formData.append('frame_index', vizFrameIndex);
    formData.append('representation_style', vizRepresentation);
    formData.append('rotation_x', vizRotationX); formData.append('rotation_y', vizRotationY);
    formData.append('rotation_z', vizRotationZ); formData.append('zoom_level', vizZoom);

    try {
      const response = await fetch(`${API_V1_BASE_URL}/visualize/molecule`, { method: 'POST', body: formData });
      if (response.ok) {
        const data = await response.json(); // API returns {"image_file": "path/to/image.png", "legend_file": "path/to/legend.txt"}
                                          // These paths are relative to UPLOAD_DIR
        if (data.image_file) {
          setVizImageUrl(getStaticFileUrl(data.image_file));
          if (data.legend_file) {
            try {
                const legendResponse = await fetch(getStaticFileUrl(data.legend_file));
                if (legendResponse.ok) setVizLegendContent(await legendResponse.text());
                else setVizLegendContent('Could not load legend.');
            } catch (legendErr) { setVizLegendContent('Error fetching legend.'); }
          } else { setVizLegendContent('No legend provided.'); }
        } else { setError('Visualization data incomplete from API.'); }
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error during visualization.' }));
        setError(`Visualization failed: ${errorData.detail || 'Please try again.'}`);
      }
    } catch (err) {
      setError(`Network error during visualization: ${err.message || 'Could not connect.'}`);
    } finally {
      setIsLoadingVisualization(false);
    }
  };
  
  const handleUpdateVisualization = () => {
    if (convertedXyzFile) fetchVisualization(convertedXyzFile);
    else setError("Please convert NPZ to XYZ first.");
  };

  const handleGenerateProbabilityPlot = async () => {
    if (!binaryPredictionCsvFile) { setProbabilityPlotError('Binary prediction CSV file not available.'); return; }
    setIsLoadingProbabilityPlot(true); setProbabilityPlotError(''); setProbabilityPlotUrl('');
    const formData = new FormData();
    formData.append('results_file', binaryPredictionCsvFile); 
    try {
      const response = await fetch(`${API_V1_BASE_URL}/predict/probability_plot`, { method: 'POST', body: formData });
      if (response.ok) {
        const blob = await response.blob();
        setProbabilityPlotUrl(window.URL.createObjectURL(blob));
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error during plot generation.' }));
        setProbabilityPlotError(`Plot generation failed: ${errorData.detail || 'Please try again.'}`);
      }
    } catch (err) {
      setProbabilityPlotError(`Network error: ${err.message || 'Could not connect.'}`);
    } finally {
      setIsLoadingProbabilityPlot(false);
    }
  };

  const handleExportFrameForAnalysis = async () => {
    if (!vizImageUrl) { setExportFrameMessage('Error: No visualization image available.'); return; }
    setIsLoadingExportFrame(true); setExportFrameMessage('');
    setExportedImageForAnalysis(null); setExportedJsonForAnalysis(null);

    const formData = new FormData();
    formData.append('frame_index', vizFrameIndex);
    formData.append('output_format', 'png'); 
    try {
        const imageResponse = await fetch(vizImageUrl); // vizImageUrl is already a full /api/v1/static_files/... URL or blob URL
        if (!imageResponse.ok) throw new Error('Failed to fetch visualization image for export.');
        const imageBlob = await imageResponse.blob();
        formData.append('current_image_file', imageBlob, 'visualization_export.png');
    } catch (e) {
        setExportFrameMessage(`Error fetching image for export: ${e.message}`);
        setIsLoadingExportFrame(false); return;
    }
    
    try {
        const response = await fetch(`${API_V1_BASE_URL}/visualize/export_frame`, { method: 'POST', body: formData });
        const data = await response.json(); // API returns {"exported_image_file": "path/img.png", "exported_json_file": "path/data.json"}
                                          // These paths are relative to UPLOAD_DIR
        if (response.ok) {
            setExportFrameMessage(data.message || 'Frame exported successfully.');
            if (data.exported_image_file) {
                try {
                    const imageUrl = getStaticFileUrl(data.exported_image_file);
                    const imageRes = await fetch(imageUrl);
                    if (!imageRes.ok) throw new Error(`Fetch exported image: ${imageRes.statusText}`);
                    const imageBlobVal = await imageRes.blob();
                    setExportedImageForAnalysis({ file: new File([imageBlobVal], data.exported_image_file), name: data.exported_image_file });
                } catch (e) { setExportFrameMessage(`Error fetching exported image: ${e.message}`); }
            }
            if (data.exported_json_file) {
                try {
                    const jsonUrl = getStaticFileUrl(data.exported_json_file);
                    const jsonRes = await fetch(jsonUrl);
                    if (!jsonRes.ok) throw new Error(`Fetch exported JSON: ${jsonRes.statusText}`);
                    setExportedJsonForAnalysis({ content: await jsonRes.json(), name: data.exported_json_file });
                } catch (e) { setExportFrameMessage(`Error fetching exported JSON: ${e.message}`); }
            }
        } else { setExportFrameMessage(`Error: ${data.detail || 'Export failed.'}`); }
    } catch (err) {
        setExportFrameMessage(`Error: ${err.message || 'Network error during export.'}`);
    } finally {
        setIsLoadingExportFrame(false);
    }
  };
  
  const clearAnalysisDataCallback = useCallback(() => {
    setExportedImageForAnalysis(null);
    setExportedJsonForAnalysis(null);
  }, []);

  // --- JSX Structure ---
  return (
    <div> 
      <div className="component-section">
        <h3>1. Upload NPZ File</h3>
        <FileUpload 
            onFileUpload={handleFileSelected} 
            label="Select .npz File" 
            accept=".npz"
            disabled={isLoadingProperty || isLoadingBinary || isLoadingConversion || isLoadingVisualization || isLoadingProbabilityPlot || isLoadingExportFrame}
        />
      </div>
      
      {error && <p className="message error mt-2">{error}</p>}

      <div className="layout-columns">
        <div className="layout-column"> {/* Left Column */}
          <div className="component-section">
              <h3>2. Run Predictions (Optional)</h3>
              <div><label htmlFor="propertyModelPath">Property Prediction Model Path: </label><input type="text" id="propertyModelPath" value={propertyModelPath} onChange={(e) => setPropertyModelPath(e.target.value)} /></div>
              <div><label htmlFor="propertyRefPath">Property Prediction Reference NPZ Path: </label><input type="text" id="propertyRefPath" value={propertyRefPath} onChange={(e) => setPropertyRefPath(e.target.value)} /></div>
              <div><label htmlFor="binaryModelPath">Binary Prediction Model Path: </label><input type="text" id="binaryModelPath" value={binaryModelPath} onChange={(e) => setBinaryModelPath(e.target.value)} /></div>
              <button onClick={handleRunAllPredictions} disabled={isLoadingProperty || isLoadingBinary || !npzFile} className="mt-2">
                  {isLoadingProperty || isLoadingBinary ? <><span className="loader-button"></span> Processing...</> : 'Run All Predictions'}
              </button>
              {isLoadingProperty && <div><div className="loader"></div><p>Loading Property Prediction...</p></div>}
              {propertyResult && ( <div className={`message ${propertyResult.downloadUrl ? 'success' : 'error'} mt-2`}><p>{propertyResult.message}</p>{propertyResult.downloadUrl && <a href={propertyResult.downloadUrl} download={propertyResult.filename} className="button secondary">Download {propertyResult.filename}</a>}</div>)}
              {isLoadingBinary && <div><div className="loader"></div><p>Loading Binary Prediction...</p></div>}
              {binaryResult && (<div className={`message ${binaryResult.downloadUrl ? 'success' : 'error'} mt-2`}><p>{binaryResult.message}</p>{binaryResult.downloadUrl && <a href={binaryResult.downloadUrl} download={binaryResult.filename} className="button secondary">Download {binaryResult.filename}</a>}</div>)}
          </div>

          <div className="component-section">
            <h3>4. Probability Analysis Plot</h3>
            <button onClick={handleGenerateProbabilityPlot} disabled={!binaryPredictionCsvFile || isLoadingProbabilityPlot}>
              {isLoadingProbabilityPlot ? <><span className="loader-button"></span> Generating Plot...</> : 'Generate Probability Plot'}
            </button>
            {isLoadingProbabilityPlot && <div><div className="loader"></div><p>Generating Plot...</p></div>}
            {probabilityPlotError && <p className="message error mt-2">{probabilityPlotError}</p>}
            {probabilityPlotUrl && (<div className="mt-2"><h4>Probability Plot</h4><img src={probabilityPlotUrl} alt="Probability Analysis Plot" style={{ maxWidth: '100%', border: '1px solid var(--border-color)', borderRadius: 'var(--border-radius)' }} /></div>)}
          </div>
        </div>

        <div className="layout-column"> {/* Right Column */}
          <div className="component-section">
            <h3>3. Molecule Visualization</h3>
            <button onClick={handleConvertAndPrepareViz} disabled={!npzFile || isLoadingConversion}>
                {isLoadingConversion ? <><span className="loader-button"></span> Converting...</> : 'Convert NPZ to XYZ & Prepare Visualization'}
            </button>
            {isLoadingConversion && <div><div className="loader"></div><p>Converting...</p></div>}
            {conversionMessage && <p className="message info mt-2">{conversionMessage}</p>}
            {convertedXyzFile && (
                <div className="mt-2">
                    <h4>Visualization Controls</h4>
                    <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--spacing-md)'}}>
                        <div><label>Frame Index: </label><input type="number" value={vizFrameIndex} onChange={e => setVizFrameIndex(parseInt(e.target.value, 10) || 0)} min="0" /></div>
                        <div><label>Representation Style: </label><select value={vizRepresentation} onChange={e => setVizRepresentation(e.target.value)}><option value="sticks">Sticks</option><option value="ball_and_stick">Ball and Stick</option><option value="spacefill">Spacefill</option><option value="wireframe">Wireframe</option><option value="surface">Surface</option></select></div>
                        <div><label>Rotation X: </label><input type="number" value={vizRotationX} onChange={e => setVizRotationX(parseFloat(e.target.value))} step="5" /></div>
                        <div><label>Rotation Y: </label><input type="number" value={vizRotationY} onChange={e => setVizRotationY(parseFloat(e.target.value))} step="5" /></div>
                        <div><label>Rotation Z: </label><input type="number" value={vizRotationZ} onChange={e => setVizRotationZ(parseFloat(e.target.value))} step="5" /></div>
                        <div><label>Zoom: </label><input type="number" value={vizZoom} onChange={e => setVizZoom(parseFloat(e.target.value))} step="0.1" min="0.1" max="5.0" /></div>
                    </div>
                    <button onClick={handleUpdateVisualization} disabled={isLoadingVisualization || !convertedXyzFile} className="mt-2">
                        {isLoadingVisualization ? <><span className="loader-button"></span> Updating...</> : 'Update Visualization'}
                    </button>
                    <button onClick={handleExportFrameForAnalysis} disabled={isLoadingExportFrame || !vizImageUrl} className="mt-2 secondary" style={{marginLeft: 'var(--spacing-sm)'}}>
                        {isLoadingExportFrame ? <><span className="loader-button"></span> Exporting...</> : 'Export Frame for Analysis'}
                    </button>
                    {exportFrameMessage && <p className={`message ${exportFrameMessage.startsWith('Error:') ? 'error' : 'success'} mt-2`}>{exportFrameMessage}</p>}
                </div>
            )}
            {isLoadingVisualization && <div><div className="loader"></div><p>Loading Visualization...</p></div>}
            {vizImageUrl && (<div className="mt-2"><h4>Molecule Image</h4><img src={vizImageUrl} alt="Molecule Visualization" style={{ maxWidth: '100%', border: '1px solid var(--border-color)', borderRadius: 'var(--border-radius)' }} /></div>)}
            {vizLegendContent && (<div className="mt-2"><h4>Color Legend</h4><pre>{vizLegendContent}</pre></div>)}
          </div>
        </div>
      </div>

      <div className="component-section">
        <ChatInterface 
          exportedImageForAnalysis={exportedImageForAnalysis}
          exportedJsonForAnalysis={exportedJsonForAnalysis}
          onAnalysisClick={handleExportFrameForAnalysis} 
          clearAnalysisData={clearAnalysisDataCallback}
        />
      </div>
    </div>
  );
}

export default ToxicityPrediction;
