import React, { useState } from 'react';

// Use relative paths for API calls
const REACTOR_API_BASE_URL = '/reactor'; // For reactor specific endpoints
const API_V1_STATIC_FILES_URL = '/api/v1/static_files'; // For accessing static files (prefixed with /api/v1)

function NanoReactor() {
  const [jobId, setJobId] = useState('');
  const [jobFiles, setJobFiles] = useState([]); 
  
  const [isLoadingUpload, setIsLoadingUpload] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(''); 

  const [param3File, setParam3File] = useState(null);
  const [isLoadingRunExtra, setIsLoadingRunExtra] = useState(false);
  const [runExtraLog, setRunExtraLog] = useState('');
  const [runExtraPlotUrl, setRunExtraPlotUrl] = useState('');
  const [runExtraResultsUrl, setRunExtraResultsUrl] = useState('');
  const [runExtraError, setRunExtraError] = useState('');

  const [isLoadingRunNet, setIsLoadingRunNet] = useState(false);
  const [runNetOutputHtml, setRunNetOutputHtml] = useState('');
  const [runNetTimelinePlotUrl, setRunNetTimelinePlotUrl] = useState('');
  const [runNetSpeciesDataJsonPath, setRunNetSpeciesDataJsonPath] = useState(''); 
  const [runNetError, setRunNetError] = useState('');

  const [timestepInput, setTimestepInput] = useState('');
  const [isLoadingExtractFragments, setIsLoadingExtractFragments] = useState(false);
  const [extractFragmentsStatus, setExtractFragmentsStatus] = useState('');
  const [extractFragmentsResultUrl, setExtractFragmentsResultUrl] = useState('');
  const [extractFragmentsError, setExtractFragmentsError] = useState('');

  const getStaticUrl = (filePath) => {
    if (!filePath) return '';
    const cleanedPath = filePath.replace(/^temp_uploads\/?/, ''); // Remove temp_uploads prefix if present
    // Example: if filePath is "job_id_hex/output_dir_hint/file.png"
    // Result: "/api/v1/static_files/job_id_hex/output_dir_hint/file.png"
    return `${API_V1_STATIC_FILES_URL}/${cleanedPath}`;
  };

  const parseDummyResultString = (messageString) => {
    if (messageString && messageString.includes("Result: [")) {
      try {
        const resultTupleStr = messageString.substring(messageString.indexOf("Result: ["));
        const actualTupleStr = resultTupleStr.substring("Result: ".length);
        const parsed = JSON.parse(actualTupleStr.replace(/'/g, '"'));
        return parsed; 
      } catch (e) {
        console.error("Error parsing result string:", e, "Original string:", messageString);
        return null;
      }
    }
    return null;
  };

  const handleJobFilesSelected = (selectedFiles) => {
    setJobFiles(Array.from(selectedFiles)); 
  };

  const handleUploadJobFiles = async () => {
    if (!jobId) {
      setUploadStatus('Error: Job ID is required.');
      return;
    }
    if (jobFiles.length === 0) {
      setUploadStatus('Error: Please select files to upload.');
      return;
    }
    setIsLoadingUpload(true);
    setUploadStatus(''); 

    const formData = new FormData();
    formData.append('job_id', jobId);
    jobFiles.forEach(file => formData.append('job_files', file));

    try {
      const response = await fetch(`${REACTOR_API_BASE_URL}/upload_files`, { method: 'POST', body: formData });
      const data = await response.json();
      if (response.ok) {
        setUploadStatus(`Success: ${data.message}. Files: ${data.uploaded_files?.join(', ')}`);
      } else {
        setUploadStatus(`Error: ${data.detail || 'Upload failed.'}`);
      }
    } catch (error) {
      setUploadStatus(`Error: ${error.message || 'Network error.'}`);
    } finally {
      setIsLoadingUpload(false);
    }
  };

  const handleParam3FileSelected = (file) => setParam3File(file);

  const handleRunExtra = async () => {
    if (!jobId || !param3File) {
      setRunExtraError("Job ID and parameters3.dat file are required.");
      return;
    }
    setIsLoadingRunExtra(true);
    setRunExtraError(''); setRunExtraLog(''); setRunExtraPlotUrl(''); setRunExtraResultsUrl('');

    const formData = new FormData();
    formData.append('job_id', jobId);
    formData.append('param3_file', param3File);

    try {
      const response = await fetch(`${REACTOR_API_BASE_URL}/run_extra`, { method: 'POST', body: formData });
      const data = await response.json();
      if (response.ok) {
        if (data.log_content || data.plot_file || data.results_file) {
          setRunExtraLog(data.log_content || "Log not available.");
          if (data.plot_file && data.output_directory_hint) setRunExtraPlotUrl(getStaticUrl(`${data.output_directory_hint}/${data.plot_file}`));
          if (data.results_file && data.output_directory_hint) setRunExtraResultsUrl(getStaticUrl(`${data.output_directory_hint}/${data.results_file}`));
        } else {
          const parsedResult = parseDummyResultString(data.message);
          if (parsedResult && parsedResult.length >= 3) {
            setRunExtraLog(parsedResult[0] || "Log not available.");
            if (parsedResult[1]) setRunExtraPlotUrl(getStaticUrl(parsedResult[1]));
            if (parsedResult[2]) setRunExtraResultsUrl(getStaticUrl(parsedResult[2]));
          } else {
            setRunExtraLog(data.message || "Run Extra completed.");
          }
        }
      } else {
        setRunExtraError(data.detail || data.error || "Run Extra failed.");
      }
    } catch (error) {
      setRunExtraError(error.message || "Network error.");
    } finally {
      setIsLoadingRunExtra(false);
    }
  };

  const handleRunNet = async () => {
    if (!jobId) {
      setRunNetError("Job ID is required.");
      return;
    }
    setIsLoadingRunNet(true);
    setRunNetError(''); setRunNetOutputHtml(''); setRunNetTimelinePlotUrl(''); setRunNetSpeciesDataJsonPath('');

    const formData = new FormData();
    formData.append('job_id', jobId);

    try {
      const response = await fetch(`${REACTOR_API_BASE_URL}/run_net`, { method: 'POST', body: formData });
      const data = await response.json();
      if (response.ok) {
        if (data.analysis_html || data.timeline_plot_file || data.species_data_file) {
          setRunNetOutputHtml(data.analysis_html || "HTML output not available.");
          if (data.timeline_plot_file && data.output_directory_hint) setRunNetTimelinePlotUrl(getStaticUrl(`${data.output_directory_hint}/${data.timeline_plot_file}`));
          if (data.species_data_file && data.output_directory_hint) setRunNetSpeciesDataJsonPath(`${data.output_directory_hint}/${data.species_data_file}`);
        } else {
          const parsedResult = parseDummyResultString(data.message);
          if (parsedResult && parsedResult.length >= 3) {
            setRunNetOutputHtml(parsedResult[0] || "HTML output not available.");
            if (parsedResult[1]) setRunNetTimelinePlotUrl(getStaticUrl(parsedResult[1]));
            if (parsedResult[2]) setRunNetSpeciesDataJsonPath(parsedResult[2]);
          } else {
            setRunNetOutputHtml(data.message || "Run Net completed.");
          }
        }
      } else {
        setRunNetError(data.detail || data.error || "Run Net failed.");
      }
    } catch (error) {
      setRunNetError(error.message || "Network error.");
    } finally {
      setIsLoadingRunNet(false);
    }
  };

  const handleExtractFragments = async () => {
    if (!jobId || !timestepInput || !runNetSpeciesDataJsonPath) {
      setExtractFragmentsError("Job ID, Timestep, and Species Data Path (from Run Net) are required.");
      return;
    }
    setIsLoadingExtractFragments(true);
    setExtractFragmentsError(''); setExtractFragmentsStatus(''); setExtractFragmentsResultUrl('');

    const formData = new FormData();
    formData.append('job_id', jobId);
    formData.append('timestep_input', timestepInput);
    formData.append('species_data_json_path', runNetSpeciesDataJsonPath);

    try {
      const response = await fetch(`${REACTOR_API_BASE_URL}/extract_fragments`, { method: 'POST', body: formData });
      const data = await response.json();
      if (response.ok) {
        if (data.status_msg || data.fragment_archive_file) {
          setExtractFragmentsStatus(data.status_msg || "Extraction completed.");
          if (data.fragment_archive_file && data.output_directory_hint) setExtractFragmentsResultUrl(getStaticUrl(`${data.output_directory_hint}/${data.fragment_archive_file}`));
        } else {
          const parsedResult = parseDummyResultString(data.message);
          if (parsedResult && parsedResult.length >= 2) {
            setExtractFragmentsStatus(parsedResult[0] || "Status not available.");
            if (parsedResult[1]) setExtractFragmentsResultUrl(getStaticUrl(parsedResult[1]));
          } else {
            setExtractFragmentsStatus(data.message || "Extraction processed.");
          }
        }
      } else {
        setExtractFragmentsError(data.detail || data.error || "Extraction failed.");
      }
    } catch (error) {
      setExtractFragmentsError(error.message || "Network error.");
    } finally {
      setIsLoadingExtractFragments(false);
    }
  };

  return (
    <div> {/* Main container for the tab */}
      <div className="component-section">
        <h3>1. Upload Files for Job</h3>
        <div>
          <label htmlFor="jobIdUpload">Job ID:</label>
          <input type="text" id="jobIdUpload" value={jobId} onChange={(e) => setJobId(e.target.value)} placeholder="Enter Job ID" />
        </div>
        <FileUpload onFileUpload={handleJobFilesSelected} label="Select Job Files (Multiple)" multiple disabled={isLoadingUpload} />
        {jobFiles.length > 0 && (
          <div className="mt-1" style={{fontSize: '0.9em'}}>
            <strong>Selected files:</strong>
            <ul>{jobFiles.map(f => <li key={f.name}>{f.name} ({f.size} bytes)</li>)}</ul>
          </div>
        )}
        <button onClick={handleUploadJobFiles} disabled={isLoadingUpload || !jobId || jobFiles.length === 0} className="mt-1">
          {isLoadingUpload ? <span className="loader-button"></span> : 'Upload Files for Job'}
        </button>
        {uploadStatus && <p className={`message ${uploadStatus.startsWith('Error:') ? 'error' : 'success'} mt-1`}>{uploadStatus}</p>}
      </div>

      <div className="component-section">
        <h3>2. Run Extra (Process Parameters)</h3>
        {!jobId && <p className="message info">Enter and use a Job ID in section 1. This section will use that Job ID.</p>}
        <FileUpload onFileUpload={handleParam3FileSelected} label="Upload parameters3.dat" accept=".dat" disabled={isLoadingRunExtra || !jobId} />
        <button onClick={handleRunExtra} disabled={isLoadingRunExtra || !jobId || !param3File} className="mt-1">
          {isLoadingRunExtra ? <span className="loader-button"></span> : 'Run Extra'}
        </button>
        {isLoadingRunExtra && <div className="loader mt-1"></div>}
        {runExtraError && <p className="message error mt-1">{runExtraError}</p>}
        {runExtraLog && <div className="mt-1"><h4>Log:</h4><pre>{runExtraLog}</pre></div>}
        {runExtraPlotUrl && <div className="mt-1"><h4>Plot:</h4><img src={runExtraPlotUrl} alt="Run Extra Plot" style={{maxWidth: '100%', border:'1px solid var(--border-color)'}} /></div>}
        {runExtraResultsUrl && <div className="mt-1"><h4>Results:</h4><a href={runExtraResultsUrl} download className="button secondary">Download Results Archive</a></div>}
      </div>

      <div className="component-section">
        <h3>3. Analyze Fragments (Run Net)</h3>
        {!jobId && <p className="message info">Uses Job ID from section 1.</p>}
        <button onClick={handleRunNet} disabled={isLoadingRunNet || !jobId} className="mt-1">
          {isLoadingRunNet ? <span className="loader-button"></span> : 'Analyze Fragments (Run Net)'}
        </button>
        {isLoadingRunNet && <div className="loader mt-1"></div>}
        {runNetError && <p className="message error mt-1">{runNetError}</p>}
        {runNetOutputHtml && <div className="mt-1"><h4>Analysis Output:</h4><div style={{border:'1px solid var(--border-color)', padding:'var(--spacing-md)', whiteSpace: 'pre-wrap', maxHeight: '400px', overflowY: 'auto'}} dangerouslySetInnerHTML={{ __html: runNetOutputHtml }} /></div>}
        {runNetTimelinePlotUrl && <div className="mt-1"><h4>Timeline Plot:</h4><img src={runNetTimelinePlotUrl} alt="Timeline Plot" style={{maxWidth: '100%', border:'1px solid var(--border-color)'}} /></div>}
        {runNetSpeciesDataJsonPath && <p className="message info mt-1" style={{fontSize: '0.8em'}}>Species data JSON path for next step: {runNetSpeciesDataJsonPath}</p>}
      </div>

      <div className="component-section">
        <h3>4. Extract Fragment Coordinates</h3>
        {!jobId && <p className="message info">Uses Job ID from section 1. Requires 'Analyze Fragments (Run Net)' to be run successfully.</p>}
        <div>
          <label htmlFor="timestepInput">Timestep(s) (e.g., "1,2,3" or "all"): </label>
          <input type="text" id="timestepInput" value={timestepInput} onChange={(e) => setTimestepInput(e.target.value)} placeholder="Enter Timesteps"/>
        </div>
        <button onClick={handleExtractFragments} disabled={isLoadingExtractFragments || !jobId || !timestepInput || !runNetSpeciesDataJsonPath} className="mt-1">
          {isLoadingExtractFragments ? <span className="loader-button"></span> : 'Extract Fragment Coordinates'}
        </button>
        {isLoadingExtractFragments && <div className="loader mt-1"></div>}
        {extractFragmentsError && <p className="message error mt-1">{extractFragmentsError}</p>}
        {extractFragmentsStatus && <p className={`message ${extractFragmentsError ? 'error' : 'info'} mt-1`}>{extractFragmentsStatus}</p>}
        {extractFragmentsResultUrl && <div className="mt-1"><h4>Fragment Archive:</h4><a href={extractFragmentsResultUrl} download className="button secondary">Download Fragments Archive</a></div>}
      </div>
    </div>
  );
}

export default NanoReactor;
