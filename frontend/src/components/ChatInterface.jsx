import React, { useState, useEffect, useCallback } from 'react';
import FileUpload from './FileUpload'; 

// Use relative paths for API calls, assuming frontend is served from the same origin as backend
const CHAT_API_BASE_URL = '/chat'; 

const DEFAULT_MODELS = [
  "google/gemini-2.0-flash-thinking-exp:free",
  "mistralai/mistral-7b-instruct:free",
  "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
  "custom" 
];

function ChatInterface({ exportedImageForAnalysis, exportedJsonForAnalysis, onAnalysisClick, clearAnalysisData }) {
  const [baseUrl, setBaseUrl] = useState('https://openrouter.ai/api/v1');
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODELS[0]);
  const [customModel, setCustomModel] = useState('');
  const [configStatus, setConfigStatus] = useState('');
  const [isConfiguring, setIsConfiguring] = useState(false);

  const [chatHistory, setChatHistory] = useState([]); 
  const [userInput, setUserInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState('');

  const [enableImageInput, setEnableImageInput] = useState(false);
  const [imageFile, setImageFile] = useState(null); 

  const handleConfigureApi = async () => {
    setIsConfiguring(true);
    setConfigStatus('');
    setError('');
    try {
      const formData = new FormData();
      formData.append('base_url', baseUrl);
      formData.append('api_key', apiKey);

      const response = await fetch(`${CHAT_API_BASE_URL}/configure`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setConfigStatus(data.message || 'API configured successfully.');
      } else {
        setConfigStatus(`Error: ${data.detail || data.error || 'Configuration failed.'}`);
      }
    } catch (err) {
      console.error("API Configuration error:", err);
      setConfigStatus(`Error: ${err.message || 'Network error.'}`);
    } finally {
      setIsConfiguring(false);
    }
  };

  const handleSendMessage = useCallback(async (messageText, imageFilePayload) => {
    const textToSend = messageText !== undefined ? messageText : userInput;
    const fileToSend = imageFilePayload !== undefined ? imageFilePayload : imageFile;

    if (!textToSend && !fileToSend) {
      setError('Please enter a message or upload an image.');
      return;
    }
    setIsSending(true);
    setError('');

    const currentModel = selectedModel === 'custom' ? customModel : selectedModel;
    if (!currentModel) {
        setError('Please select or enter a model name.');
        setIsSending(false);
        return;
    }

    const userMessageParts = [];
    if (textToSend) {
        userMessageParts.push({ text: textToSend });
    }
    if (enableImageInput && fileToSend) {
        userMessageParts.push({ text: `[Image: ${fileToSend.name}] (preview not shown)` });
    }
    
    const currentChatHistoryForAPI = [...chatHistory];
    setChatHistory(prev => [...prev, { role: 'user', parts: userMessageParts }]);

    const formData = new FormData();
    formData.append('user_message', textToSend || ''); 
    formData.append('chat_history_json', JSON.stringify(currentChatHistoryForAPI)); 
    formData.append('model_selection', currentModel);
    
    const isMultimodal = enableImageInput && fileToSend;
    formData.append('multimodal_flag', String(isMultimodal));
    
    if (isMultimodal) {
      formData.append('image_file', fileToSend, fileToSend.name);
    }

    try {
      const response = await fetch(`${CHAT_API_BASE_URL}/message`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      if (response.ok) {
        setChatHistory(data.chat_history || []); 
        setUserInput('');
        setImageFile(null); 
      } else {
        setError(data.error || data.detail || 'Failed to send message.');
        setChatHistory(currentChatHistoryForAPI); 
      }
    } catch (err) {
      console.error("Send message error:", err);
      setError(`Error: ${err.message || 'Network error.'}`);
      setChatHistory(currentChatHistoryForAPI); 
    } finally {
      setIsSending(false);
    }
  }, [userInput, imageFile, chatHistory, selectedModel, customModel, enableImageInput]);
  
  const handleAnalysisButtonClick = () => {
      if (onAnalysisClick) {
          onAnalysisClick(); 
      }
  };

  useEffect(() => {
    if (exportedImageForAnalysis && exportedJsonForAnalysis) {
        const analysisPrompt = `Please analyze the following exported frame data. JSON details: ${JSON.stringify(exportedJsonForAnalysis.content, null, 2)}. An image of the frame is also attached.`;
        handleSendMessage(analysisPrompt, exportedImageForAnalysis.file);
        if(clearAnalysisData) clearAnalysisData(); 
    }
  }, [exportedImageForAnalysis, exportedJsonForAnalysis, handleSendMessage, clearAnalysisData]);


  return (
    // Removed component-section from here as it's applied by parent (ToxicityPrediction)
    <> 
      <h3>AI Chat Interface</h3>
      
      <details className="mb-2">
        <summary>API Configuration</summary>
        <div style={{ padding: 'var(--spacing-md)', borderTop: '1px solid var(--border-color)', marginTop: 'var(--spacing-sm)' }}>
          <div><label htmlFor="chat-base-url">Base URL: </label><input type="text" id="chat-base-url" value={baseUrl} onChange={e => setBaseUrl(e.target.value)} /></div>
          <div><label htmlFor="chat-api-key">API Key: </label><input type="password" id="chat-api-key" value={apiKey} onChange={e => setApiKey(e.target.value)} /></div>
          <div>
            <label htmlFor="chat-model-select">Model: </label>
            <select id="chat-model-select" value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
              {DEFAULT_MODELS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            {selectedModel === 'custom' && (
              <input 
                type="text" 
                value={customModel} 
                onChange={e => setCustomModel(e.target.value)} 
                placeholder="Enter custom model name" 
                style={{marginLeft: 'var(--spacing-sm)', width: 'auto', display: 'inline-block'}}
              />
            )}
          </div>
          <button onClick={handleConfigureApi} disabled={isConfiguring} className="mt-1">
            {isConfiguring ? <span className="loader-button-small"></span> : 'Configure API'}
          </button>
          {configStatus && <p className={`message ${configStatus.startsWith('Error:') ? 'error' : 'success'} mt-1`}>{configStatus}</p>}
        </div>
      </details>

      <div className="chat-history-container">
        {chatHistory.length === 0 && <p className="chat-empty-message">Chat history is empty. Send a message to start.</p>}
        {chatHistory.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.role === 'user' ? 'user' : 'assistant'}`}>
            <div className="chat-bubble">
              {msg.parts.map((part, i) => <div key={i}>{part.text}</div>)}
            </div>
          </div>
        ))}
      </div>

      <div className="chat-input-area">
        <textarea 
          value={userInput} 
          onChange={e => setUserInput(e.target.value)} 
          placeholder="Type your message..." 
          rows="3" 
          onKeyPress={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); }}}
        />
        <div className="chat-actions">
          <button onClick={() => handleSendMessage()} disabled={isSending}>
            {isSending ? <span className="loader-button-small"></span> : 'Send'}
          </button>
          <button onClick={() => setChatHistory([])} disabled={isSending} className="secondary">Clear Chat</button>
          <label className="chat-image-toggle">
            <input type="checkbox" checked={enableImageInput} onChange={e => setEnableImageInput(e.target.checked)} />
            Attach Image
          </label>
          {enableImageInput && (
            <FileUpload onFileUpload={setImageFile} label="Choose Image" accept="image/*" disabled={isSending} />
          )}
           <button onClick={handleAnalysisButtonClick} title="Export current visualization and send for analysis" className="secondary">
              Analyze Frame
          </button>
        </div>
      </div>
      {imageFile && enableImageInput && <p className="file-name-display mt-1">Selected image: {imageFile.name}</p>}
      {error && <p className="message error mt-1">{error}</p>}
    </>
  );
}

export default ChatInterface;
