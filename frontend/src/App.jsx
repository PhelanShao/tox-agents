import React, { useState } from 'react';
import './App.css'; // Using App.css for component-specific styles if needed
import XYZtoNPZConverter from './components/XYZtoNPZConverter';
import ToxicityPrediction from './components/ToxicityPrediction';
import NanoReactor from './components/NanoReactor';

function App() {
  const [activeTab, setActiveTab] = useState('xyzToNpz'); // Default tab

  const renderTabContent = () => {
    switch (activeTab) {
      case 'xyzToNpz':
        return <XYZtoNPZConverter />;
      case 'toxicity':
        return <ToxicityPrediction />;
      case 'reactor':
        return <NanoReactor />;
      default:
        return <XYZtoNPZConverter />;
    }
  };

  return (
    <div className="App">
      <nav className="tab-nav">
        <button
          onClick={() => setActiveTab('xyzToNpz')}
          className={activeTab === 'xyzToNpz' ? 'active' : ''}
        >
          XYZ to NPZ Converter
        </button>
        <button
          onClick={() => setActiveTab('toxicity')}
          className={activeTab === 'toxicity' ? 'active' : ''}
        >
          Toxicity Prediction & Visualization
        </button>
        <button
          onClick={() => setActiveTab('reactor')}
          className={activeTab === 'reactor' ? 'active' : ''}
        >
          Nano Reactor
        </button>
      </nav>
      <div className="tab-content">
        {renderTabContent()}
      </div>
    </div>
  );
}

export default App;
