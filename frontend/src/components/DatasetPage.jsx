import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ShinyText from './ShinyText';

const DatasetPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { datasetResult, simulationPlanResult, step1Output, knowledgeGraph, kgResult, hypothesisResult } = location.state || {};
  
  const [displayedText, setDisplayedText] = useState('');
  const [isTypingComplete, setIsTypingComplete] = useState(false);
  const [isCodeLoading, setIsCodeLoading] = useState(false);
  const [isCodeReady, setIsCodeReady] = useState(false);
  const [codeResult, setCodeResult] = useState(null);
  const [viewingDataset, setViewingDataset] = useState(null); // {name, path, content}
  const [loadingDataset, setLoadingDataset] = useState(null);
  const hasCodeProcessed = useRef(false);
  const scrollContainerRef = useRef(null);
  
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

  // Fetch and display dataset file content
  const handleViewDataset = async (name, path) => {
    setLoadingDataset(name);
    try {
      // Extract relative path from full path (remove OUTPUT_FOLDER prefix)
      // Path format: /full/path/to/backend/outputs/datasets/synthetic/file.csv
      // We need: synthetic/file.csv
      const pathParts = path.split('datasets/');
      const relativePath = pathParts.length > 1 ? pathParts[1] : path.split('/').slice(-2).join('/');
      
      console.log(`[DatasetPage] DEBUG: Fetching dataset file: ${relativePath}`);
      const response = await fetch(`${API_URL}/api/datasets/${relativePath}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch dataset: ${response.statusText}`);
      }
      
      const content = await response.text();
      console.log(`[DatasetPage] DEBUG: Dataset content received (${content.length} chars)`);
      
      setViewingDataset({ name, path, content });
    } catch (err) {
      console.error(`[DatasetPage] ERROR: Failed to load dataset ${name}:`, err);
      alert(`Failed to load dataset: ${err.message}`);
    } finally {
      setLoadingDataset(null);
    }
  };

  // Parse CSV content into rows
  const parseCSV = (csvContent) => {
    const lines = csvContent.trim().split('\n');
    return lines.map(line => {
      // Simple CSV parsing (handles quoted fields)
      const values = [];
      let current = '';
      let inQuotes = false;
      
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          values.push(current.trim());
          current = '';
        } else {
          current += char;
        }
      }
      values.push(current.trim());
      return values;
    });
  };

  // Helper function to get file type from path
  const getFileType = (path) => {
    if (!path) return 'Unknown';
    const lowerPath = path.toLowerCase();
    if (lowerPath.endsWith('.xlsx') || lowerPath.endsWith('.xls')) {
      return 'Excel';
    } else if (lowerPath.endsWith('.csv')) {
      return 'CSV';
    } else if (lowerPath.endsWith('.edgelist')) {
      return 'Graph (Edge List)';
    } else if (lowerPath.endsWith('.pdb')) {
      return 'PDB Structure';
    } else if (lowerPath.endsWith('.txt')) {
      return 'Text';
    } else if (lowerPath.endsWith('.json')) {
      return 'JSON';
    }
    return 'Data File';
  };

  // Format dataset result as readable text
  const formatDatasetAsText = (datasetData) => {
    if (!datasetData) {
      console.log('[DatasetPage] DEBUG: No dataset data provided');
      return '';
    }
    
    console.log('[DatasetPage] DEBUG: Formatting dataset data:', JSON.stringify(datasetData, null, 2));
    console.log('[DatasetPage] DEBUG: Dataset type:', datasetData.dataset_type);
    console.log('[DatasetPage] DEBUG: Number of datasets:', datasetData.datasets ? Object.keys(datasetData.datasets).length : 0);
    
    let text = '';
    
    // Dataset Type
    if (datasetData.dataset_type) {
      text += `Dataset Type\n${'='.repeat(50)}\n`;
      text += `${datasetData.dataset_type}\n\n`;
    }
    
    // Reasoning
    if (datasetData.reasoning) {
      text += `Reasoning\n${'='.repeat(50)}\n`;
      text += `${datasetData.reasoning}\n\n`;
    }
    
    // Datasets
    if (datasetData.datasets && Object.keys(datasetData.datasets).length > 0) {
      text += `Generated Datasets\n${'='.repeat(50)}\n`;
      Object.entries(datasetData.datasets).forEach(([name, path], index) => {
        const fileType = getFileType(path);
        console.log(`[DatasetPage] DEBUG: Dataset ${index + 1}: ${name} (${fileType}) at ${path}`);
        text += `${index + 1}. ${name}\n`;
        text += `   Type: ${fileType}\n`;
        text += `   Path: ${path}\n`;
        text += '\n';
      });
    } else {
      console.log('[DatasetPage] DEBUG: No datasets found in datasetData');
      text += `Generated Datasets\n${'='.repeat(50)}\n`;
      text += `No datasets were generated (dataset_type: ${datasetData.dataset_type || 'none'})\n\n`;
    }
    
    return text;
  };

  // Typing animation effect
  useEffect(() => {
    if (datasetResult) {
      console.log('[DatasetPage] DEBUG: datasetResult received:', JSON.stringify(datasetResult, null, 2));
      const fullText = formatDatasetAsText(datasetResult);
      console.log('[DatasetPage] DEBUG: Formatted text length:', fullText.length);
      setDisplayedText('');
      
      let currentIndex = 0;
      const typingSpeed = 1; // Fast typing speed
      
      const typingInterval = setInterval(() => {
        if (currentIndex < fullText.length) {
          setDisplayedText(fullText.slice(0, currentIndex + 1));
          currentIndex++;
        } else {
          clearInterval(typingInterval);
          setIsTypingComplete(true);
        }
      }, typingSpeed);

      return () => clearInterval(typingInterval);
    }
  }, [datasetResult]);

  // Auto-scroll effect - scroll to bottom as text is typed
  useEffect(() => {
    if (scrollContainerRef.current && displayedText) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [displayedText]);

  // Call CodeGeneratorAgent immediately when we have the data (runs in parallel with typing)
  useEffect(() => {
    if (hasCodeProcessed.current) {
      return;
    }

    if (!simulationPlanResult) {
      return;
    }

    hasCodeProcessed.current = true;

    const generateCode = async () => {
      setIsCodeLoading(true);
      try {
        const response = await fetch(`${API_URL}/api/generate-code`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            simulation_plan: simulationPlanResult,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to generate code');
        }

        const data = await response.json();
        setCodeResult(data.result);
        setIsCodeReady(true);
      } catch (err) {
        console.error('Error generating code:', err);
        // Don't set error state, just log it - code generation is optional
      } finally {
        setIsCodeLoading(false);
      }
    };

    generateCode();
  }, [simulationPlanResult, API_URL]);

  if (!datasetResult) {
    return (
      <div className="min-h-screen w-full bg-black flex items-center justify-center">
        <div className="text-white text-xl">No dataset data available</div>
      </div>
    );
  }

  return (
    <div className="h-screen w-full bg-black relative flex flex-col overflow-hidden">
      {/* Big Window - 80% height */}
      <div className="h-[80vh] w-full flex items-center justify-center px-8 pt-8">
        <div className="w-full max-w-7xl h-full bg-black rounded-3xl p-8">
          <div className="h-full overflow-auto hide-scrollbar" ref={scrollContainerRef} style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
            <div className="text-gray-300 leading-relaxed" style={{ fontFamily: 'Courier New, monospace' }}>
              {(() => {
                const lines = displayedText.split('\n');
                
                return lines.map((line, lineIndex) => {
                  // Detect section headings
                  if (line && !line.startsWith('=')) {
                    // Check if this is a heading: first line OR next line is separator
                    let isHeading = false;
                    if (lineIndex === 0) {
                      isHeading = true;
                    } else {
                      // Look forward to find the next non-blank line
                      for (let i = lineIndex + 1; i < lines.length; i++) {
                        const nextLine = lines[i];
                        if (nextLine && nextLine.trim() !== '') {
                          if (nextLine.startsWith('=')) {
                            isHeading = true;
                          }
                          break;
                        }
                      }
                    }
                    
                    if (isHeading && !line.includes('.')) {
                      return (
                        <React.Fragment key={lineIndex}>
                          <h2 className="font-bold text-lg mt-3 mb-1" style={{ fontWeight: 'bold', color: '#00A86B' }}>{line}</h2>
                          {'\n'}
                        </React.Fragment>
                      );
                    }
                  }
                  
                  // Skip separator lines
                  if (line.startsWith('=')) {
                    return null;
                  }
                  
                  // Check if this is a dataset path line (contains "Path:")
                  const pathMatch = line.match(/^\s*Path:\s*(.+)$/);
                  if (pathMatch && datasetResult?.datasets) {
                    const path = pathMatch[1].trim();
                    // Find the dataset name for this path
                    const datasetEntry = Object.entries(datasetResult.datasets).find(([_, p]) => p === path);
                    if (datasetEntry) {
                      const [datasetName] = datasetEntry;
                      const fileType = getFileType(path);
                      const canView = fileType === 'CSV' || fileType === 'Text' || fileType === 'JSON';
                      
                      return (
                        <div key={lineIndex} className="mb-0.5 flex items-center gap-2">
                          <span>{line}</span>
                          {canView && (
                            <button
                              onClick={() => handleViewDataset(datasetName, path)}
                              disabled={loadingDataset === datasetName}
                              className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded transition-colors"
                            >
                              {loadingDataset === datasetName ? 'Loading...' : 'View'}
                            </button>
                          )}
                          {'\n'}
                        </div>
                      );
                    }
                  }
                  
                  // Regular text lines
                  if (line.trim()) {
                    return (
                      <p key={lineIndex} className="mb-0.5">
                        {line}
                        {'\n'}
                      </p>
                    );
                  }
                  
                  return <br key={lineIndex} />;
                }).filter(Boolean);
              })()}
            </div>
          </div>
        </div>
      </div>

      {/* Dataset Viewer Modal */}
      {viewingDataset && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
          onClick={() => setViewingDataset(null)}
        >
          <div 
            className="bg-gray-900 rounded-lg p-6 max-w-6xl max-h-[90vh] w-full mx-4 overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-bold text-white">{viewingDataset.name}</h3>
              <button
                onClick={() => setViewingDataset(null)}
                className="text-gray-400 hover:text-white text-2xl"
              >
                ×
              </button>
            </div>
            <div className="flex-1 overflow-auto">
              {viewingDataset.content ? (
                getFileType(viewingDataset.path) === 'CSV' ? (
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-sm text-gray-300 border-collapse">
                      <thead>
                        {(() => {
                          const rows = parseCSV(viewingDataset.content);
                          if (rows.length > 0) {
                            return (
                              <tr className="border-b border-gray-700">
                                {rows[0].map((header, idx) => (
                                  <th key={idx} className="px-4 py-2 text-left font-bold bg-gray-800">
                                    {header}
                                  </th>
                                ))}
                              </tr>
                            );
                          }
                          return null;
                        })()}
                      </thead>
                      <tbody>
                        {(() => {
                          const rows = parseCSV(viewingDataset.content);
                          return rows.slice(1).slice(0, 100).map((row, rowIdx) => (
                            <tr key={rowIdx} className="border-b border-gray-800 hover:bg-gray-800">
                              {row.map((cell, cellIdx) => (
                                <td key={cellIdx} className="px-4 py-2">
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ));
                        })()}
                      </tbody>
                    </table>
                    {parseCSV(viewingDataset.content).length > 101 && (
                      <p className="text-gray-400 text-sm mt-4 text-center">
                        Showing first 100 rows of {parseCSV(viewingDataset.content).length - 1} rows
                      </p>
                    )}
                  </div>
                ) : (
                  <pre className="text-gray-300 text-sm whitespace-pre-wrap font-mono bg-gray-800 p-4 rounded overflow-auto max-h-[70vh]">
                    {viewingDataset.content}
                  </pre>
                )
              ) : (
                <p className="text-gray-400">No content available</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Divider */}
      <div className="absolute bottom-[17vh] left-0 right-0 w-full flex justify-center px-8">
        <div className="w-full max-w-4xl border-t" style={{ borderColor: '#00A86B' }}></div>
      </div>

      {/* Status text below the window */}
      {isTypingComplete ? (
        // After typing completes, check code status
        isCodeLoading ? (
          <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
            <ShinyText 
              text="Generating Code..." 
              disabled={false} 
              speed={3} 
              className='text-2xl' 
            />
          </div>
        ) : isCodeReady ? (
          <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
            <div 
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => {
                if (isCodeReady && codeResult) {
                  navigate('/code', { 
                    state: { 
                      codeResult,
                      datasetResult,
                      simulationPlanResult,
                      step1Output,
                      knowledgeGraph,
                      kgResult,
                      hypothesisResult
                    } 
                  });
                }
              }}
            >
              <svg
                className="w-6 h-6 text-green-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
              <span className="text-2xl text-white animate-blink">
                Code Ready
              </span>
            </div>
          </div>
        ) : (
          // Typing complete but code not ready yet
          <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
            <div className="flex items-center gap-3">
              <svg
                className="w-6 h-6 text-green-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
              <ShinyText 
                text="Datasets Completed" 
                disabled={false} 
                speed={3} 
                className='text-2xl' 
              />
            </div>
          </div>
        )
      ) : (
        // While typing is happening, show "Datasets Completed"
        <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
          <div className="flex items-center gap-3">
            <svg
              className="w-6 h-6 text-green-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
            <ShinyText 
              text="Datasets Completed" 
              disabled={false} 
              speed={3} 
              className='text-2xl' 
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default DatasetPage;

