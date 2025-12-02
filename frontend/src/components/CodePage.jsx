import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ShinyText from './ShinyText';

const CodePage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  // All data flows from ResultsPage through all pages:
  // - step1Output: Paper understanding from ResultsPage
  // - kgResult: Knowledge graph result object from ResultsPage
  // - knowledgeGraph: Knowledge graph data (kgResult.result) from GraphPage
  // - hypothesisResult: Hypothesis from HypothesisPage
  // - simulationPlanResult: Simulation plan from SimulationPlanPage
  // - datasetResult: Dataset generation results from DatasetPage
  // - codeResult: Generated Python code from DatasetPage
  const { 
    codeResult, 
    datasetResult, 
    simulationPlanResult, 
    step1Output, 
    knowledgeGraph, 
    kgResult, 
    hypothesisResult 
  } = location.state || {};
  
  const [displayedText, setDisplayedText] = useState('');
  const [isTypingComplete, setIsTypingComplete] = useState(false);
  const [generatedCodeResult, setGeneratedCodeResult] = useState(codeResult || null);
  const [correctedCode, setCorrectedCode] = useState(null);
  const [isValidating, setIsValidating] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [simulationResult, setSimulationResult] = useState(null);
  const [simulationStarted, setSimulationStarted] = useState(false);
  const hasValidatedCode = useRef(false);
  const hasRunSimulation = useRef(false);
  const scrollContainerRef = useRef(null);
  const typingIntervalRef = useRef(null);
  
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

  // Use codeResult directly if provided
  useEffect(() => {
    if (codeResult && codeResult.python_code) {
      setGeneratedCodeResult(codeResult);
    }
  }, [codeResult]);

  // Helper function to fetch dataset metadata
  const fetchDatasetMetadata = async (datasets) => {
    if (!datasets || Object.keys(datasets).length === 0) {
      return null;
    }

    const metadata = {};
    
    for (const [name, path] of Object.entries(datasets)) {
      try {
        // Extract filename from path
        const filename = path.split('/').pop() || path.split('\\').pop();
        
        // Extract relative path for API call (everything after "datasets/")
        // Paths might be like: "outputs/datasets/synthetic/file.csv" or full paths
        let relativePath = path;
        if (path.includes('/datasets/')) {
          relativePath = path.split('/datasets/')[1];
        } else if (path.includes('datasets/')) {
          relativePath = path.split('datasets/')[1];
        }
        
        // Try to fetch the dataset file to get metadata
        const response = await fetch(`${API_URL}/api/datasets/${relativePath}`);
        
        if (response.ok) {
          const content = await response.text();
          const lines = content.split('\n').filter(line => line.trim());
          
          // Parse CSV if it's a CSV file
          if (filename.endsWith('.csv') && lines.length > 0) {
            // Handle CSV parsing more carefully (handle quoted values)
            const firstLine = lines[0];
            const headers = firstLine.split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
            const rowCount = Math.max(0, lines.length - 1); // Exclude header
            
            metadata[name] = {
              filename: filename,
              path: path,
              type: 'CSV',
              rows: rowCount,
              columns: headers.length,
              column_names: headers,
              file_size_bytes: content.length
            };
          } else {
            // For non-CSV files, just get basic info
            metadata[name] = {
              filename: filename,
              path: path,
              type: filename.split('.').pop().toUpperCase(),
              rows: lines.length,
              file_size_bytes: content.length
            };
          }
        } else {
          // If we can't fetch, just include filename and path
          metadata[name] = {
            filename: filename,
            path: path,
            type: filename.split('.').pop().toUpperCase()
          };
        }
      } catch (err) {
        console.warn(`Failed to fetch metadata for dataset ${name}:`, err);
        // Include at least filename and path
        const filename = path.split('/').pop() || path.split('\\').pop();
        metadata[name] = {
          filename: filename,
          path: path,
          type: filename.split('.').pop().toUpperCase()
        };
      }
    }
    
    return metadata;
  };

  // Validate code FIRST when page loads
  useEffect(() => {
    if (hasValidatedCode.current) {
      return;
    }

    if (!codeResult || !codeResult.python_code) {
      return;
    }

    if (!simulationPlanResult) {
      return;
    }

    hasValidatedCode.current = true;
    setIsValidating(true);

    const validateCode = async () => {
      try {
        // Fetch dataset metadata if datasets are available
        let datasetMetadata = null;
        if (datasetResult && datasetResult.datasets) {
          datasetMetadata = await fetchDatasetMetadata(datasetResult.datasets);
        }

        const requestBody = {
          python_code: codeResult.python_code,
          simulation_plan: simulationPlanResult,
        };

        // Add dataset metadata if available
        if (datasetMetadata) {
          requestBody.dataset_metadata = datasetMetadata;
        }

        const response = await fetch(`${API_URL}/api/validate-code`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to validate code');
        }

        const data = await response.json();
        const validationResult = data.result;
        
        // If returned code is different from original, store it (regardless of has_errors flag)
        if (validationResult.python_code && validationResult.python_code !== codeResult.python_code) {
          setCorrectedCode(validationResult.python_code);
        }
        
        setIsValidating(false);
        // Start typing animation and simulation after validation
        setSimulationStarted(true);
      } catch (err) {
        console.error('Error validating code:', err);
        setIsValidating(false);
        // Continue with original code if validation fails
        setSimulationStarted(true);
      }
    };

    validateCode();
  }, [codeResult, simulationPlanResult, API_URL]);

  // Run simulation after code is validated/confirmed (only if no corrected code available)
  useEffect(() => {
    if (hasRunSimulation.current) {
      return;
    }

    if (!simulationStarted) {
      return;
    }

    // If corrected code is available, wait for user to click recycle button
    if (correctedCode) {
      return;
    }

    // Use original code
    const codeToUse = generatedCodeResult && generatedCodeResult.python_code;
    
    if (!codeToUse) {
      return;
    }

    if (!simulationPlanResult) {
      return;
    }

    // Only set running state and run simulation if all conditions are met
    hasRunSimulation.current = true;
    setIsRunning(true);

    const runSimulation = async () => {
      try {
        // Extract dataset paths from datasetResult
        let dataset_paths = null;
        if (datasetResult && datasetResult.datasets) {
          // datasetResult.datasets is a dict mapping names to paths
          dataset_paths = datasetResult.datasets;
        }

        console.log('Starting simulation...', { codeLength: codeToUse?.length, hasDatasets: !!dataset_paths });

        // Create a timeout promise
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => {
            reject(new Error('Simulation request timed out after 5 minutes'));
          }, 5 * 60 * 1000); // 5 minutes timeout
        });

        // Race between fetch and timeout
        const fetchPromise = fetch(`${API_URL}/api/run-simulation`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            python_code: codeToUse,
            simulation_plan: simulationPlanResult,
            dataset_paths: dataset_paths,
          }),
        });

        const response = await Promise.race([fetchPromise, timeoutPromise]);

        if (!response.ok) {
          let errorData;
          try {
            errorData = await response.json();
          } catch (e) {
            errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
          }
          throw new Error(errorData.error || 'Failed to run simulation');
        }

        const data = await response.json();
        const result = data.result;
        console.log('Simulation completed:', { status: result?.status, exitCode: result?.exit_code });
        setSimulationResult(result);
        setIsRunning(false);
      } catch (err) {
        console.error('Error running simulation:', err);
        setIsRunning(false);
        // Create error result object
        const errorResult = {
          status: 'error',
          error_summary: err.message || 'Failed to run simulation',
          stderr: err.message || 'Unknown error occurred',
          stdout: '',
          exit_code: -1,
        };
        setSimulationResult(errorResult);
      }
    };

    // Call simulation after a short delay to allow typing animation to start
    setTimeout(() => {
      runSimulation();
    }, 100);
  }, [simulationStarted, correctedCode, generatedCodeResult, simulationPlanResult, datasetResult, API_URL]);

  // Typing animation effect for code - starts after validation is complete
  useEffect(() => {
    if (simulationStarted) {
      // Always show original code first (corrected code will replace it when recycle is clicked)
      const codeToDisplay = generatedCodeResult && generatedCodeResult.python_code;
      
      if (codeToDisplay) {
        const fullCode = codeToDisplay;
        setDisplayedText('');
        setIsTypingComplete(false);
        
        let currentIndex = 0;
        const typingSpeed = 1; // Fast typing speed
        
        const typingInterval = setInterval(() => {
          if (currentIndex < fullCode.length) {
            setDisplayedText(fullCode.slice(0, currentIndex + 1));
            currentIndex++;
          } else {
            clearInterval(typingInterval);
            setIsTypingComplete(true);
          }
        }, typingSpeed);

        typingIntervalRef.current = typingInterval;

        return () => {
          if (typingIntervalRef.current) {
            clearInterval(typingIntervalRef.current);
            typingIntervalRef.current = null;
          }
        };
      }
    }
  }, [simulationStarted, generatedCodeResult]);

  // Handle recycle button click - replace code with corrected version
  const handleRecycleClick = () => {
    if (correctedCode) {
      // Store corrected code before clearing state
      const codeToUse = correctedCode;
      
      // Update the displayed code
      setDisplayedText(codeToUse);
      setIsTypingComplete(true);
      
      // Update the code result
      const updatedCodeResult = {
        ...generatedCodeResult,
        python_code: codeToUse
      };
      setGeneratedCodeResult(updatedCodeResult);
      
      // Clear corrected code so button disappears
      setCorrectedCode(null);
      
      // Restart simulation with corrected code
      hasRunSimulation.current = false;
      setIsRunning(true);
      
      const runSimulation = async () => {
        try {
          // Extract dataset paths from datasetResult
          let dataset_paths = null;
          if (datasetResult && datasetResult.datasets) {
            dataset_paths = datasetResult.datasets;
          }

          console.log('Starting simulation with corrected code...', { codeLength: codeToUse?.length, hasDatasets: !!dataset_paths });

          // Create a timeout promise
          const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => {
              reject(new Error('Simulation request timed out after 5 minutes'));
            }, 5 * 60 * 1000); // 5 minutes timeout
          });

          // Race between fetch and timeout
          const fetchPromise = fetch(`${API_URL}/api/run-simulation`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              python_code: codeToUse,
              simulation_plan: simulationPlanResult,
              dataset_paths: dataset_paths,
            }),
          });

          const response = await Promise.race([fetchPromise, timeoutPromise]);

          if (!response.ok) {
            let errorData;
            try {
              errorData = await response.json();
            } catch (e) {
              errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
            }
            throw new Error(errorData.error || 'Failed to run simulation');
          }

          const data = await response.json();
          const result = data.result;
          console.log('Simulation completed:', { status: result?.status, exitCode: result?.exit_code });
          setSimulationResult(result);
          setIsRunning(false);
        } catch (err) {
          console.error('Error running simulation:', err);
          setIsRunning(false);
          const errorResult = {
            status: 'error',
            error_summary: err.message || 'Failed to run simulation',
            stderr: err.message || 'Unknown error occurred',
            stdout: '',
            exit_code: -1,
          };
          setSimulationResult(errorResult);
        }
      };

      runSimulation();
    }
  };

  // Handle skip button click - ignore corrected code and proceed with original
  const handleSkipClick = () => {
    if (correctedCode) {
      // Clear corrected code so button disappears
      setCorrectedCode(null);
      
      // Proceed with original code - trigger simulation if not already running
      if (!hasRunSimulation.current) {
        hasRunSimulation.current = true;
        setIsRunning(true);
        
        const runSimulation = async () => {
          try {
            // Extract dataset paths from datasetResult
            let dataset_paths = null;
            if (datasetResult && datasetResult.datasets) {
              dataset_paths = datasetResult.datasets;
            }

            const codeToUse = generatedCodeResult && generatedCodeResult.python_code;
            
            console.log('Starting simulation (skip button)...', { codeLength: codeToUse?.length, hasDatasets: !!dataset_paths });

            // Create a timeout promise
            const timeoutPromise = new Promise((_, reject) => {
              setTimeout(() => {
                reject(new Error('Simulation request timed out after 5 minutes'));
              }, 5 * 60 * 1000); // 5 minutes timeout
            });

            // Race between fetch and timeout
            const fetchPromise = fetch(`${API_URL}/api/run-simulation`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                python_code: codeToUse,
                simulation_plan: simulationPlanResult,
                dataset_paths: dataset_paths,
              }),
            });

            const response = await Promise.race([fetchPromise, timeoutPromise]);

            if (!response.ok) {
              let errorData;
              try {
                errorData = await response.json();
              } catch (e) {
                errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
              }
              throw new Error(errorData.error || 'Failed to run simulation');
            }

            const data = await response.json();
            const result = data.result;
            console.log('Simulation completed:', { status: result?.status, exitCode: result?.exit_code });
            setSimulationResult(result);
            setIsRunning(false);
          } catch (err) {
            console.error('Error running simulation:', err);
            setIsRunning(false);
            const errorResult = {
              status: 'error',
              error_summary: err.message || 'Failed to run simulation',
              stderr: err.message || 'Unknown error occurred',
              stdout: '',
              exit_code: -1,
            };
            setSimulationResult(errorResult);
          }
        };

        runSimulation();
      }
    }
  };

  // Handle click to skip typing animation
  const handleWindowClick = () => {
    if (!isTypingComplete) {
      // Stop the typing animation
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }
      // Show full original code immediately (corrected code will be shown when recycle button is clicked)
      const codeToShow = generatedCodeResult && generatedCodeResult.python_code;
      if (codeToShow) {
        setDisplayedText(codeToShow);
        setIsTypingComplete(true);
      }
    }
  };

  // Auto-scroll effect - scroll to bottom as text is typed
  useEffect(() => {
    if (scrollContainerRef.current && displayedText) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [displayedText]);


  if (!generatedCodeResult || !generatedCodeResult.python_code) {
    return (
      <div className="min-h-screen w-full bg-black flex items-center justify-center">
        <div className="text-white text-xl">No code data available</div>
      </div>
    );
  }

  return (
    <div className="h-screen w-full bg-black relative flex flex-col overflow-hidden">
      {/* Action buttons - appear on right side middle when corrected code is available */}
      {correctedCode && (
        <div className="absolute right-8 top-1/2 -translate-y-1/2 z-10 flex flex-col gap-4">
          {/* Recycle button - apply corrected code */}
          <button
            onClick={handleRecycleClick}
            className="bg-green-500 hover:bg-green-600 text-white rounded-full p-4 shadow-lg transition-all hover:scale-110 flex items-center justify-center"
            title="Apply corrected code"
          >
            <svg
              className="w-8 h-8"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
          </button>
          
          {/* Skip button - ignore corrected code and use original */}
          <button
            onClick={handleSkipClick}
            className="bg-gray-600 hover:bg-gray-700 text-white rounded-full p-4 shadow-lg transition-all hover:scale-110 flex items-center justify-center"
            title="Keep original code"
          >
            <svg
              className="w-8 h-8"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      )}

      {/* Big Window - 80% height */}
      <div className="h-[80vh] w-full flex items-center justify-center px-8 pt-8">
        <div 
          className={`w-full max-w-7xl h-full bg-black rounded-3xl p-8 ${!isTypingComplete ? 'cursor-pointer' : ''}`}
          onClick={handleWindowClick}
          title={!isTypingComplete ? 'Click to show full code' : ''}
        >
          <div className="h-full overflow-auto hide-scrollbar" ref={scrollContainerRef} style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
            <SyntaxHighlighter
              language="python"
              style={vscDarkPlus}
              customStyle={{
                backgroundColor: 'transparent',
                padding: 0,
                margin: 0,
                fontSize: '14px',
                lineHeight: '1.6',
              }}
              showLineNumbers={false}
              wrapLines={true}
              wrapLongLines={true}
            >
              {displayedText}
            </SyntaxHighlighter>
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="absolute bottom-[17vh] left-0 right-0 w-full flex justify-center px-8">
        <div className="w-full max-w-4xl border-t" style={{ borderColor: '#00A86B' }}></div>
      </div>

      {/* Status text below the window */}
      {isValidating ? (
        <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
          <ShinyText 
            text="Validating Code..." 
            disabled={false} 
            speed={3} 
            className='text-2xl' 
          />
        </div>
      ) : isTypingComplete ? (
        // Only show status after code is fully displayed
        correctedCode ? (
          // If corrected code is available, show message to choose
          <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
            <ShinyText 
              text="Code corrections available - choose an action" 
              disabled={false} 
              speed={3} 
              className='text-2xl' 
            />
          </div>
        ) : isRunning ? (
          <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
            <ShinyText 
              text="Getting Simulation Data..." 
              disabled={false} 
              speed={3} 
              className='text-2xl' 
            />
          </div>
        ) : simulationResult ? (
          <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
            <div 
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => {
                // Navigate based on status
                if (simulationResult.status === 'error') {
                  // Navigate to error page with simulation result when clicked
                  navigate('/error', { state: { errorResult: simulationResult } });
                } else {
                  // Navigate to report generator page with all data
                  navigate('/report', { 
                    state: { 
                      simulationResult,
                      step1Output,
                      knowledgeGraph,
                      hypothesisResult,
                      simulationPlanResult
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
                Simulation Status Ready
              </span>
            </div>
          </div>
        ) : (
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
                text="Code Generated" 
                disabled={false} 
                speed={3} 
                className='text-2xl' 
              />
            </div>
          </div>
        )
      ) : (
        // While code is still typing, show "Code Generated"
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
              text="Code Generated" 
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

export default CodePage;

