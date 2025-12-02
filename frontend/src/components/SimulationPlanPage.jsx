import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ShinyText from './ShinyText';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const SimulationPlanPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { simulationPlanResult, step1Output, knowledgeGraph, kgResult, hypothesisResult } = location.state || {};
  
  const [displayedText, setDisplayedText] = useState('');
  const [isTypingComplete, setIsTypingComplete] = useState(false);
  const [isDatasetLoading, setIsDatasetLoading] = useState(false);
  const [isDatasetReady, setIsDatasetReady] = useState(false);
  const [datasetResult, setDatasetResult] = useState(null);
  const [simulationEquations, setSimulationEquations] = useState([]);
  const hasDatasetProcessed = useRef(false);
  const scrollContainerRef = useRef(null);
  const typingIntervalRef = useRef(null);
  
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

  // Convert equation text to LaTeX format for KaTeX rendering
  const convertToLaTeX = (formula) => {
    if (!formula) return '';
    
    let latex = formula.trim();
    
    // Clean up any existing LaTeX artifacts first
    latex = latex.replace(/\\right\)/g, ')');
    latex = latex.replace(/\\left\(/g, '(');
    
    // Handle brackets - convert [gnd] to [\text{gnd}] (do this first)
    latex = latex.replace(/\[([a-zA-Z]+)\]/g, '[\\text{$1}]');
    
    // Convert function names (do this before other replacements)
    latex = latex.replace(/exp\s*\(/g, '\\exp(');
    latex = latex.replace(/ln\s*\(/g, '\\ln(');
    
    // Protect variable names with underscores by wrapping in \text{}
    // Match variable names like infection_rate, final_infected_count_when_i_is_seed
    // These are typically lowercase variable names with underscores, not mathematical subscripts
    const protectedVars = new Map();
    let varIndex = 0;
    
    // Find and protect variable names with underscores
    // Pattern: variable names with underscores (like infection_rate, k_shell_decomposition)
    // Match both lowercase and mixed case variable/function names
    latex = latex.replace(/\b([a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9][a-zA-Z0-9_]*)\b/g, (match) => {
      // Only protect if it's a multi-character name with underscores
      // and doesn't look like a simple mathematical subscript pattern (like x_i, A_1)
      if (match.includes('_') && match.length > 3) {
        // Check if it's a simple subscript pattern (single char before underscore, single char/number after)
        const simpleSubscriptPattern = /^[a-zA-Z]_[a-z0-9]$/;
        if (!simpleSubscriptPattern.test(match)) {
          // Use placeholder without underscores and with special markers to avoid conversion issues
          const placeholder = `XXPROTECTEDVAR${varIndex}XX`;
          protectedVars.set(placeholder, match);
          varIndex++;
          return placeholder;
        }
      }
      return match;
    });
    
    // Handle subscripts - process in specific order
    // 1. Handle explicit underscores like DH_U2TS -> DH_{U2TS}
    // But skip placeholders (they don't have underscores anyway)
    latex = latex.replace(/_([A-Z0-9]+)/g, '_{$1}');
    
    // 2. Handle patterns like U2TS -> U_{2TS} (but only if not already subscripted)
    latex = latex.replace(/([A-Z])(\d+)([A-Z]+)(?![_^{}])/g, '$1_{$2$3}');
    
    // 3. Handle simple trailing numbers like T0 -> T_{0}
    latex = latex.replace(/([A-Z])(\d+)(?![_^{}a-zA-Z])/g, '$1_{$2}');
    
    // Handle superscripts
    latex = latex.replace(/\^(\d+)/g, '^{$1}');
    
    // Convert multiplication
    latex = latex.replace(/\s*\*\s*/g, ' \\cdot ');
    
    // Clean up spacing around operators
    latex = latex.replace(/\s*=\s*/g, ' = ');
    latex = latex.replace(/\s*\+\s*/g, ' + ');
    latex = latex.replace(/\s*-\s*/g, ' - ');
    
    // Fix nested subscripts - merge adjacent ones
    let prevLatex = '';
    let iterations = 0;
    while (prevLatex !== latex && iterations < 10) {
      prevLatex = latex;
      // Merge _{X}_{Y} -> _{XY}
      latex = latex.replace(/(_{[^}]+)}_{([^}]+)}/g, '_{$1$2}');
      iterations++;
    }
    
    // Restore protected variable names wrapped in \text{}
    // Process in reverse order to avoid conflicts
    const sortedPlaceholders = Array.from(protectedVars.entries()).sort((a, b) => b[0].length - a[0].length);
    sortedPlaceholders.forEach(([placeholder, originalVar]) => {
      const escapedPlaceholder = placeholder.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      latex = latex.replace(new RegExp(escapedPlaceholder, 'g'), `\\text{${originalVar}}`);
    });
    
    return latex;
  };

  // Render equation with KaTeX
  const renderEquation = (equation) => {
    if (!equation || !equation.trim()) {
      return '';
    }
    
    const equationTrimmed = equation.trim();
    
    try {
      // Convert to LaTeX
      const latex = convertToLaTeX(equationTrimmed);
      
      // Validate the LaTeX before rendering
      if (!latex || latex.trim() === '') {
        return `<span class="text-gray-300 font-mono">${equationTrimmed}</span>`;
      }
      
      // Try to render with KaTeX
      const rendered = katex.renderToString(latex, {
        throwOnError: false,
        displayMode: true,
        strict: false,
      });
      
      if (!rendered || rendered.trim() === '') {
        return `<span class="text-gray-300 font-mono">${equationTrimmed}</span>`;
      }
      
      return rendered;
    } catch (e) {
      console.error('KaTeX rendering failed for equation:', equationTrimmed, 'Error:', e);
      // Fallback to plain text with monospace font
      return `<span class="text-gray-300 font-mono">${equationTrimmed}</span>`;
    }
  };

  // Format simulation plan result as readable text
  const formatSimulationPlanAsText = (planData) => {
    if (!planData) return '';
    
    let text = '';
    
    // Simulation Equations - will be rendered separately with LaTeX
    if (planData.simulation_equations && Array.isArray(planData.simulation_equations) && planData.simulation_equations.length > 0) {
      text += `Simulation Equations\n${'='.repeat(50)}\n`;
      planData.simulation_equations.forEach((eq, index) => {
        text += `${index + 1}. ${eq}`;
        if (index < planData.simulation_equations.length - 1) {
          text += '\n';
        }
      });
      text += '\n';
    }
    
    // Constants Required
    if (planData.constants_required && Array.isArray(planData.constants_required) && planData.constants_required.length > 0) {
      text += `Constants Required\n${'='.repeat(50)}\n`;
      planData.constants_required.forEach((constant, index) => {
        text += `${index + 1}. ${constant.name || 'Unknown'}`;
        if (constant.description) {
          text += ` - ${constant.description}`;
        }
        if (constant.value_or_range) {
          text += `\n   Value/Range: ${constant.value_or_range}`;
        }
        text += '\n';
      });
      text += '\n';
    }
    
    // Variables to Vary
    if (planData.variables_to_vary && Array.isArray(planData.variables_to_vary) && planData.variables_to_vary.length > 0) {
      text += `Variables To Vary\n${'='.repeat(50)}\n`;
      planData.variables_to_vary.forEach((variable, index) => {
        text += `${index + 1}. ${variable.name || 'Unknown'}`;
        if (variable.description) {
          text += ` - ${variable.description}`;
        }
        if (variable.range) {
          text += `\n   Range: ${variable.range}`;
        }
        if (variable.units) {
          text += `\n   Units: ${variable.units}`;
        }
        text += '\n';
      });
      text += '\n';
    }
    
    // Procedure Steps
    if (planData.procedure_steps && Array.isArray(planData.procedure_steps) && planData.procedure_steps.length > 0) {
      text += `Procedure Steps\n${'='.repeat(50)}\n`;
      planData.procedure_steps.forEach((step, index) => {
        text += `${step}\n`;
      });
      text += '\n';
    }
    
    // Expected Outcomes
    if (planData.expected_outcomes) {
      text += `Expected Outcomes\n${'='.repeat(50)}\n`;
      text += `${planData.expected_outcomes}\n`;
    }
    
    return text;
  };

  // Typing animation effect
  useEffect(() => {
    if (simulationPlanResult) {
      // Extract simulation equations separately for rendering
      if (simulationPlanResult.simulation_equations && Array.isArray(simulationPlanResult.simulation_equations)) {
        setSimulationEquations(simulationPlanResult.simulation_equations);
      }
      
      const fullText = formatSimulationPlanAsText(simulationPlanResult);
      setDisplayedText('');
      setIsTypingComplete(false);
      
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

      typingIntervalRef.current = typingInterval;

      return () => {
        if (typingIntervalRef.current) {
          clearInterval(typingIntervalRef.current);
          typingIntervalRef.current = null;
        }
      };
    }
  }, [simulationPlanResult]);

  // Handle click to skip typing animation
  const handleWindowClick = () => {
    if (!isTypingComplete && simulationPlanResult) {
      // Stop the typing animation
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }
      // Show full text immediately
      const fullText = formatSimulationPlanAsText(simulationPlanResult);
      setDisplayedText(fullText);
      setIsTypingComplete(true);
    }
  };

  // Auto-scroll effect - scroll to bottom as text is typed
  useEffect(() => {
    if (scrollContainerRef.current && displayedText) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [displayedText]);

  // Call DatasetAgent immediately when we have the data (runs in parallel with typing)
  useEffect(() => {
    if (hasDatasetProcessed.current) {
      return;
    }

    if (!simulationPlanResult) {
      return;
    }

    hasDatasetProcessed.current = true;

    const generateDatasets = async () => {
      setIsDatasetLoading(true);
      try {
        console.log('[Frontend] DEBUG: Calling DatasetAgent API...');
        console.log('[Frontend] DEBUG: API URL:', `${API_URL}/api/generate-datasets`);
        console.log('[Frontend] DEBUG: Simulation plan keys:', simulationPlanResult ? Object.keys(simulationPlanResult) : 'null');
        console.log('[Frontend] DEBUG: Simulation plan:', JSON.stringify(simulationPlanResult, null, 2));
        
        const response = await fetch(`${API_URL}/api/generate-datasets`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            simulation_plan: simulationPlanResult,
          }),
        });

        console.log('[Frontend] DEBUG: Response status:', response.status, response.statusText);

        if (!response.ok) {
          const errorData = await response.json();
          console.error('[Frontend] ERROR: API error response:', errorData);
          throw new Error(errorData.error || 'Failed to generate datasets');
        }

        const data = await response.json();
        console.log('[Frontend] DEBUG: Full API response:', JSON.stringify(data, null, 2));
        console.log('[Frontend] DEBUG: Dataset result:', JSON.stringify(data.result, null, 2));
        console.log('[Frontend] DEBUG: Dataset type:', data.result?.dataset_type);
        console.log('[Frontend] DEBUG: Number of datasets:', data.result?.datasets ? Object.keys(data.result.datasets).length : 0);
        console.log('[Frontend] DEBUG: Dataset names:', data.result?.datasets ? Object.keys(data.result.datasets) : []);
        
        if (data.result?.datasets) {
          Object.entries(data.result.datasets).forEach(([name, path]) => {
            console.log(`[Frontend] DEBUG: Dataset '${name}': ${path}`);
          });
        }
        
        setDatasetResult(data.result);
        setIsDatasetReady(true);
        console.log('[Frontend] DEBUG: Dataset result set in state, isDatasetReady = true');
      } catch (err) {
        console.error('[Frontend] ERROR: Error generating datasets:', err);
        console.error('[Frontend] ERROR: Error details:', err.message);
        // Don't set error state, just log it - dataset generation is optional
      } finally {
        setIsDatasetLoading(false);
        console.log('[Frontend] DEBUG: Dataset loading completed');
      }
    };

    generateDatasets();
  }, [simulationPlanResult, API_URL]);

  if (!simulationPlanResult) {
    return (
      <div className="min-h-screen w-full bg-black flex items-center justify-center">
        <div className="text-white text-xl">No simulation plan data available</div>
      </div>
    );
  }

  return (
    <div className="h-screen w-full bg-black relative flex flex-col overflow-hidden">
      {/* Big Window - 80% height */}
      <div className="h-[80vh] w-full flex items-center justify-center px-8 pt-8">
        <div 
          className={`w-full max-w-7xl h-full bg-black rounded-3xl p-8 ${!isTypingComplete ? 'cursor-pointer' : ''}`}
          onClick={handleWindowClick}
          title={!isTypingComplete ? 'Click to show full text' : ''}
        >
          <div className="h-full overflow-auto hide-scrollbar" ref={scrollContainerRef} style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
            <div className="text-gray-300 leading-relaxed" style={{ fontFamily: 'Courier New, monospace' }}>
              {(() => {
                const lines = displayedText.split('\n');
                let equationsIndex = 0;
                
                // Helper function to check if we're in Simulation Equations section
                const isInSimulationEquationsSection = (lineIndex) => {
                  // Find the last complete section heading before this line
                  let lastSection = '';
                  for (let i = 0; i <= lineIndex && i < lines.length; i++) {
                    const line = lines[i];
                    if (line && !line.startsWith('=') && !line.match(/^\d+\./)) {
                      const isHeading = i === 0 || (i > 0 && lines[i - 1].startsWith('='));
                      if (isHeading) {
                        const trimmedLine = line.trim();
                        if (trimmedLine === 'Simulation Equations' || trimmedLine.startsWith('Simulation Equations')) {
                          lastSection = trimmedLine;
                        }
                      }
                    }
                  }
                  // Also check if "Simulation Equations" appears in the text before this line (for typing animation)
                  if (lastSection !== 'Simulation Equations') {
                    const textUpToLine = lines.slice(0, lineIndex + 1).join('\n');
                    const equationsHeadingIndex = textUpToLine.indexOf('Simulation Equations');
                    if (equationsHeadingIndex > -1) {
                      // Check if we're past the Simulation Equations heading and separator
                      const afterEquations = textUpToLine.substring(equationsHeadingIndex);
                      if (afterEquations.includes('=') && afterEquations.split('\n').length > 2) {
                        return true;
                      }
                    }
                  }
                  return lastSection === 'Simulation Equations';
                };
                
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
                  
                  // Check if this is an equation line (starts with number and dot)
                  const equationMatch = line.match(/^(\d+)\.\s*(.+)$/);
                  
                  if (equationMatch) {
                    const inEquationsSection = isInSimulationEquationsSection(lineIndex);
                    const equationContent = equationMatch[2].trim();
                    
                    // Check if this looks like an equation (has = sign or mathematical operators)
                    const looksLikeEquation = equationContent.includes('=') || 
                                             equationContent.includes('+') || 
                                             equationContent.includes('-') || 
                                             equationContent.includes('*') ||
                                             equationContent.includes('/') ||
                                             equationContent.includes('exp') ||
                                             equationContent.includes('ln');
                    
                    // If we're in Simulation Equations section and have equations, render them
                    if (inEquationsSection && simulationEquations.length > equationsIndex && looksLikeEquation) {
                      const equationText = simulationEquations[equationsIndex];
                      
                      // Check if line is complete (next line is empty, next item, or end of text)
                      const nextLine = lineIndex < lines.length - 1 ? lines[lineIndex + 1] : '';
                      const isComplete = equationContent.length > 0 && 
                                        (nextLine.trim() === '' || 
                                         nextLine.match(/^\d+\./) ||
                                         lineIndex === lines.length - 1);
                      
                      // Render with KaTeX when line is complete or we have enough content
                      if (isComplete || equationContent.length > 20) {
                        const currentEquationIndex = equationsIndex;
                        equationsIndex++;
                        const renderedEquation = renderEquation(equationText);
                        
                        return (
                          <div key={lineIndex} className="my-1">
                            <span className="text-gray-400">{equationMatch[1]}.</span>
                            <div 
                              className="ml-2 inline-block"
                              dangerouslySetInnerHTML={{ __html: renderedEquation }}
                            />
                            {'\n'}
                          </div>
                        );
                      } else {
                        // Partial equation - show as text while typing
                        return (
                          <div key={lineIndex} className="my-2 font-mono text-sm">
                            {line}
                            {'\n'}
                          </div>
                        );
                      }
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

      {/* Divider */}
      <div className="absolute bottom-[17vh] left-0 right-0 w-full flex justify-center px-8">
        <div className="w-full max-w-4xl border-t" style={{ borderColor: '#00A86B' }}></div>
      </div>

      {/* Status text below the window */}
      {isTypingComplete ? (
        // After typing completes, check dataset status
        isDatasetLoading ? (
          <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
            <ShinyText 
              text="Generating Datasets..." 
              disabled={false} 
              speed={3} 
              className='text-2xl' 
            />
          </div>
        ) : isDatasetReady ? (
          <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
            <div 
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => {
                if (isDatasetReady && datasetResult) {
                  // Navigate to dataset page with all previous data
                  navigate('/datasets', { 
                    state: { 
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
                Datasets Ready
              </span>
            </div>
          </div>
        ) : (
          // Typing complete but datasets not ready yet
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
                text="Simulation Plan Completed" 
                disabled={false} 
                speed={3} 
                className='text-2xl' 
              />
            </div>
          </div>
        )
      ) : (
        // While typing is happening, show "Simulation Plan Completed"
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
              text="Simulation Plan Completed" 
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

export default SimulationPlanPage;

