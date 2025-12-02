import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ShinyText from './ShinyText';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { file } = location.state || {};
  
  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [displayedText, setDisplayedText] = useState('');
  const [formulas, setFormulas] = useState([]);
  const [isTypingComplete, setIsTypingComplete] = useState(false);
  const [isKgLoading, setIsKgLoading] = useState(false);
  const [isKgReady, setIsKgReady] = useState(false);
  const [kgResult, setKgResult] = useState(null);
  const hasProcessed = useRef(false); // Prevent multiple API calls
  const hasKgProcessed = useRef(false); // Prevent multiple KG API calls
  const scrollContainerRef = useRef(null);
  
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  // Convert formula text to LaTeX format for KaTeX rendering
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
    
    // Handle subscripts - process in specific order
    // 1. Handle explicit underscores like DH_U2TS -> DH_{U2TS}
    //    Simple replacement - underscores become subscripts
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
    
    return latex;
  };

  // Render formula with KaTeX
  const renderFormula = (formula) => {
    if (!formula || !formula.trim()) {
      return '';
    }
    
    const formulaTrimmed = formula.trim();
    
    try {
      // Convert to LaTeX
      const latex = convertToLaTeX(formulaTrimmed);
      
      // Validate the LaTeX before rendering
      if (!latex || latex.trim() === '') {
        return `<span class="text-gray-300 font-mono">${formulaTrimmed}</span>`;
      }
      
      // Try to render with KaTeX
      const rendered = katex.renderToString(latex, {
        throwOnError: false,
        displayMode: true,
        strict: false,
      });
      
      if (!rendered || rendered.trim() === '') {
        return `<span class="text-gray-300 font-mono">${formulaTrimmed}</span>`;
      }
      
      return rendered;
    } catch (e) {
      console.error('KaTeX rendering failed for formula:', formulaTrimmed, 'Error:', e);
      // Fallback to plain text with monospace font
      return `<span class="text-gray-300 font-mono">${formulaTrimmed}</span>`;
    }
  };


  useEffect(() => {
    // Prevent multiple API calls
    if (hasProcessed.current) {
      return;
    }

    if (!file) {
      navigate('/');
      return;
    }

    hasProcessed.current = true; // Mark as processed

    const processFile = async () => {
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/api/analyze-pdf`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to process PDF');
        }

        const data = await response.json();
        setResult({ ...data, fileName: file.name });
      } catch (err) {
        setError(err.message || 'An error occurred while processing the PDF');
        console.error('Error processing PDF:', err);
      } finally {
        setLoading(false);
      }
    };

    processFile();
    // Only run once on mount - file and navigate are stable references
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Format result as readable text with headings
  const formatResultAsText = (resultData) => {
    let text = '';

    //if formula key is present, change it to formulas and delete the formula key
    if (resultData.formula) {
        resultData.formulas = [resultData.formula];
        delete resultData.formula;
    }

    // Helper function to format key names (replace underscores with spaces, capitalize)
    const formatKeyName = (key) => {
      return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    };

    // Process all keys dynamically
    Object.keys(resultData).forEach(key => {
      // Skip the old 'formula' key if it still exists (shouldn't after conversion)
      if (key === 'formula') return;
      
      // Skip 'formulas' key - it will be handled separately with LaTeX rendering
      if (key === 'formulas') return;
      
      const sectionName = formatKeyName(key);
      text += `${sectionName}\n${'='.repeat(50)}\n`;
      
      if (Array.isArray(resultData[key])) {
        // Arrays - one item per line
        resultData[key].forEach((item, index) => {
          text += `${index + 1}. ${item}\n`;
        });
      } else {
        // Non-array values (like summary)
        text += `${resultData[key]}\n`;
      }
      text += '\n';
    });
    
    // Add formulas section separately (will be rendered with LaTeX)
    if (resultData.formulas && Array.isArray(resultData.formulas) && resultData.formulas.length > 0) {
      text += `Formulas\n${'='.repeat(50)}\n`;
      resultData.formulas.forEach((item, index) => {
        text += `${index + 1}. ${item}`;
        if (index < resultData.formulas.length - 1) {
          text += '\n';
        }
      });
      text += '\n';
    }
    
    return text;
  };

  // Typing animation effect - format as readable text
  useEffect(() => {
    if (result && result.result) {
      // Extract formulas separately for rendering
      if (result.result.formulas && Array.isArray(result.result.formulas)) {
        setFormulas(result.result.formulas);
      }
      
      // Format result as readable text instead of JSON
      const fullText = formatResultAsText(result.result);
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
  }, [result]);

  // Call KnowledgeGraphAgent after typing completes and 2 second wait
  useEffect(() => {
    if (isTypingComplete && result && result.result && !hasKgProcessed.current) {
      hasKgProcessed.current = true;
      
      const callKnowledgeGraph = async () => {
        // Wait 2 seconds after typing completes
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        setIsKgLoading(true);
        try {
          const response = await fetch(`${API_URL}/api/build-knowledge-graph`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ result: result.result }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to build knowledge graph');
          }

          const data = await response.json();
          console.log('Knowledge Graph Result:', JSON.stringify(data, null, 2));
          setKgResult(data);
          setIsKgReady(true);
        } catch (err) {
          console.error('Error building knowledge graph:', err);
          // Don't set error state, just log it - KG is optional
        } finally {
          setIsKgLoading(false);
        }
      };

      callKnowledgeGraph();
    }
  }, [isTypingComplete, result, API_URL]);


  // Auto-scroll effect - scroll to bottom as text is typed
  useEffect(() => {
    if (scrollContainerRef.current && displayedText) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [displayedText]);

  if (!file) {
    return null;
  }

  return (
    <div className="h-screen w-full bg-black relative flex flex-col overflow-hidden">
      {/* Big Window - 80% height */}
      <div className="h-[80vh] w-full flex items-center justify-center px-8 pt-8">
        <div className="w-full max-w-7xl h-full bg-black rounded-3xl p-8">
          {loading ? (
            <div className="h-full flex items-center justify-center">
              <p className="text-white text-xl">{file?.name || 'Loading...'}</p>
            </div>
          ) : error ? (
            <div className="h-full flex items-center justify-center">
              <div className="bg-red-900/30 border border-red-700 rounded-lg p-6 w-full">
                <div className="flex items-center gap-2 mb-2">
                  <svg
                    className="w-5 h-5 text-red-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <h3 className="text-lg font-semibold text-red-300">Error</h3>
                </div>
                <p className="text-red-200">{error}</p>
              </div>
            </div>
          ) : result && result.result ? (
            <div className="h-full overflow-auto hide-scrollbar" ref={scrollContainerRef} style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
              <div className="text-gray-300 leading-relaxed" style={{ fontFamily: 'Courier New, monospace' }}>
                  {(() => {
                    const lines = displayedText.split('\n');
                  let formulasIndex = 0;
                  
                  // Get section names from the actual JSON keys, matching formatResultAsText
                  // Use the same formatKeyName function logic
                  const formatKeyName = (key) => {
                    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                  };
                  
                  // Get section names from first result
                  const sectionNames = Object.keys(result.result).map(key => {
                    // Skip formulas if it exists (handled separately)
                    if (key === 'formulas' || key === 'formula') return null;
                    return formatKeyName(key);
                  }).filter(Boolean);
                  
                  // Helper function to check if we're in Formulas section
                  const isInFormulasSection = (lineIndex) => {
                    // Find the last complete section heading before this line
                    let lastSection = '';
                    for (let i = 0; i <= lineIndex && i < lines.length; i++) {
                      const line = lines[i];
                      if (line && !line.startsWith('=') && !line.match(/^\d+\./)) {
                        const isHeading = i === 0 || (i > 0 && lines[i - 1].startsWith('='));
                        if (isHeading) {
                          const trimmedLine = line.trim();
                          // Check if this matches any section name (including partial matches during typing)
                          if (sectionNames.some(section => section === trimmedLine || section.startsWith(trimmedLine) || trimmedLine.startsWith(section))) {
                            lastSection = trimmedLine;
                          }
                        }
                      }
                    }
                    // Also check if "Formulas" appears in the text before this line (for typing animation)
                    if (lastSection !== 'Formulas') {
                      const textUpToLine = lines.slice(0, lineIndex + 1).join('\n');
                      const formulasHeadingIndex = textUpToLine.indexOf('Formulas');
                      if (formulasHeadingIndex > -1) {
                        // Check if we're past the Formulas heading and separator
                        const afterFormulas = textUpToLine.substring(formulasHeadingIndex);
                        if (afterFormulas.includes('=') && afterFormulas.split('\n').length > 2) {
                          return true;
                        }
                      }
                    }
                    return lastSection === 'Formulas';
                  };
                  
                  return lines.map((line, lineIndex) => {
                    // Detect section headings
                    if (line && !line.startsWith('=') && !line.match(/^\d+\./)) {
                      // Check if this is a heading: first line OR next line is separator (skip blank lines)
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
                        const trimmedLine = line.trim();
                        // Match section names - allow exact match or partial match during typing (but require at least 3 chars)
                        if (sectionNames.some(section => {
                          // Exact match
                          if (trimmedLine === section) return true;
                          // Partial match during typing - only if at least 3 characters and section starts with the line
                          if (trimmedLine.length >= 3 && section.startsWith(trimmedLine)) {
                            return true;
                          }
                          return false;
                        })) {
                          return (
                            <React.Fragment key={lineIndex}>
                              <h2 className="font-bold text-lg mt-3 mb-1" style={{ fontWeight: 'bold', color: '#00A86B' }}>{line}</h2>
                              {'\n'}
                            </React.Fragment>
                          );
                        }
                      }
                    }
                    
                    // Skip separator lines
                    if (line.startsWith('=')) {
                      return null;
                    }
                    
                    // Check if this is a formula line (starts with number and dot)
                    const formulaMatch = line.match(/^(\d+)\.\s*(.+)$/);
                    
                    if (formulaMatch) {
                      const inFormulasSection = isInFormulasSection(lineIndex);
                      const formulaContent = formulaMatch[2].trim();
                      
                      // Check if this looks like a formula (has = sign or mathematical operators)
                      const looksLikeFormula = formulaContent.includes('=') || 
                                               formulaContent.includes('+') || 
                                               formulaContent.includes('-') || 
                                               formulaContent.includes('*') ||
                                               formulaContent.includes('/') ||
                                               formulaContent.includes('exp') ||
                                               formulaContent.includes('ln');
                      
                      // If we're in Formulas section and have formulas, render them
                      if (inFormulasSection && formulas.length > formulasIndex && looksLikeFormula) {
                        const formulaText = formulas[formulasIndex];
                        
                        // Check if line is complete (next line is empty, next item, or end of text)
                        const nextLine = lineIndex < lines.length - 1 ? lines[lineIndex + 1] : '';
                        const isComplete = formulaContent.length > 0 && 
                                          (nextLine.trim() === '' || 
                                           nextLine.match(/^\d+\./) ||
                                           lineIndex === lines.length - 1);
                        
                        // Render with KaTeX when line is complete or we have enough content
                        if (isComplete || formulaContent.length > 20) {
                          const currentFormulaIndex = formulasIndex;
                          formulasIndex++;
                          const renderedFormula = renderFormula(formulaText);
                          
                          return (
                            <div key={lineIndex} className="my-1">
                              <span className="text-gray-400">{formulaMatch[1]}.</span>
                              <div 
                                className="ml-2 inline-block"
                                dangerouslySetInnerHTML={{ __html: renderedFormula }}
                              />
                              {'\n'}
                            </div>
                          );
                        } else {
                          // Partial formula - show as text while typing
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
          ) : null}
        </div>
      </div>

      {/* Divider */}
      <div className="absolute bottom-[17vh] left-0 right-0 w-full flex justify-center px-8">
        <div className="w-full max-w-4xl border-t" style={{ borderColor: '#00A86B' }}></div>
      </div>

      {/* Status text below the window */}
      {loading ? (
        <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
          <ShinyText 
            text="Understanding..." 
            disabled={false} 
            speed={3} 
            className='text-2xl' 
          />
        </div>
      ) : result && result.result ? (
        <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
          <div 
            className={`flex items-center gap-3 ${isKgReady ? 'cursor-pointer hover:opacity-80 transition-opacity' : ''}`}
            onClick={() => {
              if (isKgReady && kgResult) {
                navigate('/graph', { state: { kgResult, step1Output: result.result } });
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
            {isKgReady ? (
              <span className="text-2xl text-white animate-blink">
                Knowledge Graph Ready
              </span>
            ) : (
              <ShinyText 
                text="Paper Understood" 
                disabled={false} 
                speed={3} 
                className='text-2xl' 
              />
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default ResultsPage;

