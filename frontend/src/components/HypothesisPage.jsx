import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ShinyText from './ShinyText';

const HypothesisPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { hypothesisResult, step1Output, knowledgeGraph, kgResult } = location.state || {};
  
  const [displayedText, setDisplayedText] = useState('');
  const [isTypingComplete, setIsTypingComplete] = useState(false);
  const [isSimulationPlanLoading, setIsSimulationPlanLoading] = useState(false);
  const [isSimulationPlanReady, setIsSimulationPlanReady] = useState(false);
  const [simulationPlanResult, setSimulationPlanResult] = useState(null);
  const hasSimulationPlanProcessed = useRef(false);
  const scrollContainerRef = useRef(null);
  
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

  // Format hypothesis result as readable text
  const formatHypothesisAsText = (hypothesisData) => {
    if (!hypothesisData) return '';
    
    let text = '';
    
    if (hypothesisData.hypothesis) {
      text += `Hypothesis\n${'='.repeat(50)}\n`;
      text += `${hypothesisData.hypothesis}\n\n`;
    }
    
    if (hypothesisData.justification) {
      text += `Justification\n${'='.repeat(50)}\n`;
      text += `${hypothesisData.justification}\n`;
    }
    
    return text;
  };

  // Typing animation effect
  useEffect(() => {
    if (hypothesisResult) {
      const fullText = formatHypothesisAsText(hypothesisResult);
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
  }, [hypothesisResult]);

  // Auto-scroll effect - scroll to bottom as text is typed
  useEffect(() => {
    if (scrollContainerRef.current && displayedText) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [displayedText]);

  // Call SimulationPlanAgent immediately when we have all the data (runs in parallel with typing)
  useEffect(() => {
    if (hasSimulationPlanProcessed.current) {
      return;
    }

    if (!hypothesisResult || !step1Output || !knowledgeGraph) {
      return;
    }

    hasSimulationPlanProcessed.current = true;

    const generateSimulationPlan = async () => {
      setIsSimulationPlanLoading(true);
      try {
        const response = await fetch(`${API_URL}/api/generate-simulation-plan`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            hypothesis: hypothesisResult,
            step1_output: step1Output,
            knowledge_graph: knowledgeGraph,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to generate simulation plan');
        }

        const data = await response.json();
        setSimulationPlanResult(data.result);
        setIsSimulationPlanReady(true);
      } catch (err) {
        console.error('Error generating simulation plan:', err);
        // Don't set error state, just log it - simulation plan is optional
      } finally {
        setIsSimulationPlanLoading(false);
      }
    };

    generateSimulationPlan();
  }, [hypothesisResult, step1Output, knowledgeGraph, API_URL]);

  if (!hypothesisResult) {
    return (
      <div className="min-h-screen w-full bg-black flex items-center justify-center">
        <div className="text-white text-xl">No hypothesis data available</div>
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
        // After typing completes, check simulation plan status
        isSimulationPlanLoading ? (
          <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
            <ShinyText 
              text="Generating Simulation Plan..." 
              disabled={false} 
              speed={3} 
              className='text-2xl' 
            />
          </div>
        ) : isSimulationPlanReady ? (
          <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
            <div 
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => {
                if (isSimulationPlanReady && simulationPlanResult) {
                  navigate('/simulation-plan', { 
                    state: { 
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
                Simulation Plan Ready
              </span>
            </div>
          </div>
        ) : (
          // Typing complete but simulation plan not ready yet
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
                text="Hypothesis Completed" 
                disabled={false} 
                speed={3} 
                className='text-2xl' 
              />
            </div>
          </div>
        )
      ) : (
        // While typing is happening, show "Hypothesis Completed"
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
              text="Hypothesis Completed" 
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

export default HypothesisPage;

