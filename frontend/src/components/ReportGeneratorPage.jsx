import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import ShinyText from './ShinyText';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ReportGeneratorPage = () => {
  const location = useLocation();
  const { 
    simulationResult, 
    step1Output, 
    knowledgeGraph, 
    hypothesisResult, 
    simulationPlanResult 
  } = location.state || {};
  
  const [loading, setLoading] = useState(true);
  const [reportResult, setReportResult] = useState(null);
  const [error, setError] = useState(null);
  const [displayedText, setDisplayedText] = useState('');
  const [isTypingComplete, setIsTypingComplete] = useState(false);
  const hasProcessed = useRef(false);
  const scrollContainerRef = useRef(null);
  const typingIntervalRef = useRef(null);
  
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

  // Call ReportAgent when page loads
  useEffect(() => {
    if (hasProcessed.current) {
      return;
    }

    if (!simulationResult || !step1Output || !knowledgeGraph || !hypothesisResult || !simulationPlanResult) {
      setError('Missing required data for report generation');
      setLoading(false);
      return;
    }

    hasProcessed.current = true;

    const generateReport = async () => {
      setLoading(true);
      try {
        const requestData = {
          paper_understanding: step1Output,
          knowledge_graph: knowledgeGraph,
          hypothesis: hypothesisResult,
          simulation_plan: simulationPlanResult,
          simulation_result: simulationResult,
          error_history: null, // Optional, can be added later
        };
        
        // Log what's being sent to ReportAgent
        console.log('=== SENDING TO REPORTAGENT ===');
        console.log('paper_understanding:', JSON.stringify(step1Output, null, 2));
        console.log('knowledge_graph:', JSON.stringify(knowledgeGraph, null, 2));
        console.log('hypothesis:', JSON.stringify(hypothesisResult, null, 2));
        console.log('simulation_plan:', JSON.stringify(simulationPlanResult, null, 2));
        console.log('simulation_result:', JSON.stringify(simulationResult, null, 2));
        console.log('error_history:', null);
        console.log('=== END OF DATA SENT TO REPORTAGENT ===');
        
        const response = await fetch(`${API_URL}/api/generate-report`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestData),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to generate report');
        }

        const data = await response.json();
        setReportResult(data.result);
      } catch (err) {
        setError(err.message || 'An error occurred while generating the report');
        console.error('Error generating report:', err);
      } finally {
        setLoading(false);
      }
    };

    generateReport();
  }, [simulationResult, step1Output, knowledgeGraph, hypothesisResult, simulationPlanResult, API_URL]);

  // Typing animation effect for markdown report
  useEffect(() => {
    if (reportResult && reportResult.report_markdown) {
      const fullText = reportResult.report_markdown;
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
  }, [reportResult]);

  // Handle click to skip typing animation
  const handleWindowClick = () => {
    if (!isTypingComplete && reportResult && reportResult.report_markdown) {
      // Stop the typing animation
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }
      // Show full text immediately
      setDisplayedText(reportResult.report_markdown);
      setIsTypingComplete(true);
    }
  };

  // Auto-scroll effect - scroll to bottom as text is typed
  useEffect(() => {
    if (scrollContainerRef.current && displayedText) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [displayedText]);

  if (!simulationResult || !step1Output || !knowledgeGraph || !hypothesisResult || !simulationPlanResult) {
    return (
      <div className="min-h-screen w-full bg-black flex items-center justify-center">
        <div className="text-white text-xl">Missing required data for report generation</div>
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
          title={!isTypingComplete ? 'Click to show full report' : ''}
        >
          {loading ? (
            <div className="h-full flex items-center justify-center">
              <ShinyText 
                text="Generating Report..." 
                disabled={false} 
                speed={3} 
                className='text-4xl' 
              />
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
          ) : reportResult && reportResult.report_markdown ? (
            <div className="h-full overflow-auto hide-scrollbar" ref={scrollContainerRef} style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
              <div className="prose prose-invert prose-lg max-w-none" style={{ color: '#e5e7eb' }}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    h1: ({node, ...props}) => <h1 className="text-3xl font-bold mb-4 mt-6 text-green-400" {...props} />,
                    h2: ({node, ...props}) => <h2 className="text-2xl font-bold mb-3 mt-5 text-green-400" {...props} />,
                    h3: ({node, ...props}) => <h3 className="text-xl font-semibold mb-2 mt-4 text-green-300" {...props} />,
                    p: ({node, ...props}) => <p className="mb-3 leading-relaxed" {...props} />,
                    ul: ({node, ...props}) => <ul className="mb-3 space-y-1" style={{ paddingLeft: '1.5rem', listStyleType: 'disc' }} {...props} />,
                    ol: ({node, ...props}) => <ol className="mb-3 space-y-1" style={{ paddingLeft: '1.5rem', listStyleType: 'decimal' }} {...props} />,
                    li: ({node, children, ...props}) => {
                      // Flatten paragraph content in list items to render inline
                      const flattenChildren = (children) => {
                        return React.Children.map(children, (child) => {
                          if (React.isValidElement(child) && child.type === 'p') {
                            return child.props.children;
                          }
                          return child;
                        });
                      };
                      return (
                        <li className="mb-1" style={{ display: 'list-item', listStylePosition: 'outside', paddingLeft: '0.5rem' }} {...props}>
                          {flattenChildren(children)}
                        </li>
                      );
                    },
                    code: ({node, ...props}) => <code className="bg-gray-800 px-1 py-0.5 rounded text-sm font-mono" {...props} />,
                    pre: ({node, ...props}) => <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto mb-3" {...props} />,
                    strong: ({node, ...props}) => <strong className="font-bold text-white" {...props} />,
                  }}
                >
                  {displayedText}
                </ReactMarkdown>
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
            text="Generating Report..." 
            disabled={false} 
            speed={3} 
            className='text-2xl' 
          />
        </div>
      ) : reportResult && reportResult.report_markdown ? (
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
              text="Report Generated" 
              disabled={false} 
              speed={3} 
              className='text-2xl' 
            />
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default ReportGeneratorPage;

