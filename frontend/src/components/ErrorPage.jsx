import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import ShinyText from './ShinyText';

const ErrorPage = () => {
  const location = useLocation();
  const { errorResult } = location.state || {};
  
  const [displayedText, setDisplayedText] = useState('');
  const [isTypingComplete, setIsTypingComplete] = useState(false);
  const scrollContainerRef = useRef(null);
  const typingIntervalRef = useRef(null);

  // Format error result as readable text
  const formatErrorAsText = (errorData) => {
    if (!errorData) return 'No error information available';
    
    let text = '';
    
    if (errorData.error_summary) {
      text += `Error Summary\n${'='.repeat(50)}\n`;
      text += `${errorData.error_summary}\n\n`;
    }
    
    if (errorData.stderr) {
      text += `Error Details\n${'='.repeat(50)}\n`;
      text += `${errorData.stderr}\n\n`;
    }
    
    if (errorData.stdout) {
      text += `Output\n${'='.repeat(50)}\n`;
      text += `${errorData.stdout}\n\n`;
    }
    
    if (errorData.exit_code !== undefined) {
      text += `Exit Code\n${'='.repeat(50)}\n`;
      text += `${errorData.exit_code}\n`;
    }
    
    return text;
  };

  // Typing animation effect
  useEffect(() => {
    if (errorResult) {
      const fullText = formatErrorAsText(errorResult);
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
  }, [errorResult]);

  // Handle click to skip typing animation
  const handleWindowClick = () => {
    if (!isTypingComplete && errorResult) {
      // Stop the typing animation
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }
      // Show full text immediately
      const fullText = formatErrorAsText(errorResult);
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

  if (!errorResult) {
    return (
      <div className="min-h-screen w-full bg-black flex items-center justify-center">
        <div className="text-white text-xl">No error data available</div>
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
          title={!isTypingComplete ? 'Click to show full error' : ''}
        >
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
                          <h2 className="font-bold text-lg mt-3 mb-1" style={{ fontWeight: 'bold', color: '#ef4444' }}>{line}</h2>
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
        <div className="w-full max-w-4xl border-t" style={{ borderColor: '#ef4444' }}></div>
      </div>

      {/* Status text below the window */}
      {errorResult && (
        <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
          <div className="flex items-center gap-3">
            <svg
              className="w-6 h-6 text-red-400"
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
            <ShinyText 
              text="Simulation Error" 
              disabled={false} 
              speed={3} 
              className='text-2xl text-red-400' 
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ErrorPage;

