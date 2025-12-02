import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

const FileUpload = () => {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files).filter(file => 
      file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
    );
    setFiles((prevFiles) => [...prevFiles, ...droppedFiles]);
  };

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files).filter(file => 
      file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
    );
    setFiles((prevFiles) => [...prevFiles, ...selectedFiles]);
  };

  const handleRemoveFile = (index) => {
    setFiles((prevFiles) => prevFiles.filter((_, i) => i !== index));
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const handleProcessFile = (file) => {
    navigate('/results', { state: { file } });
  };

  const handleAnalyze = () => {
    if (files.length > 0) {
      handleProcessFile(files[0]);
    }
  };

  return (
    <div className="w-full flex flex-col">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        className="hidden"
        onChange={handleFileSelect}
      />

      {/* Top Section - File Upload Area */}
      <div
        className="w-full flex flex-col items-center justify-center py-8"
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Upload Icon */}
        <svg
          className="w-16 h-16 mb-6"
          fill="none"
          stroke="#00A86B"
          viewBox="0 0 24 24"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
          />
        </svg>
        
        {/* Drag and Drop text */}
        <p className="text-gray-900 text-center mb-2 text-2xl font-light">
          {isDragging ? 'Drop file' : 'Drag and Drop file'}
        </p>
        
        {/* Or text */}
        <p className="text-gray-900 text-center mb-4 text-2xl font-light">
          or
        </p>
        
        {/* Browse Button */}
        <button
          onClick={handleClick}
          className="text-white font-light text-xl px-8 py-3 rounded-lg transition-colors"
          style={{ backgroundColor: '#00A86B' }}
          onMouseEnter={(e) => e.target.style.backgroundColor = '#008B5A'}
          onMouseLeave={(e) => e.target.style.backgroundColor = '#00A86B'}
        >
          Browse
        </button>
      </div>

      {/* Divider Line */}
      <div className="w-1/3 mx-auto border-t border-gray-300 my-4"></div>

      {/* Bottom Section - Uploaded Files */}
      <div className="w-full flex flex-col items-center">
        <div className="space-y-8 max-h-96 overflow-y-auto mb-16 py-4" style={{ width: '60%' }}>
          {files.length > 0 ? (
            files.map((file, index) => {
              return (
                <div
                  key={index}
                  className="flex items-center justify-between py-6 px-4 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors"
                >
                  <div className="flex items-center flex-1 min-w-0">
                    <svg
                      className="w-5 h-5 text-gray-400 mr-3 flex-shrink-0"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {file.name}
                      </p>
                      <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemoveFile(index);
                    }}
                    className="text-red-500 hover:text-red-700 transition-colors ml-2"
                    aria-label="Remove file"
                  >
                    <svg
                      className="w-5 h-5"
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
              );
            })
          ) : (
            <p className="text-gray-400 text-center py-4 text-sm">No files uploaded</p>
          )}
        </div>

        {/* Analyze Button */}
        <div className="flex justify-center">
          <button
            onClick={handleAnalyze}
            disabled={files.length === 0}
            className={`px-12 py-3 rounded-lg transition-colors font-light text-lg ${
              files.length === 0
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-black hover:bg-gray-800 text-white'
            }`}
          >
            Analyze
          </button>
        </div>
      </div>
    </div>
  );
};

export default FileUpload;

