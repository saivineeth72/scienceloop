"""
PDF Reader Tool for SpoonOS
Reads a PDF file and extracts raw text content using PyPDF2
"""
from pathlib import Path
from typing import Any

from spoon_ai.tools import BaseTool
from spoon_ai.tools.base import ToolResult

try:
    from PyPDF2 import PdfReader
except ImportError:
    try:
        import pypdf
        PdfReader = pypdf.PdfReader
    except ImportError:
        raise ImportError("PyPDF2 or pypdf is required. Install with: pip install PyPDF2")


class PDFReaderTool(BaseTool):
    """Tool to read PDF files and extract raw text content"""
    
    name: str = "read_pdf"
    description: str = "Reads a PDF file and extracts raw text content from all pages. Returns the complete text without any parsing or extraction - just raw text."
    parameters: dict = {
        "type": "object",
        "properties": {
            "pdf_path": {
                "type": "string",
                "description": "Path to the PDF file (relative to project root or absolute path)"
            }
        },
        "required": ["pdf_path"]
    }
    
    async def execute(self, pdf_path: str, **kwargs) -> ToolResult:
        """
        Execute the PDF reading tool.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ToolResult with the raw text content
        """
        try:
            pdf_path_obj = Path(pdf_path)
            
            # Check if file exists
            if not pdf_path_obj.exists():
                return ToolResult(
                    error=f"PDF file not found: {pdf_path}",
                    system="FileNotFoundError"
                )
            
            # Read the PDF file
            reader = PdfReader(str(pdf_path_obj))
            
            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            # Join all pages with newlines
            raw_text = "\n".join(text_parts)
            
            if not raw_text or len(raw_text.strip()) == 0:
                return ToolResult(
                    error="PDF appears to be empty or could not extract text",
                    system="EmptyPDFError"
                )
            
            return ToolResult(
                output=raw_text,
                system=f"Successfully extracted {len(raw_text)} characters from PDF"
            )
            
        except Exception as e:
            return ToolResult(
                error=f"Error reading PDF: {str(e)}",
                system="PDFReadError"
            )

