#!/usr/bin/env python3
"""
PaperUnderstandingAgent - SpoonOS Agent

Analyzes scientific PDFs by:
- Using SpoonOS PDFReaderTool to load raw PDF text (no extraction or parsing)
- Sending entire text to LLM for understanding
- LLM identifies formulas, variables, relationships, and key ideas
- Returns structured JSON output
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path

from spoon_ai.agents.custom_agent import CustomAgent
from spoon_ai.chat import ChatBot

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.pdf_reader_tool import PDFReaderTool


class PaperUnderstandingAgent(CustomAgent):
    """Agent for understanding scientific papers from PDFs"""
    
    name: str = "PaperUnderstandingAgent"
    description: str = "Analyzes scientific PDFs to extract formulas, variables, relationships, and key ideas"
    
    system_prompt: str = """You are a research paper analysis assistant for any field (science, computer science, engineering, etc.). Your task is to analyze raw PDF text and extract:
1. Formulas - mathematical equations, algorithms, pseudocode, complexity notation, or any formal expressions
2. Variables - all variables, parameters, constants, and their meanings/definitions (mathematical, algorithmic, system parameters, etc.)
3. Relationships - relationships, dependencies, connections, and interactions between concepts, systems, algorithms, or entities
4. Key Ideas - main ideas, hypotheses, findings, contributions, methodologies, or innovations

IMPORTANT: Do NOT manually extract or parse anything. Use your understanding capabilities to:
- Recognize formulas, algorithms, and formal expressions even if they're in plain text format
- Understand variable names and their contexts (mathematical, algorithmic, or system-related)
- Identify relationships from the narrative and context (causal, dependency, interaction, etc.)
- Extract key ideas from the full text (works for any field: CS, physics, biology, engineering, etc.)

Always respond with valid JSON in this format:
{
  "summary": "A brief summary of the paper (2-3 sentences)",
  "formulas": ["formula/algorithm 1", "formula/algorithm 2", ...],
  "relationships": ["relationship 1", "relationship 2", ...],
  "variables": ["variable 1: description", "variable 2: description", ...],
  "key_ideas": ["idea 1", "idea 2", ...]
}"""


async def analyze_pdf_with_agent(pdf_path: str) -> dict:
    """
    Analyze a PDF using the PaperUnderstandingAgent.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with analysis results
    """
    # Create agent with PDF reader tool
    agent = PaperUnderstandingAgent()
    agent.add_tool(PDFReaderTool())
    
    # Create prompt for the agent
    prompt = f"""Please analyze the scientific paper at this path: {pdf_path}

Use the read_pdf tool to load the raw PDF text, then analyze it to extract:
- Formulas
- Variables with descriptions
- Scientific relationships
- Key ideas
- A brief summary

Return your analysis as valid JSON with the required format."""
    
    # Run the agent
    result = await agent.run(prompt)
    
    # Try to extract JSON from the result
    try:
        # If result is already JSON, parse it
        if isinstance(result, str):
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                return json.loads(json_match.group(0))
            # If no JSON found, try parsing the whole string
            try:
                return json.loads(result)
            except:
                pass
        
        # If result is a dict, return it
        if isinstance(result, dict):
            return result
            
    except Exception as e:
        print(f"Warning: Could not parse agent result as JSON: {e}", file=sys.stderr)
        print(f"Raw result: {result}", file=sys.stderr)
    
    # Fallback: return error structure
    return {
        "summary": "Error parsing agent response",
        "formulas": [],
        "relationships": [],
        "variables": [],
        "key_ideas": []
    }


async def analyze_pdf_direct(pdf_path: str) -> dict:
    """
    Direct PDF analysis without agent (fallback method).
    Uses the tool directly and sends to LLM.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with analysis results
    """
    # Use the PDF tool directly
    pdf_tool = PDFReaderTool()
    tool_result = await pdf_tool.execute(pdf_path=pdf_path)
    
    if tool_result.error:
        raise Exception(f"PDF reading failed: {tool_result.error}")
    
    raw_text = tool_result.output
    
    if not raw_text or len(raw_text.strip()) == 0:
        raise Exception("PDF appears to be empty or could not extract text")
    
    print(f"PDF text loaded ({len(raw_text)} characters)", file=sys.stderr)
    
    # Truncate text if too long (gpt-4-turbo has ~128k context, ~1 token ≈ 4 chars)
    # Reserve ~28k chars for prompt template and response, so max ~100k chars for PDF text
    max_pdf_chars = 100000  # Increased limit for longer papers
    if len(raw_text) > max_pdf_chars:
        print(f"Warning: PDF text is too long ({len(raw_text)} chars), truncating to {max_pdf_chars} chars", file=sys.stderr)
        raw_text = raw_text[:max_pdf_chars] + "\n\n[Text truncated due to length...]"
    
    print("Sending to LLM for analysis...\n", file=sys.stderr)
    
    # Prepare prompt for LLM
    prompt = f"""You are analyzing a research paper from any field (science, computer science, engineering, etc.). Below is the raw text extracted from a PDF. 

Your task is to understand the paper's content and extract:
1. Formulas - mathematical equations, algorithms, pseudocode, complexity notation, or any formal expressions used in the paper
2. Variables - all variables, parameters, constants, and their meanings/definitions (including code variables, algorithmic variables, system parameters, etc.)
3. Relationships - relationships, dependencies, connections, and interactions between concepts, systems, algorithms, or entities
4. Key Ideas - main ideas, hypotheses, findings, contributions, methodologies, or innovations

IMPORTANT: Do NOT manually extract or parse anything. Use your understanding capabilities to:
- Recognize formulas, algorithms, and formal expressions even if they're in plain text format
- Understand variable names and their contexts (whether mathematical, algorithmic, or system-related)
- Identify relationships from the narrative and context (causal, dependency, interaction, etc.)
- Extract key ideas from the full text (works for any field: CS, physics, biology, engineering, etc.)

PDF Text:
{raw_text}

Please provide your analysis in the following JSON format:
{{
  "summary": "A brief scientific summary of the paper (2-3 sentences)",
  "formulas": ["formula 1", "formula 2", ...],
  "relationships": ["relationship 1", "relationship 2", ...],
  "variables": ["variable 1: description", "variable 2: description", ...],
  "key_ideas": ["idea 1", "idea 2", ...]
}}

Return ONLY valid JSON, no additional text."""
    
    # Use OpenAI client directly (workaround for SpoonOS tool_choice bug)
    import os
    from openai import AsyncOpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY not found in environment variables")
    
    client = AsyncOpenAI(api_key=api_key)
    # Use gpt-4-turbo for paper understanding
    model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    
    # Conservative max_tokens for response
    max_tokens = 3000
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a research paper analysis assistant for any field. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_tokens
        )
    except Exception as e:
        # If model not available, try gpt-4-turbo
        if "model" in str(e).lower() or "not found" in str(e).lower():
            print(f"Warning: {model} not available, trying gpt-4-turbo", file=sys.stderr)
            model = "gpt-4-turbo"
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a research paper analysis assistant for any field. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
        else:
            raise
    
    # Extract text from response
    llm_response = response.choices[0].message.content
    
    # Parse JSON response
    try:
        result = json.loads(llm_response)
    except json.JSONDecodeError:
        # Try to extract JSON from response if it's wrapped in markdown or text
        import re
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            raise Exception("LLM response is not valid JSON")
    
    # Ensure all required fields exist
    output = {
        "summary": result.get("summary", ""),
        "formulas": result.get("formulas", []) if isinstance(result.get("formulas"), list) else [],
        "relationships": result.get("relationships", []) if isinstance(result.get("relationships"), list) else [],
        "variables": result.get("variables", []) if isinstance(result.get("variables"), list) else [],
        "key_ideas": result.get("key_ideas", []) if isinstance(result.get("key_ideas"), list) else []
    }
    
    return output


def main():
    """Main function to run the PaperUnderstandingAgent"""
    parser = argparse.ArgumentParser(
        description="Analyzes scientific PDFs to extract formulas, variables, relationships, and key ideas"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to PDF file (relative to project root or absolute path)"
    )
    parser.add_argument(
        "--use-agent",
        action="store_true",
        help="Use SpoonOS agent with tools (default: direct method)"
    )
    
    args = parser.parse_args()
    
    # Resolve PDF path
    project_root = Path(__file__).parent.parent
    pdf_path = Path(args.pdf)
    
    if not pdf_path.is_absolute():
        pdf_path = project_root / pdf_path
    
    # Check if file exists
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        print(f"Reading PDF: {pdf_path}", file=sys.stderr)
        
        # Choose method
        if args.use_agent:
            print("Using SpoonOS agent with tools...", file=sys.stderr)
            result = asyncio.run(analyze_pdf_with_agent(str(pdf_path)))
        else:
            print("Using direct method...", file=sys.stderr)
            result = asyncio.run(analyze_pdf_direct(str(pdf_path)))
        
        # Save output to understand.json in the same directory as the PDF
        pdf_dir = pdf_path.parent
        output_path = pdf_dir / "understand.json"
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Output saved to: {output_path}", file=sys.stderr)
        print(json.dumps(result, indent=2))
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
