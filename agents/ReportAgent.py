#!/usr/bin/env python3
"""
ReportAgent - SpoonOS Agent

Analyzes simulation results and generates comprehensive reports comparing
expected outcomes with actual results. Uses Claude 4.5 Sonnet for reasoning.
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
from anthropic import AsyncAnthropic


async def generate_report(
    paper_understanding: Dict[str, Any],
    knowledge_graph: Dict[str, Any],
    hypothesis: Dict[str, Any],
    simulation_plan: Dict[str, Any],
    simulation_result: Dict[str, Any],
    error_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive report analyzing simulation results.
    
    Args:
        paper_understanding: JSON from PaperUnderstandingAgent
        knowledge_graph: JSON from KnowledgeGraphAgent
        hypothesis: JSON from HypothesisAgent
        simulation_plan: JSON from SimulationPlanAgent
        simulation_result: JSON from SimulationRunnerAgent
        error_history: Optional list of past error reports
        
    Returns:
        Dictionary with report_markdown and summary
    """
    # Prepare input for Claude
    input_text = f"""You are a ReportAgent analyzing scientific simulation results. Your task is to generate a comprehensive, well-reasoned report comparing expected outcomes with actual results.

PAPER CONTEXT:
Paper Understanding:
{json.dumps(paper_understanding, indent=2)}

Knowledge Graph:
{json.dumps(knowledge_graph, indent=2)}

HYPOTHESIS:
{json.dumps(hypothesis, indent=2)}

SIMULATION PLAN:
Expected Outcomes: {simulation_plan.get('expected_outcomes', 'Not specified')}
Simulation Equations: {json.dumps(simulation_plan.get('simulation_equations', []), indent=2)}
Variables to Vary: {json.dumps(simulation_plan.get('variables_to_vary', []), indent=2)}
Procedure Steps: {json.dumps(simulation_plan.get('procedure_steps', []), indent=2)}

SIMULATION RESULT:
Status: {simulation_result.get('status', 'unknown')}
Exit Code: {simulation_result.get('exit_code', -1)}
Run ID: {simulation_result.get('run_id', 'unknown')}
Artifacts Generated: {json.dumps([{'filename': art.get('filename'), 'path': art.get('path')} for art in simulation_result.get('artifacts', [])], indent=2)}

ARTIFACT FILE CONTENTS (only results.csv):
{chr(10).join([f"=== {art.get('filename', 'unknown')} ===" + chr(10) + art.get('content', '[No content]') + chr(10) for art in simulation_result.get('artifacts', []) if art.get('filename') == 'results.csv' and art.get('content')]) or 'No results.csv content found'}

STDOUT (simulation output):
{simulation_result.get('stdout', 'No output')}

STDERR (errors/warnings):
{simulation_result.get('stderr', 'No errors')}

ERROR HISTORY (if any):
{json.dumps(error_history if error_history else [], indent=2)}

Your task:

1. ANALYZE THE RESULTS:
   - If status == "error": Clearly state we did NOT reach expected outcomes and explain why based on stderr/logs
   - If status == "success": Analyze stdout and artifacts to determine if patterns described in expected_outcomes were observed
   - Look for evidence in stdout (numbers, metrics, convergence messages, etc.)
   - Reference artifacts by filename (e.g., plots, CSV files)

2. GENERATE A HUMAN-READABLE MARKDOWN REPORT with these sections:
   - Title: "Simulation Results Report"
   - Short Experiment Summary (2-3 sentences)
   - Section: "Goal & Hypothesis" (from hypothesis + paper context)
   - Section: "What We Planned to See" (from expected_outcomes)
   - Section: "What We Actually Observed"
     * Reference stdout (numbers, metrics) if present
     * Reference artifacts by filename (e.g., plots)
   - Section: "Did We Meet the Expected Outcome?"
     * Explicit YES / PARTIALLY / NO
     * A short paragraph of reasoning
   - Section: "Key Notes About the Simulation"
     * Bullet points with interesting behaviors, edge cases, limitations
   - Section: "Next Steps / Recommendations"
     * What to tweak in code, plan, or hypothesis
     * If more runs are needed, or different parameter sweeps

3. GENERATE A MACHINE-READABLE SUMMARY JSON:
{{
  "success": true | false | "partial",
  "reason": "<one-paragraph explanation>",
  "matched_expectations": [
     "<which parts of expected_outcomes were clearly observed>"
  ],
  "unmet_expectations": [
     "<which expected behaviors were not clearly observed or contradicted>"
  ],
  "key_observations": [
     "<short bullet strings for main findings>"
  ],
  "recommendations": [
     "<concrete next actions for future runs or code changes>"
  ]
}}

Return ONLY valid JSON in this exact format:
{{
  "report_markdown": "# Simulation Results Report\\n\\n[full markdown report here]",
  "summary": {{
    "success": true | false | "partial",
    "reason": "...",
    "matched_expectations": [...],
    "unmet_expectations": [...],
    "key_observations": [...],
    "recommendations": [...]
  }}
}}

Focus on strong, explicit reasoning that can be read and understood quickly. Be specific about what evidence supports or contradicts the expected outcomes."""

    # Use Anthropic Claude API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise Exception("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env file.")
    
    client = AsyncAnthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    
    # Estimate token count (rough: 1 token ≈ 4 characters)
    estimated_tokens = len(input_text) // 4
    print(f"Estimated prompt tokens: ~{estimated_tokens}", file=sys.stderr)
    
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=8192,  # Increased to allow full report generation
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        )
        
        # Extract text from response
        llm_response = ""
        for content_block in response.content:
            if hasattr(content_block, 'text'):
                llm_response += content_block.text
        
        # Check if response was truncated
        stop_reason = getattr(response, 'stop_reason', None)
        if stop_reason == 'max_tokens':
            print(f"Warning: Response was truncated due to max_tokens limit. Response length: {len(llm_response)}", file=sys.stderr)
            print(f"Consider increasing max_tokens or simplifying the prompt.", file=sys.stderr)
        
        # Parse JSON from response
        try:
            # Clean up the response - remove markdown code blocks if present
            import re
            # Remove ```json and ``` markers
            cleaned_response = re.sub(r'^```json\s*\n?', '', llm_response, flags=re.MULTILINE)
            cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response, flags=re.MULTILINE)
            
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', cleaned_response)
            if json_match:
                json_str = json_match.group(0)
                # Try to fix incomplete JSON if truncated
                if stop_reason == 'max_tokens':
                    # Count braces to see if JSON is incomplete
                    open_braces = json_str.count('{')
                    close_braces = json_str.count('}')
                    if open_braces > close_braces:
                        # Try to close incomplete strings and JSON structure
                        # Find the last unclosed string in report_markdown
                        if '"report_markdown"' in json_str and '"summary"' not in json_str:
                            # Find where report_markdown value starts
                            markdown_start = json_str.find('"report_markdown"')
                            if markdown_start != -1:
                                # Find the colon and opening quote
                                value_start = json_str.find('"', markdown_start + len('"report_markdown"'))
                                if value_start != -1:
                                    value_start += 1  # Skip opening quote
                                    # Check if string is properly closed
                                    remaining = json_str[value_start:]
                                    # Count escaped quotes vs unescaped quotes
                                    unescaped_quotes = 0
                                    i = 0
                                    while i < len(remaining):
                                        if remaining[i] == '\\':
                                            i += 2  # Skip escaped character
                                        elif remaining[i] == '"':
                                            unescaped_quotes += 1
                                            if unescaped_quotes == 1:
                                                # This might be the closing quote, but check if summary follows
                                                if i + 1 < len(remaining) and remaining[i+1:i+10] != ',\n  "sum':
                                                    # String is not closed, close it
                                                    json_str = json_str[:value_start + i + 1] + '",\n  "summary": {\n    "success": false,\n    "reason": "Response was truncated",\n    "matched_expectations": [],\n    "unmet_expectations": [],\n    "key_observations": [],\n    "recommendations": []\n  }\n}'
                                                    break
                                            i += 1
                                        else:
                                            i += 1
                
                result = json.loads(json_str)
            else:
                # Try parsing the whole response
                result = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse LLM response as JSON: {e}", file=sys.stderr)
            print(f"Raw response length: {len(llm_response)} characters", file=sys.stderr)
            print(f"Stop reason: {stop_reason}", file=sys.stderr)
            print(f"Raw response (first 2000 chars):\n{llm_response[:2000]}", file=sys.stderr)
            print(f"Raw response (last 500 chars):\n{llm_response[-500:]}", file=sys.stderr)
            
            # Try to extract markdown content even if JSON is malformed
            # Look for report_markdown field and extract its value
            markdown_pattern = r'"report_markdown"\s*:\s*"(.*?)"(?:\s*,|\s*\})'
            markdown_match = re.search(markdown_pattern, llm_response, re.DOTALL)
            if not markdown_match:
                # Try without the closing quote (truncated case)
                markdown_pattern2 = r'"report_markdown"\s*:\s*"(.*)'
                markdown_match = re.search(markdown_pattern2, llm_response, re.DOTALL)
            
            if markdown_match:
                # Unescape the markdown
                markdown_content = markdown_match.group(1)
                # Handle escaped characters
                markdown_content = markdown_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                return {
                    "report_markdown": markdown_content,
                    "summary": {
                        "success": False,
                        "reason": f"Failed to parse full JSON response. Response may have been truncated. Error: {str(e)}",
                        "matched_expectations": [],
                        "unmet_expectations": [],
                        "key_observations": [],
                        "recommendations": []
                    }
                }
            
            # Return error structure
            return {
                "report_markdown": f"# Error Generating Report\n\nFailed to parse LLM response as JSON.\n\nError: {str(e)}\n\nStop reason: {stop_reason}\n\nRaw response (first 2000 chars):\n```json\n{llm_response[:2000]}\n```",
                "summary": {
                    "success": False,
                    "reason": f"Failed to parse report from LLM response: {str(e)}",
                    "matched_expectations": [],
                    "unmet_expectations": [],
                    "key_observations": [],
                    "recommendations": []
                }
            }
        
        return result
        
    except Exception as e:
        print(f"Error calling Claude API: {e}", file=sys.stderr)
        raise


def load_json_from_file(file_path: str) -> dict:
    """Load JSON from a file path."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")


def main():
    """Main entry point for the agent."""
    parser = argparse.ArgumentParser(
        description="ReportAgent - Generate comprehensive simulation reports"
    )
    parser.add_argument(
        "--paper-understanding",
        type=str,
        help="Path to JSON file from PaperUnderstandingAgent"
    )
    parser.add_argument(
        "--knowledge-graph",
        type=str,
        help="Path to JSON file from KnowledgeGraphAgent"
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        help="Path to JSON file from HypothesisAgent"
    )
    parser.add_argument(
        "--simulation-plan",
        type=str,
        help="Path to JSON file from SimulationPlanAgent"
    )
    parser.add_argument(
        "--simulation-result",
        type=str,
        help="Path to JSON file from SimulationRunnerAgent result"
    )
    parser.add_argument(
        "--error-history",
        type=str,
        help="Path to JSON file containing list of error reports (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the report JSON (default: print to stdout)"
    )
    parser.add_argument(
        "--markdown-output",
        type=str,
        help="Path to save the markdown report (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load all JSON inputs
        paper_understanding = load_json_from_file(args.paper_understanding)
        knowledge_graph = load_json_from_file(args.knowledge_graph)
        hypothesis = load_json_from_file(args.hypothesis)
        simulation_plan = load_json_from_file(args.simulation_plan)
        simulation_result = load_json_from_file(args.simulation_result)
        
        error_history = None
        if args.error_history:
            error_history = load_json_from_file(args.error_history)
        
        # Generate report
        print("Generating report...", file=sys.stderr)
        result = asyncio.run(generate_report(
            paper_understanding=paper_understanding,
            knowledge_graph=knowledge_graph,
            hypothesis=hypothesis,
            simulation_plan=simulation_plan,
            simulation_result=simulation_result,
            error_history=error_history
        ))
        
        # Save outputs
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Report JSON saved to: {output_path}", file=sys.stderr)
        
        if args.markdown_output:
            markdown_path = Path(args.markdown_output)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            with open(markdown_path, 'w') as f:
                f.write(result["report_markdown"])
            print(f"Markdown report saved to: {markdown_path}", file=sys.stderr)
        
        # Print JSON result to stdout
        print(json.dumps(result, indent=2))
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
