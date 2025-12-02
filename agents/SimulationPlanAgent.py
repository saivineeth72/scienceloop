#!/usr/bin/env python3
"""
SimulationPlanAgent - SpoonOS Agent

Generates a precise, simulation-ready plan from:
- Hypothesis JSON (from HypothesisAgent)
- Step 1 output (PaperUnderstandingAgent): formulas, variables, relationships
- Knowledge graph (KnowledgeGraphAgent): nodes and edges

Uses Claude 4.5 Sonnet to create executable Python simulation plans.
"""

import sys
import json
import argparse
import asyncio
import re
from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
from anthropic import AsyncAnthropic


async def generate_simulation_plan(
    hypothesis: Dict[str, Any],
    step1_output: Dict[str, Any],
    knowledge_graph: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a simulation-ready plan from hypothesis, paper understanding, and knowledge graph.
    
    Args:
        hypothesis: Dictionary from HypothesisAgent with hypothesis and justification
        step1_output: Dictionary from PaperUnderstandingAgent with formulas, variables, relationships
        knowledge_graph: Dictionary from KnowledgeGraphAgent with nodes and edges
    
    Returns:
        Dictionary with simulation plan (equations, constants, variables, procedure, outcomes)
    """
    # Prepare input for Claude
    input_text = f"""You are a Simulation Plan Agent. Your job is to create a precise, executable simulation plan based on a scientific hypothesis, the paper's formulas, and the knowledge graph.

HYPOTHESIS:
{json.dumps(hypothesis, indent=2)}

STEP 1 OUTPUT (Paper Understanding - Formulas, Variables, Relationships):
{json.dumps(step1_output, indent=2)}

KNOWLEDGE GRAPH:
{json.dumps(knowledge_graph, indent=2)}

Your task:
1. Understand the hypothesis and how it relates to the formulas in the paper
2. Translate the hypothesis into a concrete, executable scientific simulation plan
3. The plan MUST be implementable in Python using NumPy/SciPy/Matplotlib
4. Be precise: no ambiguity, no guesswork
5. Must work for ALL scientific fields (biology, chemistry, physics, computer science, ML, algorithms, systems, etc.)

Requirements:
- Identify the exact equations that need to be implemented numerically
- List all constants required (with their typical values or ranges if known from the paper)
- Specify which variables to vary and their ranges
- Provide step-by-step procedure for running the simulation
- Describe expected outcomes/patterns the simulation should reveal

The simulation plan should:
- Be directly executable (clear enough for a developer to implement)
- Use the formulas from the paper
- Be grounded in the hypothesis
- Include numerical ranges and parameter sweeps
- Specify what to plot/visualize

Return ONLY valid JSON in this exact format:
{{
  "simulation_equations": [
    "equation1 in LaTeX or Python notation",
    "equation2 in LaTeX or Python notation"
  ],
  "constants_required": [
    {{
      "name": "constant_name",
      "description": "what it represents",
      "value_or_range": "specific value or range if known from paper, otherwise 'to be determined'"
    }}
  ],
  "variables_to_vary": [
    {{
      "name": "variable_name",
      "description": "what it represents",
      "range": "[min, max] or list of values",
      "units": "if applicable"
    }}
  ],
  "procedure_steps": [
    "Step 1: Initialize constants and set up parameter ranges",
    "Step 2: Implement the main simulation loop",
    "Step 3: Compute dependent variables using equations",
    "Step 4: Store results and generate visualizations"
  ],
  "expected_outcomes": "Description of what patterns, trends, or results the simulation should reveal based on the hypothesis"
}}

Return ONLY the JSON, no additional text or explanation."""

    # Use Claude/Anthropic API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise Exception("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env file.")
    
    client = AsyncAnthropic(api_key=api_key)
    # Use Claude Sonnet 4.5
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=6000,  # Larger context for detailed simulation plans
            temperature=0.2,  # Lower temperature for precision
            system="You are a Simulation Plan Agent that creates precise, executable simulation plans from scientific hypotheses. Always respond with valid JSON only.",
            messages=[
                {"role": "user", "content": input_text}
            ]
        )
        
        llm_response = response.content[0].text
        
    except Exception as e:
        raise Exception(f"Claude API call failed: {e}")
    
    # Parse JSON response
    try:
        # Try to extract JSON from response (in case there's extra text)
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = json.loads(llm_response)
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse was: {llm_response[:500]}")
    
    # Validate and ensure required fields exist
    required_fields = ["simulation_equations", "constants_required", "variables_to_vary", "procedure_steps", "expected_outcomes"]
    for field in required_fields:
        if field not in result:
            raise Exception(f"LLM response missing required field: {field}")
    
    # Ensure lists are actually lists
    if not isinstance(result["simulation_equations"], list):
        result["simulation_equations"] = []
    if not isinstance(result["constants_required"], list):
        result["constants_required"] = []
    if not isinstance(result["variables_to_vary"], list):
        result["variables_to_vary"] = []
    if not isinstance(result["procedure_steps"], list):
        result["procedure_steps"] = []
    if not isinstance(result["expected_outcomes"], str):
        result["expected_outcomes"] = str(result.get("expected_outcomes", ""))
    
    return result


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
    """Main function to run the SimulationPlanAgent"""
    parser = argparse.ArgumentParser(
        description="Generates a simulation-ready plan from hypothesis, paper understanding, and knowledge graph"
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        required=True,
        help="Path to JSON file from HypothesisAgent (hypothesis output)"
    )
    parser.add_argument(
        "--paper",
        type=str,
        required=True,
        help="Path to JSON file from PaperUnderstandingAgent (Step 1 output with formulas, variables, relationships)"
    )
    parser.add_argument(
        "--kg",
        type=str,
        required=True,
        help="Path to JSON file from KnowledgeGraphAgent (knowledge graph with nodes and edges)"
    )
    
    args = parser.parse_args()
    
    # Load all input files
    try:
        hypothesis = load_json_from_file(args.hypothesis)
    except Exception as e:
        print(f"Error loading hypothesis file: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        step1_output = load_json_from_file(args.paper)
    except Exception as e:
        print(f"Error loading paper file: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        knowledge_graph = load_json_from_file(args.kg)
    except Exception as e:
        print(f"Error loading knowledge graph file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate inputs
    if "hypothesis" not in hypothesis:
        print("Error: Hypothesis file must contain 'hypothesis' field", file=sys.stderr)
        sys.exit(1)
    
    if "formulas" not in step1_output or "variables" not in step1_output:
        print("Error: Paper file must contain 'formulas' and 'variables' fields", file=sys.stderr)
        sys.exit(1)
    
    if "nodes" not in knowledge_graph or "edges" not in knowledge_graph:
        print("Error: Knowledge graph must contain 'nodes' and 'edges' fields", file=sys.stderr)
        sys.exit(1)
    
    try:
        print("Generating simulation plan from hypothesis, paper understanding, and knowledge graph...", file=sys.stderr)
        
        # Generate simulation plan
        result = asyncio.run(generate_simulation_plan(hypothesis, step1_output, knowledge_graph))
        
        # Determine output directory (same as hypothesis file directory)
        hypothesis_path = Path(args.hypothesis)
        output_dir = hypothesis_path.parent
        
        # Save output to simulation_plan.json in the same directory
        output_path = output_dir / "simulation_plan.json"
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Simulation plan saved to: {output_path}", file=sys.stderr)
        print(json.dumps(result, indent=2))
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

