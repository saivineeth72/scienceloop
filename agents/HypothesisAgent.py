#!/usr/bin/env python3
"""
HypothesisAgent - SpoonOS Agent

Generates scientific hypotheses from:
- Step 1 output (PaperUnderstandingAgent): summary, formulas, relationships, variables, key_ideas
- Knowledge graph (KnowledgeGraphAgent): nodes and edges

Uses Claude 4.5 Sonnet to generate testable, grounded hypotheses.
"""

import sys
import json
import argparse
import asyncio
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


async def generate_hypothesis(step1_output: Dict[str, Any], knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a scientific hypothesis from Step 1 output and knowledge graph.
    
    Args:
        step1_output: Dictionary from PaperUnderstandingAgent
        knowledge_graph: Dictionary from KnowledgeGraphAgent with nodes and edges
    
    Returns:
        Dictionary with hypothesis and justification
    """
    # Prepare input for Claude
    input_text = f"""You are a Hypothesis Agent. Your job is to generate the best possible scientific hypothesis based on a research paper's content and its knowledge graph.

STEP 1 OUTPUT (Paper Understanding):
{json.dumps(step1_output, indent=2)}

KNOWLEDGE GRAPH:
{json.dumps(knowledge_graph, indent=2)}

Your task:
1. Understand the paper's concepts, formulas, relationships, and key ideas
2. Analyze the knowledge graph to identify causal links and relationships between entities
3. Generate a scientifically testable hypothesis that:
   - Is grounded STRICTLY in the paper (no hallucination - only use information from the provided data)
   - Is specific and measurable
   - Is simulation-friendly for next steps
   - Works for any scientific field (biology, chemistry, physics, computer science, ML, algorithms, systems, etc.)

The hypothesis should:
- Be a clear, testable statement
- Build on the relationships and concepts in the knowledge graph
- Be specific enough to be experimentally or computationally testable
- Follow logically from the paper's findings and relationships

Return ONLY valid JSON in this exact format:
{{
  "hypothesis": "A clear, testable scientific hypothesis statement based on the paper",
  "justification": "Explanation of why this hypothesis follows logically from the paper's content and knowledge graph, citing specific relationships, formulas, or concepts"
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
            max_tokens=4000,
            temperature=0.3,
            system="You are a Hypothesis Agent that generates scientifically testable hypotheses from research papers. Always respond with valid JSON only.",
            messages=[
                {"role": "user", "content": input_text}
            ]
        )
        
        llm_response = response.content[0].text
        
    except Exception as e:
        raise Exception(f"Claude API call failed: {e}")
    
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
    
    # Validate and ensure required fields exist
    if "hypothesis" not in result or "justification" not in result:
        raise Exception("LLM response missing 'hypothesis' or 'justification' fields")
    
    output = {
        "hypothesis": str(result.get("hypothesis", "")),
        "justification": str(result.get("justification", ""))
    }
    
    return output


def main():
    """Main function to run the HypothesisAgent"""
    parser = argparse.ArgumentParser(
        description="Generates scientific hypotheses from paper understanding and knowledge graph"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON file from Step 1 (PaperUnderstandingAgent output)"
    )
    parser.add_argument(
        "--kg",
        type=str,
        required=True,
        help="Path to JSON file from KnowledgeGraphAgent (knowledge graph with nodes and edges)"
    )
    
    args = parser.parse_args()
    
    # Read Step 1 input JSON
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        
        with open(input_path, 'r') as f:
            step1_output = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Read knowledge graph JSON
    try:
        kg_path = Path(args.kg)
        if not kg_path.exists():
            print(f"Error: Knowledge graph file not found: {kg_path}", file=sys.stderr)
            sys.exit(1)
        
        with open(kg_path, 'r') as f:
            knowledge_graph = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in knowledge graph file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading knowledge graph file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate knowledge graph structure
    if "nodes" not in knowledge_graph or "edges" not in knowledge_graph:
        print("Error: Knowledge graph must have 'nodes' and 'edges' fields", file=sys.stderr)
        sys.exit(1)
    
    try:
        print("Generating hypothesis from paper understanding and knowledge graph...", file=sys.stderr)
        
        # Generate hypothesis
        result = asyncio.run(generate_hypothesis(step1_output, knowledge_graph))
        
        # Determine output directory (same as input file directory)
        input_path = Path(args.input)
        output_dir = input_path.parent
        
        # Save output to hypothesis.json in the same directory
        output_path = output_dir / "hypothesis.json"
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Hypothesis saved to: {output_path}", file=sys.stderr)
        print(json.dumps(result, indent=2))
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

