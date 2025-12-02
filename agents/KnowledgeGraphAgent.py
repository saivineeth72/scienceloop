#!/usr/bin/env python3
"""
KnowledgeGraphAgent - SpoonOS Agent

Builds a clean scientific knowledge graph from Step 1 JSON output.
Takes the output from PaperUnderstandingAgent and extracts:
- Scientific entities (variables, concepts, processes, conditions)
- Directional relationships between them
- Returns a knowledge graph as nodes and edges
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Any

# Use OpenAI client directly (same as PaperUnderstandingAgent)
import os
from openai import AsyncOpenAI

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on system environment variables


async def build_knowledge_graph(step1_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a knowledge graph from Step 1 JSON output.
    
    Args:
        step1_output: Dictionary from PaperUnderstandingAgent with:
            - summary
            - formulas
            - relationships
            - variables
            - key_ideas
    
    Returns:
        Dictionary with nodes and edges
    """
    # Prepare input for LLM
    input_text = f"""You are a Knowledge Graph Agent. Your job is to build a clean knowledge graph from the following information extracted from a research paper (any field: computer science, science, engineering, etc.).

STEP 1 OUTPUT:
{json.dumps(step1_output, indent=2)}

Your task:
1. Identify all important entities (use descriptive, human-readable names):
   - Use descriptive names for variables (e.g., "Temperature" not "T", "Folding rate" not "k", "Free energy change of folding" not "DG")
   - Concepts (from key ideas and summary - use full descriptive names)
   - Processes (from relationships and formulas - use descriptive names)
   - Conditions (from relationships - use descriptive names)

2. Identify all directional relationships between entities:
   - Prioritize relationships from the "relationships" list - these are the most important
   - Extract conceptual relationships from key ideas
   - Extract relationships from formulas but use descriptive entity names
   - Each relationship should be directional: source → relationship → target
   - Use meaningful relationship descriptions like: "is due to", "exhibits", "is caused by", "is modeled by", "is dependent on", "maintains", "drives", etc.

3. Build a knowledge graph with:
   - nodes: list of unique entity names (strings) - use descriptive, human-readable names
   - edges: list of relationship objects with source, relation, target

IMPORTANT RULES:
- Do NOT invent concepts not present in the input
- Only use entities and relationships that can be derived from the provided information
- Use DESCRIPTIVE names for entities (e.g., "Temperature" not "T", "Folding rate" not "rate" or "k")
- Convert variable names to their full descriptive meanings from the variables list
- Focus on conceptual relationships, not low-level formula dependencies
- Prioritize relationships from the "relationships" list in Step 1 output
- Each edge must have: source (descriptive entity name), relation (clear relationship description), target (descriptive entity name)

Return ONLY valid JSON in this exact format:
{{
  "nodes": ["entity1", "entity2", ...],
  "edges": [
    {{
      "source": "entity1",
      "relation": "relationship description",
      "target": "entity2"
    }},
    ...
  ]
}}

Return ONLY the JSON, no additional text or explanation."""

    # Use OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY not found in environment variables")
    
    client = AsyncOpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Knowledge Graph Agent. You build scientific knowledge graphs from structured data. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            temperature=0.3,
            max_tokens=4000
        )
    except Exception as e:
        # Fallback to gpt-4-turbo if model not available
        if "model" in str(e).lower() or "not found" in str(e).lower():
            print(f"Warning: {model} not available, trying gpt-4-turbo", file=sys.stderr)
            model = "gpt-4-turbo"
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Knowledge Graph Agent. You build scientific knowledge graphs from structured data. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": input_text
                    }
                ],
                temperature=0.3,
                max_tokens=4000
            )
        else:
            raise
    
    # Extract response
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
    
    # Validate and ensure required fields exist
    if "nodes" not in result or "edges" not in result:
        raise Exception("LLM response missing 'nodes' or 'edges' fields")
    
    # Ensure nodes is a list of strings
    if not isinstance(result["nodes"], list):
        result["nodes"] = []
    
    # Ensure edges is a list with proper structure
    if not isinstance(result["edges"], list):
        result["edges"] = []
    
    # Validate edge structure
    validated_edges = []
    for edge in result["edges"]:
        if isinstance(edge, dict) and "source" in edge and "relation" in edge and "target" in edge:
            validated_edges.append({
                "source": str(edge["source"]),
                "relation": str(edge["relation"]),
                "target": str(edge["target"])
            })
    
    result["edges"] = validated_edges
    
    return result


def main():
    """Main function to run the KnowledgeGraphAgent"""
    parser = argparse.ArgumentParser(
        description="Builds a scientific knowledge graph from Step 1 JSON output"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to JSON file from Step 1 (PaperUnderstandingAgent output)"
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON from stdin instead of file"
    )
    
    args = parser.parse_args()
    
    # Read input JSON
    try:
        if args.stdin:
            # Read from stdin
            input_text = sys.stdin.read()
            step1_output = json.loads(input_text)
        elif args.input:
            # Read from file
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"Error: Input file not found: {input_path}", file=sys.stderr)
                sys.exit(1)
            
            with open(input_path, 'r') as f:
                step1_output = json.load(f)
        else:
            print("Error: Either --input or --stdin must be specified", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        
        # Validate input structure
        required_fields = ["summary", "formulas", "relationships", "variables", "key_ideas"]
        for field in required_fields:
            if field not in step1_output:
                print(f"Warning: Input missing field '{field}'", file=sys.stderr)
        
        print("Building knowledge graph from Step 1 output...", file=sys.stderr)
        
        # Build knowledge graph
        result = asyncio.run(build_knowledge_graph(step1_output))
        
        # Determine output directory (same as input file directory)
        if args.stdin:
            # If reading from stdin, save to current directory
            output_dir = Path.cwd()
        else:
            # Use the same directory as the input file
            input_path = Path(args.input)
            output_dir = input_path.parent
        
        # Save output to knowledge_graph.json in the same directory
        output_path = output_dir / "knowledge_graph.json"
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Knowledge graph saved to: {output_path}", file=sys.stderr)
        print(json.dumps(result, indent=2))
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

