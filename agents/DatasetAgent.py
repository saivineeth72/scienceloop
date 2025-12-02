#!/usr/bin/env python3
"""
DatasetAgent - SpoonOS Agent

Automatically provides ALL data required for generated simulation code to run.
Analyzes simulation plans and generates/downloads datasets as needed.

Uses Claude 4.5 Sonnet for structured reasoning about dataset requirements.
"""

import sys
import json
import argparse
import asyncio
import re
import ast
from pathlib import Path
from typing import Dict, Any, List
import shutil

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
from anthropic import AsyncAnthropic

# For dataset generation
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    print("Warning: networkx not installed. Graph dataset generation may fail.", file=sys.stderr)
    HAS_NETWORKX = False
    nx = None

# Numpy is required for synthetic dataset generation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("ERROR: numpy is required for synthetic dataset generation. Please install: pip install numpy", file=sys.stderr)
    HAS_NUMPY = False
    np = None

try:
    import requests
except ImportError:
    print("Warning: requests not installed. Dataset downloads may fail.", file=sys.stderr)


async def analyze_dataset_requirements(simulation_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use Claude to analyze simulation plan and determine dataset requirements.
    
    Args:
        simulation_plan: Dictionary from SimulationPlanAgent
    
    Returns:
        Dictionary with dataset_type and dataset specifications
    """
    # Prepare input for Claude
    input_text = f"""You are a Dataset Agent. Your job is to analyze a simulation plan and determine what datasets are required.

SIMULATION PLAN:
{json.dumps(simulation_plan, indent=2)}

Your task:
1. Analyze the simulation_equations, variables_to_vary, and procedure_steps
2. If ANY mention of data, datasets, loading data, generating data, sequences, samples, etc. appears:
   → Generate appropriate dataset files (even if code might generate inline, provide files for consistency)

3. Determine the dataset_type based on what data is mentioned:

   - If terms like "network", "topology", "scale_free", "small_world", "random",
     "Gamma(i)", "neighbors", "k-shell", "graph", "degree", "nodes", "edges" appear
     → dataset_type = "graph"

   - If ML dataset names appear (MNIST, CIFAR, ImageNet, training data, test data)
     → dataset_type = "ml"

   - If chemistry/bio mentions "molecular structure", "PDB", "protein structure"
     → dataset_type = "bio_structures"

   - If mentions "peptide", "sequence", "amino acid", "sequences", "samples", "data points",
     "classification data", "labeled data", "synthetic data", "generate data"
     → dataset_type = "synthetic" (generate synthetic dataset files)

   - If mentions time series, signals, measurements, observations
     → dataset_type = "synthetic" (generate synthetic time series)

   - If nothing about data is mentioned:
     → dataset_type = "none"

3. For each dataset_type, specify what needs to be generated:

   If dataset_type = "graph":
      - Determine which graph types are needed:
        * "barabasi_albert" if scale-free is mentioned
        * "watts_strogatz" if small_world is mentioned
        * "erdos_renyi" if random is mentioned
      - Specify parameters (n_nodes, n_edges, etc.) if mentioned in the plan
      - Check if "real_world" networks are needed

   If dataset_type = "ml":
      - Specify which ML dataset (MNIST, CIFAR, etc.)
      - Specify subset size if mentioned

   If dataset_type = "bio_structures":
      - Specify PDB IDs if mentioned, or suggest common ones (1CRN, 1UBQ, etc.)

   If dataset_type = "synthetic":
      - Determine what type: peptide_sequences, time_series, classification_data, regression_data, etc.
      - Extract parameters from the plan:
        * num_samples (how many data points)
        * sequence_length (if sequences)
        * num_features (if feature vectors)
        * num_classes (if classification)
        * file_format (CSV, JSON, TXT, etc.)
      - Specify all parameters needed to generate the dataset

   If dataset_type = "none":
      - Return empty datasets list

Return ONLY valid JSON in this exact format:
{{
  "dataset_type": "graph|ml|bio_structures|none",
  "graph_types": ["barabasi_albert", "watts_strogatz", ...] or [],
  "graph_params": {{
    "n_nodes": 1000,
    "n_edges": 3000,
    ...
  }},
  "real_world_graphs": ["karate", "dolphins", ...] or [],
  "ml_dataset": "MNIST" or null,
  "pdb_ids": ["1CRN", ...] or [],
  "synthetic_type": "peptide_sequences|time_series|classification_data|regression_data|other" or null,
  "synthetic_params": {{
    "num_samples": 1000,
    "sequence_length": 9,
    "num_features": 20,
    "num_classes": 2,
    "file_format": "CSV"
  }} or {{}},
  "reasoning": "Brief explanation of why this dataset_type was chosen and what data will be generated"
}}

Return ONLY the JSON, no additional text or explanation."""

    # Use Claude/Anthropic API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise Exception("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env file.")
    
    client = AsyncAnthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.2,  # Low temperature for precise analysis
            system="You are a Dataset Agent that analyzes simulation plans to determine dataset requirements. Always respond with valid JSON only.",
            messages=[
                {"role": "user", "content": input_text}
            ]
        )
        
        llm_response = response.content[0].text
        
    except Exception as e:
        raise Exception(f"Claude API call failed: {e}")
    
    # Parse JSON response
    try:
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = json.loads(llm_response)
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse was: {llm_response[:500]}")
    
    return result


def generate_graph_datasets(dataset_spec: Dict[str, Any], datasets_dir: Path) -> Dict[str, str]:
    """
    Generate graph datasets based on specification.
    
    Returns:
        Dictionary mapping dataset names to file paths
    """
    if not HAS_NETWORKX:
        print("Error: networkx is required for graph generation. Please install: pip install networkx", file=sys.stderr)
        return {}
    
    datasets = {}
    graph_types = dataset_spec.get("graph_types", [])
    graph_params = dataset_spec.get("graph_params", {})
    real_world = dataset_spec.get("real_world_graphs", [])
    
    # Handle n_nodes - could be a list or single value
    n_nodes_raw = graph_params.get("n_nodes", 1000)
    if isinstance(n_nodes_raw, list):
        n_nodes = n_nodes_raw[0] if n_nodes_raw else 1000  # Use first value
    else:
        n_nodes = int(n_nodes_raw)
    
    # Generate synthetic graphs
    for graph_type in graph_types:
        try:
            if graph_type == "barabasi_albert":
                m_raw = graph_params.get("m", 3)
                m = int(m_raw[0] if isinstance(m_raw, list) else m_raw)
                G = nx.barabasi_albert_graph(n_nodes, m)
                filename = f"barabasi_albert_n{n_nodes}_m{m}.edgelist"
            elif graph_type == "watts_strogatz":
                k_raw = graph_params.get("k", 6)
                k = int(k_raw[0] if isinstance(k_raw, list) else k_raw)
                p_raw = graph_params.get("p", 0.3)
                p = float(p_raw[0] if isinstance(p_raw, list) else p_raw)
                G = nx.watts_strogatz_graph(n_nodes, k, p)
                filename = f"watts_strogatz_n{n_nodes}_k{k}_p{p:.2f}.edgelist"
            elif graph_type == "erdos_renyi":
                p_raw = graph_params.get("p", 0.01)
                p = float(p_raw[0] if isinstance(p_raw, list) else p_raw)
                G = nx.erdos_renyi_graph(n_nodes, p)
                filename = f"erdos_renyi_n{n_nodes}_p{p:.2f}.edgelist"
            else:
                continue
            
            filepath = datasets_dir / filename
            nx.write_edgelist(G, filepath, data=False)
            datasets[graph_type] = str(filepath)
            print(f"Generated {graph_type} graph: {filepath}", file=sys.stderr)
            
        except Exception as e:
            print(f"Warning: Failed to generate {graph_type} graph: {e}", file=sys.stderr)
    
    # Download real-world graphs
    real_world_urls = {
        "karate": "https://raw.githubusercontent.com/networkx/networkx/main/networkx/algorithms/community/tests/test_utils.py",
        "dolphins": "https://raw.githubusercontent.com/networkx/networkx/main/networkx/algorithms/community/tests/test_utils.py",
    }
    
    # For real-world graphs, we'll generate small examples
    if "karate" in real_world:
        try:
            G = nx.karate_club_graph()
            filepath = datasets_dir / "karate.edgelist"
            nx.write_edgelist(G, filepath, data=False)
            datasets["karate"] = str(filepath)
            print(f"Generated karate graph: {filepath}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to generate karate graph: {e}", file=sys.stderr)
    
    if "dolphins" in real_world:
        try:
            # Generate a small-world-like graph as dolphin substitute
            G = nx.watts_strogatz_graph(62, 5, 0.3)  # Approximate dolphin network size
            filepath = datasets_dir / "dolphins.edgelist"
            nx.write_edgelist(G, filepath, data=False)
            datasets["dolphins"] = str(filepath)
            print(f"Generated dolphins graph: {filepath}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to generate dolphins graph: {e}", file=sys.stderr)
    
    return datasets


def download_ml_datasets(dataset_spec: Dict[str, Any], datasets_dir: Path) -> Dict[str, str]:
    """
    Download ML datasets (minimal subsets).
    
    Returns:
        Dictionary mapping dataset names to paths
    """
    datasets = {}
    ml_dataset = dataset_spec.get("ml_dataset")
    
    if ml_dataset == "MNIST":
        # For now, create a placeholder - actual MNIST download would require torch/tensorflow
        ml_dir = datasets_dir / "ml"
        ml_dir.mkdir(parents=True, exist_ok=True)
        placeholder = ml_dir / "mnist_placeholder.txt"
        placeholder.write_text("MNIST dataset placeholder. Install torch/tensorflow to download full dataset.")
        datasets["mnist"] = str(placeholder)
        print(f"Created MNIST placeholder: {placeholder}", file=sys.stderr)
    
    return datasets


def download_pdb_files(dataset_spec: Dict[str, Any], datasets_dir: Path) -> Dict[str, str]:
    """
    Download PDB structure files from RCSB.
    
    Returns:
        Dictionary mapping PDB IDs to file paths
    """
    datasets = {}
    pdb_ids = dataset_spec.get("pdb_ids", [])
    
    if not pdb_ids:
        return datasets
    
    pdb_dir = datasets_dir / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    
    for pdb_id in pdb_ids:
        try:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filepath = pdb_dir / f"{pdb_id.upper()}.pdb"
                filepath.write_bytes(response.content)
                datasets[pdb_id.lower()] = str(filepath)
                print(f"Downloaded PDB file {pdb_id}: {filepath}", file=sys.stderr)
            else:
                print(f"Warning: Failed to download PDB {pdb_id}: HTTP {response.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to download PDB {pdb_id}: {e}", file=sys.stderr)
    
    return datasets


def generate_synthetic_datasets(dataset_spec: Dict[str, Any], datasets_dir: Path) -> Dict[str, str]:
    """
    Generate synthetic datasets based on specification (peptides, time series, classification data, etc.).
    
    Returns:
        Dictionary mapping dataset names to file paths
    """
    print(f"[DatasetAgent] DEBUG: generate_synthetic_datasets called", file=sys.stderr)
    print(f"[DatasetAgent] DEBUG: dataset_spec keys: {list(dataset_spec.keys())}", file=sys.stderr)
    
    datasets = {}
    synthetic_type = dataset_spec.get("synthetic_type")
    synthetic_params = dataset_spec.get("synthetic_params", {})
    
    print(f"[DatasetAgent] DEBUG: synthetic_type: {synthetic_type}", file=sys.stderr)
    print(f"[DatasetAgent] DEBUG: synthetic_params: {synthetic_params}", file=sys.stderr)
    
    if not synthetic_type:
        print(f"[DatasetAgent] WARNING: synthetic_type is None or empty, returning empty datasets", file=sys.stderr)
        return datasets
    
    synthetic_dir = datasets_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DatasetAgent] DEBUG: Created synthetic directory: {synthetic_dir}", file=sys.stderr)
    
    num_samples = synthetic_params.get("num_samples", 1000)
    file_format = synthetic_params.get("file_format", "CSV").upper()
    
    print(f"[DatasetAgent] DEBUG: num_samples: {num_samples}, file_format: {file_format}", file=sys.stderr)
    
    if not HAS_NUMPY or np is None:
        print(f"[DatasetAgent] ERROR: numpy is not available. Cannot generate synthetic datasets.", file=sys.stderr)
        return datasets
    
    try:
        if synthetic_type == "peptide_sequences":
            # Amino acid alphabet
            amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                          'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
            
            sequence_length = synthetic_params.get("sequence_length", 9)
            num_classes = synthetic_params.get("num_classes", 2)
            # Check for both dataset_types and classification_problems (Claude might use either)
            dataset_types = synthetic_params.get("dataset_types") or synthetic_params.get("classification_problems", ["dataset1"])
            
            if isinstance(dataset_types, str):
                dataset_types = [dataset_types]
            
            print(f"[DatasetAgent] DEBUG: Generating datasets for types: {dataset_types}", file=sys.stderr)
            
            for dataset_type in dataset_types:
                positive_samples = []
                negative_samples = []
                samples_per_class = num_samples // (2 * len(dataset_types))
                
                # Generate positive samples
                for _ in range(samples_per_class):
                    sequence = ''.join(np.random.choice(amino_acids, size=sequence_length))
                    positive_samples.append(sequence)
                
                # Generate negative samples
                for _ in range(samples_per_class):
                    sequence = ''.join(np.random.choice(amino_acids, size=sequence_length))
                    negative_samples.append(sequence)
                
                # Save dataset
                safe_name = str(dataset_type).replace(' ', '_').replace('-', '_')
                if file_format == "CSV":
                    filename = f"{safe_name}_peptides.csv"
                    filepath = synthetic_dir / filename
                    with open(filepath, 'w') as f:
                        f.write("sequence,label\n")
                        for seq in positive_samples:
                            f.write(f"{seq},1\n")
                        for seq in negative_samples:
                            f.write(f"{seq},0\n")
                else:
                    filename = f"{safe_name}_peptides.txt"
                    filepath = synthetic_dir / filename
                    with open(filepath, 'w') as f:
                        for seq in positive_samples:
                            f.write(f"{seq}\t1\n")
                        for seq in negative_samples:
                            f.write(f"{seq}\t0\n")
                
                datasets[safe_name] = str(filepath)
                file_size = filepath.stat().st_size if filepath.exists() else 0
                print(f"[DatasetAgent] Generated synthetic {dataset_type} dataset: {filepath} ({len(positive_samples) + len(negative_samples)} samples, {file_size} bytes)", file=sys.stderr)
        
        elif synthetic_type == "classification_data":
            num_features = synthetic_params.get("num_features", 20)
            num_classes = synthetic_params.get("num_classes", 2)
            
            # Generate feature matrix and labels
            X = np.random.randn(num_samples, num_features)
            y = np.random.randint(0, num_classes, size=num_samples)
            
            filename = "classification_data.csv"
            filepath = synthetic_dir / filename
            with open(filepath, 'w') as f:
                # Write header
                f.write(",".join([f"feature_{i}" for i in range(num_features)] + ["label"]) + "\n")
                # Write data
                for i in range(num_samples):
                    f.write(",".join([str(x) for x in X[i]] + [str(y[i])]) + "\n")
            
            datasets["classification_data"] = str(filepath)
            file_size = filepath.stat().st_size if filepath.exists() else 0
            print(f"[DatasetAgent] Generated synthetic classification dataset: {filepath} ({num_samples} samples, {num_features} features, {file_size} bytes)", file=sys.stderr)
        
        elif synthetic_type == "time_series":
            sequence_length = synthetic_params.get("sequence_length", 100)
            num_features = synthetic_params.get("num_features", 1)
            
            # Generate time series data
            time_points = np.arange(sequence_length)
            data = np.random.randn(num_samples, sequence_length, num_features)
            
            filename = "time_series_data.csv"
            filepath = synthetic_dir / filename
            with open(filepath, 'w') as f:
                f.write("sample_id,time,value\n")
                for sample_id in range(num_samples):
                    for t in range(sequence_length):
                        f.write(f"{sample_id},{t},{data[sample_id, t, 0]}\n")
            
            datasets["time_series"] = str(filepath)
            file_size = filepath.stat().st_size if filepath.exists() else 0
            print(f"[DatasetAgent] Generated synthetic time series dataset: {filepath} ({num_samples} samples, {sequence_length} time points, {file_size} bytes)", file=sys.stderr)
        
        else:
            # Generic synthetic data - just create a CSV with random data
            num_features = synthetic_params.get("num_features", 10)
            filename = f"{synthetic_type}_data.csv"
            filepath = synthetic_dir / filename
            
            X = np.random.randn(num_samples, num_features)
            with open(filepath, 'w') as f:
                f.write(",".join([f"feature_{i}" for i in range(num_features)]) + "\n")
                for i in range(num_samples):
                    f.write(",".join([str(x) for x in X[i]]) + "\n")
            
            datasets[synthetic_type] = str(filepath)
            file_size = filepath.stat().st_size if filepath.exists() else 0
            print(f"[DatasetAgent] Generated synthetic {synthetic_type} dataset: {filepath} ({num_samples} samples, {file_size} bytes)", file=sys.stderr)
    
    except Exception as e:
        print(f"Warning: Failed to generate synthetic {synthetic_type} dataset: {e}", file=sys.stderr)
    
    return datasets


async def generate_datasets(simulation_plan: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Main function to analyze and generate/download all required datasets.
    
    Args:
        simulation_plan: Dictionary from SimulationPlanAgent
        output_dir: Directory where datasets should be saved (same as simulation plan location)
    
    Returns:
        Dictionary with dataset_type and datasets (name -> path mapping)
    """
    print(f"[DatasetAgent] DEBUG: Starting dataset generation", file=sys.stderr)
    print(f"[DatasetAgent] DEBUG: Output directory: {output_dir}", file=sys.stderr)
    print(f"[DatasetAgent] DEBUG: Simulation plan keys: {list(simulation_plan.keys())}", file=sys.stderr)
    
    # Analyze requirements using Claude
    print(f"[DatasetAgent] DEBUG: Analyzing dataset requirements with Claude...", file=sys.stderr)
    dataset_spec = await analyze_dataset_requirements(simulation_plan)
    dataset_type = dataset_spec.get("dataset_type", "none")
    
    print(f"[DatasetAgent] DEBUG: Dataset type determined: {dataset_type}", file=sys.stderr)
    print(f"[DatasetAgent] DEBUG: Dataset spec: {json.dumps(dataset_spec, indent=2)}", file=sys.stderr)
    
    # Create datasets directory
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DatasetAgent] DEBUG: Created datasets directory: {datasets_dir}", file=sys.stderr)
    
    all_datasets = {}
    
    # Generate/download datasets based on type
    if dataset_type == "graph":
        print(f"[DatasetAgent] DEBUG: Generating graph datasets...", file=sys.stderr)
        graph_datasets = generate_graph_datasets(dataset_spec, datasets_dir)
        all_datasets.update(graph_datasets)
        print(f"[DatasetAgent] DEBUG: Generated {len(graph_datasets)} graph datasets", file=sys.stderr)
    
    elif dataset_type == "ml":
        print(f"[DatasetAgent] DEBUG: Downloading ML datasets...", file=sys.stderr)
        ml_datasets = download_ml_datasets(dataset_spec, datasets_dir)
        all_datasets.update(ml_datasets)
        print(f"[DatasetAgent] DEBUG: Downloaded {len(ml_datasets)} ML datasets", file=sys.stderr)
    
    elif dataset_type == "bio_structures":
        print(f"[DatasetAgent] DEBUG: Downloading PDB files...", file=sys.stderr)
        pdb_datasets = download_pdb_files(dataset_spec, datasets_dir)
        all_datasets.update(pdb_datasets)
        print(f"[DatasetAgent] DEBUG: Downloaded {len(pdb_datasets)} PDB files", file=sys.stderr)
    
    elif dataset_type == "synthetic":
        print(f"[DatasetAgent] DEBUG: Generating synthetic datasets...", file=sys.stderr)
        print(f"[DatasetAgent] DEBUG: synthetic_type from spec: {dataset_spec.get('synthetic_type')}", file=sys.stderr)
        print(f"[DatasetAgent] DEBUG: synthetic_params from spec: {dataset_spec.get('synthetic_params')}", file=sys.stderr)
        
        # If synthetic_type is missing but dataset_type is synthetic, infer from simulation plan
        if not dataset_spec.get("synthetic_type"):
            print(f"[DatasetAgent] DEBUG: synthetic_type missing, inferring from simulation plan...", file=sys.stderr)
            # Check simulation plan for clues
            procedure_steps = simulation_plan.get("procedure_steps", [])
            procedure_text = " ".join(procedure_steps).lower()
            
            if "peptide" in procedure_text or "sequence" in procedure_text:
                dataset_spec["synthetic_type"] = "peptide_sequences"
                print(f"[DatasetAgent] DEBUG: Inferred synthetic_type: peptide_sequences", file=sys.stderr)
            elif "time series" in procedure_text or "time_series" in procedure_text:
                dataset_spec["synthetic_type"] = "time_series"
                print(f"[DatasetAgent] DEBUG: Inferred synthetic_type: time_series", file=sys.stderr)
            else:
                dataset_spec["synthetic_type"] = "classification_data"
                print(f"[DatasetAgent] DEBUG: Inferred synthetic_type: classification_data (default)", file=sys.stderr)
            
            # Set default synthetic_params if missing
            if not dataset_spec.get("synthetic_params"):
                dataset_spec["synthetic_params"] = {
                    "num_samples": 1000,
                    "file_format": "CSV"
                }
                # Add sequence_length for peptides
                if dataset_spec["synthetic_type"] == "peptide_sequences":
                    # Try to get M from constants_required
                    constants = simulation_plan.get("constants_required", [])
                    for const in constants:
                        if const.get("name") == "M":
                            m_value = const.get("value_or_range", "9")
                            # Extract number from string like "9 for HIV-protease"
                            match = re.search(r'\d+', str(m_value))
                            if match:
                                dataset_spec["synthetic_params"]["sequence_length"] = int(match.group())
                            break
                    if "sequence_length" not in dataset_spec["synthetic_params"]:
                        dataset_spec["synthetic_params"]["sequence_length"] = 9
                    dataset_spec["synthetic_params"]["num_classes"] = 2
                    # Generate datasets for each classification problem mentioned
                    variables = simulation_plan.get("variables_to_vary", [])
                    for var in variables:
                        if var.get("name") == "classification_problem":
                            range_val = var.get("range", "")
                            # Extract list from string like "['HIV-protease', 'T-cell epitopes', 'HLA binding']"
                            try:
                                if isinstance(range_val, str):
                                    problems = ast.literal_eval(range_val)
                                else:
                                    problems = range_val
                                if isinstance(problems, list):
                                    dataset_spec["synthetic_params"]["dataset_types"] = problems
                                    print(f"[DatasetAgent] DEBUG: Found classification problems: {problems}", file=sys.stderr)
                            except:
                                pass
                            break
                
                print(f"[DatasetAgent] DEBUG: Set default synthetic_params: {dataset_spec['synthetic_params']}", file=sys.stderr)
        
        synthetic_datasets = generate_synthetic_datasets(dataset_spec, datasets_dir)
        all_datasets.update(synthetic_datasets)
        print(f"[DatasetAgent] DEBUG: Generated {len(synthetic_datasets)} synthetic datasets", file=sys.stderr)
    
    elif dataset_type == "none":
        print(f"[DatasetAgent] DEBUG: No datasets needed (dataset_type: none)", file=sys.stderr)
        pass
    
    # Verify files exist and log details
    print(f"[DatasetAgent] DEBUG: Verifying generated files...", file=sys.stderr)
    verified_datasets = {}
    for name, filepath_str in all_datasets.items():
        filepath = Path(filepath_str)
        if filepath.exists():
            file_size = filepath.stat().st_size
            print(f"[DatasetAgent] DEBUG: ✓ {name}: {filepath} ({file_size} bytes)", file=sys.stderr)
            verified_datasets[name] = str(filepath)
        else:
            print(f"[DatasetAgent] WARNING: ✗ {name}: File not found at {filepath}", file=sys.stderr)
    
    result = {
        "dataset_type": dataset_type,
        "datasets": verified_datasets,
        "reasoning": dataset_spec.get("reasoning", "")
    }
    
    print(f"[DatasetAgent] DEBUG: Final result: {json.dumps(result, indent=2)}", file=sys.stderr)
    print(f"[DatasetAgent] DEBUG: Total datasets: {len(verified_datasets)}", file=sys.stderr)
    
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
    """Main function to run the DatasetAgent"""
    parser = argparse.ArgumentParser(
        description="Generates and downloads datasets required for simulation code"
    )
    parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="Path to JSON file from SimulationPlanAgent (simulation plan)"
    )
    
    args = parser.parse_args()
    
    # Load simulation plan
    try:
        simulation_plan = load_json_from_file(args.plan)
    except Exception as e:
        print(f"Error loading simulation plan file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate simulation plan structure
    required_fields = ["simulation_equations", "constants_required", "variables_to_vary", "procedure_steps"]
    for field in required_fields:
        if field not in simulation_plan:
            print(f"Error: Simulation plan missing required field: {field}", file=sys.stderr)
            sys.exit(1)
    
    try:
        print("Analyzing simulation plan and generating datasets...", file=sys.stderr)
        
        # Determine output directory (same as simulation plan file directory)
        plan_path = Path(args.plan)
        output_dir = plan_path.parent
        
        # Generate datasets
        result = asyncio.run(generate_datasets(simulation_plan, output_dir))
        
        # Save dataset manifest
        manifest_path = output_dir / "datasets_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Dataset manifest saved to: {manifest_path}", file=sys.stderr)
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

