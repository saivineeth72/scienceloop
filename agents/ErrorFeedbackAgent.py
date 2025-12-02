#!/usr/bin/env python3
"""
ErrorFeedbackAgent - SpoonOS Agent

Analyzes error reports from SimulationRunnerAgent and generates structured
fix requests for CodeGeneratorAgent to regenerate corrected simulation code.
"""

import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os


def extract_error_context(stderr: str, stdout: str) -> str:
    """
    Extract the most relevant error context from stderr and stdout.
    
    Args:
        stderr: Standard error output
        stdout: Standard output
        
    Returns:
        Clean error context string
    """
    # Combine stderr and stdout, prioritize stderr
    full_output = stderr if stderr else stdout
    
    if not full_output:
        return "No error output available"
    
    # Try to extract traceback
    traceback_match = re.search(
        r'Traceback \(most recent call last\):.*?(?=\n\n|\Z)',
        full_output,
        re.DOTALL
    )
    if traceback_match:
        return traceback_match.group(0).strip()
    
    # Try to extract last error line
    lines = full_output.strip().split('\n')
    error_lines = []
    for line in reversed(lines):
        if line.strip():
            if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'traceback']):
                error_lines.insert(0, line)
                if len(error_lines) >= 5:  # Get last 5 relevant lines
                    break
    
    if error_lines:
        return '\n'.join(error_lines)
    
    return full_output[:500]  # Fallback: first 500 chars


def analyze_error_root_cause(stderr: str, stdout: str, exit_code: int) -> tuple[str, list[str]]:
    """
    Analyze error output to determine root cause and generate fix instructions.
    
    Args:
        stderr: Standard error output
        stdout: Standard output
        exit_code: Exit code from simulation
        
    Returns:
        Tuple of (explanation, fix_instructions)
    """
    full_output = (stderr + "\n" + stdout).lower()
    fix_instructions = []
    explanation_parts = []
    
    # Syntax errors
    if "syntaxerror" in full_output or "invalid syntax" in full_output:
        explanation_parts.append("Syntax error in the generated code")
        fix_instructions.append("Fix Python syntax errors in the code")
        # Try to extract the problematic line
        syntax_match = re.search(r'line (\d+).*?invalid syntax', full_output, re.IGNORECASE)
        if syntax_match:
            fix_instructions.append(f"Check line {syntax_match.group(1)} for syntax issues")
    
    # Import errors
    if "importerror" in full_output or "modulenotfounderror" in full_output:
        explanation_parts.append("Missing or incorrect import")
        module_match = re.search(r"no module named ['\"]([^'\"]+)['\"]", full_output, re.IGNORECASE)
        if module_match:
            module_name = module_match.group(1)
            fix_instructions.append(f"Add import for module: {module_name}")
            # Suggest common alternatives
            if module_name == "cv2":
                fix_instructions.append("Use: import cv2 (requires opencv-python package)")
            elif module_name in ["sklearn", "scikit-learn"]:
                fix_instructions.append("Use: from sklearn import ... (requires scikit-learn package)")
        else:
            fix_instructions.append("Fix import statements - ensure all required libraries are imported")
    
    # Name errors (undefined variables)
    if "nameerror" in full_output:
        explanation_parts.append("Undefined variable or function")
        name_match = re.search(r"name ['\"]([^'\"]+)['\"] is not defined", full_output, re.IGNORECASE)
        if name_match:
            var_name = name_match.group(1)
            # Check if this is a scope issue with test/train data
            if var_name in ['X_test', 'y_test', 'X_train', 'y_train']:
                explanation_parts.append(f"Variable scope issue: {var_name} is not accessible in current function scope")
                fix_instructions.append(f"CRITICAL: {var_name} is not in scope - this is a variable scope issue")
                fix_instructions.append(f"SOLUTION: Pass {var_name} as a parameter to functions that use it")
                fix_instructions.append(f"  - Example WRONG: def evaluate(): return score({var_name})  # {var_name} not in scope!")
                fix_instructions.append(f"  - Example CORRECT: def evaluate({var_name}): return score({var_name})")
                fix_instructions.append(f"  - OR define {var_name} at module/global scope before calling the function")
                fix_instructions.append(f"  - OR return {var_name} from train_test_split and pass it to functions that need it")
            else:
                fix_instructions.append(f"Define variable or function: {var_name}")
        else:
            fix_instructions.append("Check for undefined variables and ensure all variables are defined before use")
            fix_instructions.append("If error mentions X_test, y_test, X_train, y_train: These must be passed as function parameters or defined at global scope")
    
    # Attribute errors
    if "attributeerror" in full_output:
        explanation_parts.append("Object does not have the expected attribute or method")
        attr_match = re.search(r"['\"]([^'\"]+)['\"] object has no attribute ['\"]([^'\"]+)['\"]", full_output, re.IGNORECASE)
        if attr_match:
            obj_type = attr_match.group(1)
            attr_name = attr_match.group(2)
            fix_instructions.append(f"Fix attribute access: {obj_type} does not have '{attr_name}'")
            # Common fixes
            if attr_name == "predict_proba" and "svc" in full_output:
                fix_instructions.append("Initialize SVC with probability=True: SVC(probability=True)")
        else:
            fix_instructions.append("Check object attributes and methods - ensure correct API usage")
    
    # File not found errors
    if "filenotfounderror" in full_output or "no such file" in full_output:
        explanation_parts.append("File or dataset path not found")
        file_match = re.search(r"['\"]([^'\"]+\.(csv|json|txt|png|jpg|npy))['\"]", full_output, re.IGNORECASE)
        if file_match:
            file_path = file_match.group(1)
            # Check if path contains "papers/" - means it's using project-relative path instead of script-relative
            if "papers/" in file_path:
                # Extract relative path from papers/{paper_name}/datasets/... to datasets/...
                path_parts = file_path.split("/")
                if "papers" in path_parts and len(path_parts) > 2:
                    papers_idx = path_parts.index("papers")
                    if papers_idx + 2 < len(path_parts):
                        # Get everything after papers/{paper_name}/
                        relative_path = "/".join(path_parts[papers_idx + 2:])
                        fix_instructions.append(f"CRITICAL: Change dataset path from '{file_path}' to '{relative_path}' (relative to script directory)")
                        fix_instructions.append("The simulation runs from the script's directory (papers/{paper_name}/), so use paths relative to that directory")
                        fix_instructions.append(f"Example: Replace pd.read_csv('{file_path}') with pd.read_csv('{relative_path}')")
                    else:
                        fix_instructions.append(f"Fix dataset file path: {file_path}")
                else:
                    fix_instructions.append(f"Fix dataset file path: {file_path}")
            else:
                fix_instructions.append(f"Check if dataset file exists at path: {file_path}")
            fix_instructions.append("CRITICAL: All file paths in generated code must be relative to the script's directory, NOT relative to project root")
            fix_instructions.append("If datasets_manifest shows 'papers/X/datasets/file.csv', use 'datasets/file.csv' in code")
        else:
            fix_instructions.append("Check all file paths - ensure datasets exist at specified locations")
            fix_instructions.append("CRITICAL: Use paths relative to script directory, not project root")
    
    # Index errors
    if "indexerror" in full_output:
        explanation_parts.append("List or array index out of range")
        fix_instructions.append("Check array/list indexing - ensure indices are within bounds")
        fix_instructions.append("Verify data dimensions match expected sizes")
    
    # Key errors
    if "keyerror" in full_output:
        explanation_parts.append("Dictionary key not found")
        key_match = re.search(r"['\"]([^'\"]+)['\"]", full_output)
        if key_match:
            key_name = key_match.group(1)
            fix_instructions.append(f"Add missing dictionary key or check key name: {key_name}")
        else:
            fix_instructions.append("Check dictionary keys - ensure all required keys exist")
    
    # Zero division errors
    if "zerodivisionerror" in full_output or "divide by zero" in full_output:
        explanation_parts.append("Division by zero")
        fix_instructions.append("Add zero-division checks before division operations")
        fix_instructions.append("Ensure denominators are non-zero before division")
    
    # Type errors
    if "typeerror" in full_output:
        explanation_parts.append("Type mismatch or incorrect type usage")
        type_match = re.search(r"unsupported operand type\(s\)|can't|must be", full_output, re.IGNORECASE)
        if type_match:
            fix_instructions.append("Fix type mismatches - ensure correct data types for operations")
        else:
            fix_instructions.append("Check variable types and ensure compatibility with operations")
    
    # Dimension/shape mismatch errors (especially for plotting)
    if "valueerror" in full_output and ("shape" in full_output or "dimension" in full_output or "must have same" in full_output):
        explanation_parts.append("Array dimension or shape mismatch")
        # Extract shape information
        shape_match = re.search(r"shapes \(([^)]+)\) and \(([^)]+)\)", full_output, re.IGNORECASE)
        if shape_match:
            shape1 = shape_match.group(1)
            shape2 = shape_match.group(2)
            fix_instructions.append(f"CRITICAL: Dimension mismatch detected - arrays have shapes ({shape1}) and ({shape2})")
            
            # Check if this is a plotting error
            if "plt.plot" in stderr.lower() or "plot" in stderr.lower():
                fix_instructions.append("CRITICAL: This is a plotting dimension mismatch error")
                fix_instructions.append("Root cause: You have nested loops collecting multiple values per x-axis point")
                
                # Calculate the multiplier
                try:
                    shape1_num = int(shape1.split(',')[0].strip())
                    shape2_num = int(shape2.split(',')[0].strip())
                    multiplier = shape2_num // shape1_num if shape1_num > 0 else 0
                    if multiplier > 1:
                        fix_instructions.append(f"Problem: x-axis has {shape1_num} values but y-axis has {shape2_num} values (ratio: {multiplier}x)")
                        fix_instructions.append(f"This means you're collecting {multiplier} values per x-axis point (likely from nested loops)")
                except:
                    pass
                
                fix_instructions.append("SOLUTION - Choose ONE:")
                fix_instructions.append("  Option 1: Collect only ONE accuracy per x-axis value")
                fix_instructions.append("    - Move accuracy collection OUTSIDE inner loops")
                fix_instructions.append("    - Example: for n_nonlinear in range(11): ... accuracies.append(acc)  # Only append once per n_nonlinear")
                fix_instructions.append("  Option 2: Aggregate results per x-axis value")
                fix_instructions.append("    - Group results by x-axis value and compute mean/median")
                fix_instructions.append("    - Example: Use pandas groupby or numpy to average accuracies per n_nonlinear_features")
                fix_instructions.append("  Option 3: Fix x-axis to match all collected values")
                fix_instructions.append("    - Create x-axis array with correct length matching all results")
                fix_instructions.append("    - Example: x_axis = [result[0] for result in results]  # Extract x-value from each result")
                fix_instructions.append("MOST COMMON FIX: Move accuracy.append() outside inner loops - only append once per outer loop iteration")
        else:
            fix_instructions.append("Check array shapes and dimensions - ensure x and y arrays have matching lengths for plotting")
            fix_instructions.append("If plotting, ensure x-axis array length matches y-axis array length")
    
    # NetworkX/graph errors
    if "networkx" in full_output or "graph" in full_output:
        if "node" in full_output and ("not in" in full_output or "not found" in full_output):
            explanation_parts.append("Graph node not found")
            fix_instructions.append("Check graph node names - ensure nodes exist before accessing")
        if "edge" in full_output:
            explanation_parts.append("Graph edge issue")
            fix_instructions.append("Verify graph edges and connectivity")
    
    # NumPy/SciPy errors (only if not already handled by ValueError above)
    if ("numpy" in full_output or "scipy" in full_output) and "valueerror" not in full_output:
        if "nan" in full_output or "inf" in full_output:
            explanation_parts.append("Numerical instability (NaN or Inf values)")
            fix_instructions.append("Add checks for NaN/Inf values and handle edge cases")
    
    # Timeout errors
    if "timeout" in full_output or exit_code == -1:
        if "timeout" in full_output:
            explanation_parts.append("Simulation timed out")
            fix_instructions.append("Optimize computation - reduce iterations or data size")
            fix_instructions.append("Check for infinite loops or very slow operations")
    
    # Generic error handling
    if not explanation_parts:
        explanation_parts.append("Runtime error occurred during simulation execution")
        fix_instructions.append("Review the error traceback and fix the root cause")
        fix_instructions.append("Ensure all variables are initialized and all functions are defined")
    
    explanation = ". ".join(explanation_parts) if explanation_parts else "Unknown error occurred"
    
    return explanation, fix_instructions


def generate_fix_request(
    error_report: Dict[str, Any],
    simulation_plan: Dict[str, Any],
    original_python_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a structured fix request for CodeGeneratorAgent.
    
    Args:
        error_report: Error report from SimulationRunnerAgent
        simulation_plan: Original simulation plan JSON
        original_python_code: Optional original Python code (for context)
        
    Returns:
        Structured fix request JSON
    """
    status = error_report.get("status", "error")
    stderr = error_report.get("stderr", "")
    stdout = error_report.get("stdout", "")
    exit_code = error_report.get("exit_code", -1)
    error_summary = error_report.get("error_summary", "")
    
    # Extract error context
    error_context = extract_error_context(stderr, stdout)
    
    # Analyze root cause
    explanation, fix_instructions = analyze_error_root_cause(stderr, stdout, exit_code)
    
    # Build fix request
    fix_request = {
        "needs_regeneration": True,
        "error_context": error_context,
        "error_summary": error_summary,
        "explanation": explanation,
        "fix_instructions": fix_instructions,
        "simulation_plan": simulation_plan
    }
    
    # Optionally include original code for reference
    if original_python_code:
        fix_request["original_code_reference"] = original_python_code[:1000]  # First 1000 chars
    
    return fix_request


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
        description="ErrorFeedbackAgent - Analyze simulation errors and generate fix requests"
    )
    parser.add_argument(
        "--error-report",
        type=str,
        required=True,
        help="Path to JSON file containing error report from SimulationRunnerAgent"
    )
    parser.add_argument(
        "--simulation-plan",
        type=str,
        required=True,
        help="Path to JSON file containing the original simulation plan"
    )
    parser.add_argument(
        "--original-code",
        type=str,
        help="Path to the original Python simulation code file (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the fix request JSON (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load error report
        error_report = load_json_from_file(args.error_report)
        
        # Load simulation plan
        simulation_plan = load_json_from_file(args.simulation_plan)
        
        # Load original code if provided
        original_python_code = None
        if args.original_code:
            code_path = Path(args.original_code)
            if code_path.exists():
                original_python_code = code_path.read_text()
            else:
                print(f"Warning: Original code file not found: {args.original_code}", file=sys.stderr)
        
        # Generate fix request
        fix_request = generate_fix_request(
            error_report=error_report,
            simulation_plan=simulation_plan,
            original_python_code=original_python_code
        )
        
        # Output or save fix request
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(fix_request, f, indent=2)
            print(f"Fix request saved to: {output_path}", file=sys.stderr)
        else:
            # Print to stdout
            print(json.dumps(fix_request, indent=2))
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

