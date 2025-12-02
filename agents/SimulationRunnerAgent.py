#!/usr/bin/env python3
"""
SimulationRunnerAgent - SpoonOS Agent

Executes generated Python simulation files, captures outputs, and stores
all artifacts in structured directories.

Uses pure Python (subprocess, os, shutil) - NO MCP tools.
"""

import sys
import json
import argparse
import asyncio
import uuid
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os


def validate_inputs(
    python_file_path: str,
    dataset_paths: Optional[Union[Dict[str, str], List[str]]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate that all required files exist.
    
    Args:
        python_file_path: Path to the Python simulation file
        dataset_paths: Dictionary or list of dataset file paths
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate Python simulation file
    python_path = Path(python_file_path)
    if not python_path.exists():
        return False, f"Simulation Python file not found: {python_file_path}"
    
    if not python_path.is_file():
        return False, f"Path is not a file: {python_file_path}"
    
    # Validate datasets if provided
    if dataset_paths:
        if isinstance(dataset_paths, dict):
            paths_to_check = list(dataset_paths.values())
        elif isinstance(dataset_paths, list):
            paths_to_check = dataset_paths
        else:
            return False, f"dataset_paths must be dict or list, got {type(dataset_paths)}"
        
        for dataset_path in paths_to_check:
            dataset_file = Path(dataset_path)
            if not dataset_file.exists():
                return False, f"Required dataset file not found: {dataset_path}"
    
    return True, None


def create_run_directory(base_dir: str = "results") -> Path:
    """
    Create a unique run directory structure.
    
    Args:
        base_dir: Base directory for results (default: "results")
        
    Returns:
        Path to the created run directory
    """
    run_id = str(uuid.uuid4())
    results_base = Path(base_dir)
    run_dir = results_base / run_id
    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    
    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def install_missing_package(package_name: str) -> bool:
    """
    Attempt to install a missing Python package.
    
    Args:
        package_name: Name of the package to install
        
    Returns:
        True if installation succeeded, False otherwise
    """
    try:
        print(f"Attempting to install missing package: {package_name}", file=sys.stderr)
        result = subprocess.run(
            ["pip3", "install", package_name],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            print(f"Successfully installed {package_name}", file=sys.stderr)
            return True
        else:
            print(f"Failed to install {package_name}: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"Error installing {package_name}: {e}", file=sys.stderr)
        return False


def extract_missing_module(error_text: str) -> Optional[str]:
    """
    Extract the missing module name from a ModuleNotFoundError.
    
    Args:
        error_text: Error message text
        
    Returns:
        Module name if found, None otherwise
    """
    import re
    # Match patterns like "No module named 'sklearn'" or "No module named \"sklearn\""
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_text)
    if match:
        module_name = match.group(1)
        # Map common module names to pip package names
        module_mapping = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'yaml': 'PyYAML',
        }
        return module_mapping.get(module_name, module_name)
    return None


def run_simulation(
    python_file_path: str,
    work_dir: Optional[str] = None,
    timeout: int = 600,
    auto_install: bool = True
) -> Dict[str, Any]:
    """
    Execute the Python simulation using subprocess.
    
    Args:
        python_file_path: Path to the Python simulation file
        work_dir: Working directory for the simulation (default: parent of python_file_path)
        timeout: Timeout in seconds (default: 600 = 10 minutes)
        auto_install: Whether to automatically install missing packages (default: True)
        
    Returns:
        Dictionary with stdout, stderr, and exit_code
    """
    print(f"[SIMULATION-RUNNER] run_simulation() ENTERED", file=sys.stderr)
    sys.stderr.flush()
    
    abs_python_path = Path(python_file_path).resolve()
    
    if work_dir is None:
        work_dir = str(abs_python_path.parent)
    else:
        work_dir = str(Path(work_dir).resolve())
    
    max_retries = 3 if auto_install else 1
    retry_count = 0
    last_result = None
    
    print(f"[SIMULATION-RUNNER] Starting subprocess execution (timeout={timeout}s, max_retries={max_retries})", file=sys.stderr)
    sys.stderr.flush()
    while retry_count < max_retries:
        try:
            print(f"[SIMULATION-RUNNER] Attempt {retry_count + 1}/{max_retries}", file=sys.stderr)
            print(f"[SIMULATION-RUNNER] Running: python3 {abs_python_path}", file=sys.stderr)
            print(f"[SIMULATION-RUNNER] Working directory: {work_dir}", file=sys.stderr)
            sys.stderr.flush()
            
            # Run the simulation
            print(f"[SIMULATION-RUNNER] About to call subprocess.run()...", file=sys.stderr)
            print(f"[SIMULATION-RUNNER] Command: python3 {abs_python_path}", file=sys.stderr)
            print(f"[SIMULATION-RUNNER] Timeout: {timeout} seconds", file=sys.stderr)
            sys.stderr.flush()
            
            # Use Popen to get real-time output and detect if it's actually running
            import time
            start_time = time.time()
            print(f"[SIMULATION-RUNNER] Starting subprocess at {start_time}", file=sys.stderr)
            sys.stderr.flush()
            
            process = subprocess.Popen(
                ["python3", str(abs_python_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=work_dir,
                bufsize=1  # Line buffered
            )
            
            print(f"[SIMULATION-RUNNER] Subprocess started, PID: {process.pid}", file=sys.stderr)
            sys.stderr.flush()
            
            # Stream output in real-time and wait for completion with timeout
            try:
                import threading
                import queue
                
                stdout_lines = []
                stderr_lines = []
                stdout_queue = queue.Queue()
                stderr_queue = queue.Queue()
                
                def read_output(pipe, queue, label):
                    """Read lines from pipe and put them in queue"""
                    try:
                        for line in iter(pipe.readline, ''):
                            if line:
                                queue.put(line)
                                # Also log to stderr for debugging
                                print(f"[SIMULATION-OUTPUT-{label}] {line.rstrip()}", file=sys.stderr)
                                sys.stderr.flush()
                    except Exception as e:
                        print(f"[SIMULATION-RUNNER] Error reading {label}: {e}", file=sys.stderr)
                        sys.stderr.flush()
                    finally:
                        pipe.close()
                
                # Start threads to read stdout and stderr
                stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_queue, "STDOUT"), daemon=True)
                stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_queue, "STDERR"), daemon=True)
                stdout_thread.start()
                stderr_thread.start()
                
                # Wait for process to complete, checking periodically
                last_log_time = start_time
                while process.poll() is None:
                    elapsed = time.time() - start_time
                    
                    # Check for timeout
                    if elapsed > timeout:
                        raise subprocess.TimeoutExpired(process.args, timeout)
                    
                    # Log progress every 30 seconds
                    if elapsed - last_log_time >= 30:
                        print(f"[SIMULATION-RUNNER] Subprocess still running after {elapsed:.1f} seconds (PID: {process.pid})", file=sys.stderr)
                        sys.stderr.flush()
                        last_log_time = elapsed
                    
                    # Process any available output
                    try:
                        while True:
                            line = stdout_queue.get_nowait()
                            stdout_lines.append(line)
                    except queue.Empty:
                        pass
                    
                    try:
                        while True:
                            line = stderr_queue.get_nowait()
                            stderr_lines.append(line)
                    except queue.Empty:
                        pass
                    
                    time.sleep(0.1)  # Small sleep to avoid busy-waiting
                
                # Process is done, collect remaining output
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
                
                # Get any remaining output
                while True:
                    try:
                        stdout_lines.append(stdout_queue.get_nowait())
                    except queue.Empty:
                        break
                
                while True:
                    try:
                        stderr_lines.append(stderr_queue.get_nowait())
                    except queue.Empty:
                        break
                
                stdout = ''.join(stdout_lines)
                stderr = ''.join(stderr_lines)
                
                elapsed = time.time() - start_time
                print(f"[SIMULATION-RUNNER] Subprocess completed in {elapsed:.2f} seconds", file=sys.stderr)
                print(f"[SIMULATION-RUNNER] Return code: {process.returncode}", file=sys.stderr)
                sys.stderr.flush()
                
                result = subprocess.CompletedProcess(
                    process.args,
                    process.returncode,
                    stdout,
                    stderr
                )
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_time
                print(f"[SIMULATION-RUNNER] Subprocess timed out after {elapsed:.2f} seconds", file=sys.stderr)
                print(f"[SIMULATION-RUNNER] Killing process PID: {process.pid}", file=sys.stderr)
                sys.stderr.flush()
                process.kill()
                # Try to get any remaining output
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except:
                    stdout, stderr = "", ""
                raise subprocess.TimeoutExpired(process.args, timeout)
            
            print(f"[SIMULATION-RUNNER] subprocess.run() returned", file=sys.stderr)
            sys.stderr.flush()
            
            print(f"[SIMULATION-RUNNER] Subprocess completed", file=sys.stderr)
            print(f"[SIMULATION-RUNNER] Return code: {result.returncode}", file=sys.stderr)
            print(f"[SIMULATION-RUNNER] Stdout length: {len(result.stdout)}", file=sys.stderr)
            print(f"[SIMULATION-RUNNER] Stderr length: {len(result.stderr)}", file=sys.stderr)
            
            last_result = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
            
            # If successful, return immediately
            if result.returncode == 0:
                print(f"[SIMULATION-RUNNER] Simulation succeeded, returning result", file=sys.stderr)
                return last_result
            
            # Check if it's a ModuleNotFoundError and auto-install is enabled
            if auto_install and retry_count < max_retries - 1:
                missing_module = extract_missing_module(result.stderr)
                if missing_module:
                    print(f"Detected missing module: {missing_module}", file=sys.stderr)
                    if install_missing_package(missing_module):
                        retry_count += 1
                        continue  # Retry running the simulation
            
            # If we can't auto-install or exhausted retries, return the result
            return last_result
            
        except subprocess.TimeoutExpired:
            print(f"[SIMULATION-RUNNER] ERROR: Simulation timed out after {timeout} seconds", file=sys.stderr)
            return {
                "stdout": "",
                "stderr": f"Simulation timed out after {timeout} seconds",
                "exit_code": -1
            }
        except Exception as e:
            print(f"[SIMULATION-RUNNER] ERROR: Execution error: {e}", file=sys.stderr)
            import traceback
            print(f"[SIMULATION-RUNNER] Traceback: {traceback.format_exc()}", file=sys.stderr)
            return {
                "stdout": "",
                "stderr": f"Execution error: {e}",
                "exit_code": -1
            }
    
    # If we exhausted retries, return the last result
    return last_result if last_result else {
        "stdout": "",
        "stderr": "Failed after retries",
        "exit_code": -1
    }


def collect_artifacts(
    simulation_work_dir: str,
    artifacts_dir: Path,
    exclude_patterns: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Collect all generated output files from the simulation directory.
    
    Args:
        simulation_work_dir: Directory where simulation was run
        artifacts_dir: Directory to copy artifacts to
        exclude_patterns: List of filename patterns to exclude
        
    Returns:
        List of dictionaries with filename and path for each artifact
    """
    if exclude_patterns is None:
        exclude_patterns = [
            'understand.json',
            'knowledge_graph.json',
            'hypothesis.json',
            'simulation_plan.json',
            'datasets_manifest.json',
            'simulation.py',
            '.pdf'  # Exclude original PDFs
        ]
    
    output_extensions = ['.png', '.jpg', '.jpeg', '.csv', '.txt', '.json', '.svg']
    artifacts = []
    
    work_path = Path(simulation_work_dir)
    
    if not work_path.exists():
        return artifacts
    
    # Scan directory recursively
    for file_path in work_path.rglob('*'):
        if not file_path.is_file():
            continue
        
        filename = file_path.name
        
        # Skip if matches exclude patterns
        if any(pattern in filename for pattern in exclude_patterns):
            continue
        
        # Check if it's an output file type
        if any(filename.lower().endswith(ext) for ext in output_extensions):
            try:
                # Copy to artifacts directory
                artifact_path = artifacts_dir / filename
                
                # Handle potential filename conflicts
                counter = 1
                while artifact_path.exists():
                    stem = artifact_path.stem
                    suffix = artifact_path.suffix
                    artifact_path = artifacts_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                # Copy file
                shutil.copy2(file_path, artifact_path)
                
                artifacts.append({
                    "filename": artifact_path.name,
                    "path": str(artifact_path)
                })
            except Exception as e:
                print(f"Warning: Failed to copy artifact {filename}: {e}", file=sys.stderr)
    
    return artifacts


def generate_error_summary(stderr: str, exit_code: int) -> str:
    """
    Generate a short, readable error summary from stderr and exit code.
    
    Args:
        stderr: Standard error output
        exit_code: Exit code from simulation
        
    Returns:
        Short error summary string
    """
    if exit_code == -1:
        if "timeout" in stderr.lower():
            return "Simulation timed out"
        return "Execution failed with unknown error"
    
    if exit_code != 0:
        # Try to extract key error message
        lines = stderr.strip().split('\n')
        for line in reversed(lines):
            if line.strip() and not line.startswith(' '):
                if len(line) < 200:
                    return line[:200]
                else:
                    return line[:197] + "..."
    
    return "Unknown error occurred"


async def run_simulation_agent(
    python_file_path: str,
    dataset_paths: Optional[Union[Dict[str, str], List[str]]] = None,
    simulation_workdir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main agent function to run simulation and collect results.
    
    Args:
        python_file_path: Path to the Python simulation file
        dataset_paths: Dictionary or list of dataset file paths
        simulation_workdir: Working directory for simulation (default: parent of python_file_path)
        
    Returns:
        Dictionary with status, run_id, stdout, stderr, exit_code, artifacts, results_path
    """
    print("="*80, file=sys.stderr)
    print("[SIMULATION-RUNNER] AGENT CALLED", file=sys.stderr)
    print("="*80, file=sys.stderr)
    print(f"[SIMULATION-RUNNER] python_file_path: {python_file_path}", file=sys.stderr)
    print(f"[SIMULATION-RUNNER] dataset_paths: {dataset_paths}", file=sys.stderr)
    print(f"[SIMULATION-RUNNER] simulation_workdir: {simulation_workdir}", file=sys.stderr)
    
    # Validate inputs
    print(f"[SIMULATION-RUNNER] Validating inputs...", file=sys.stderr)
    is_valid, error_msg = validate_inputs(python_file_path, dataset_paths)
    print(f"[SIMULATION-RUNNER] Validation result: is_valid={is_valid}, error_msg={error_msg}", file=sys.stderr)
    if not is_valid:
        # Return error payload
        run_dir = create_run_directory()
        run_id = run_dir.name
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save error to log
        (logs_dir / "error.txt").write_text(error_msg)
        
        return {
            "status": "error",
            "run_id": run_id,
            "stdout": "",
            "stderr": error_msg,
            "exit_code": -1,
            "error_summary": error_msg,
            "results_path": str(run_dir)
        }
    
    # Create run directory
    run_dir = create_run_directory()
    run_id = run_dir.name
    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    
    # Determine simulation working directory
    abs_python_path = Path(python_file_path).resolve()
    if simulation_workdir is None:
        sim_work_dir = str(abs_python_path.parent)
    else:
        sim_work_dir = str(Path(simulation_workdir).resolve())
    
    # Run simulation
    print(f"[SIMULATION-RUNNER] Executing simulation: {python_file_path}", file=sys.stderr)
    print(f"[SIMULATION-RUNNER] Working directory: {sim_work_dir}", file=sys.stderr)
    if dataset_paths:
        count = len(dataset_paths) if isinstance(dataset_paths, dict) else len(dataset_paths)
        print(f"[SIMULATION-RUNNER] Using {count} dataset(s)", file=sys.stderr)
    
    print(f"[SIMULATION-RUNNER] Calling run_simulation()...", file=sys.stderr)
    print(f"[SIMULATION-RUNNER] Function exists: {callable(run_simulation)}", file=sys.stderr)
    sys.stderr.flush()  # Ensure log is flushed before potentially long operation
    
    # Add try-except to catch any issues with the function call itself
    try:
        print(f"[SIMULATION-RUNNER] About to invoke run_simulation()...", file=sys.stderr)
        sys.stderr.flush()
        result = run_simulation(python_file_path, work_dir=sim_work_dir, auto_install=True)
        print(f"[SIMULATION-RUNNER] run_simulation() completed and returned", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(f"[SIMULATION-RUNNER] ERROR calling run_simulation(): {e}", file=sys.stderr)
        import traceback
        print(f"[SIMULATION-RUNNER] Traceback: {traceback.format_exc()}", file=sys.stderr)
        sys.stderr.flush()
        raise
    print(f"[SIMULATION-RUNNER] Result keys: {list(result.keys())}", file=sys.stderr)
    
    stdout = result["stdout"]
    stderr = result["stderr"]
    exit_code = result["exit_code"]
    
    print(f"[SIMULATION-RUNNER] Exit code: {exit_code}", file=sys.stderr)
    print(f"[SIMULATION-RUNNER] Stdout length: {len(stdout)}", file=sys.stderr)
    print(f"[SIMULATION-RUNNER] Stderr length: {len(stderr)}", file=sys.stderr)
    if stderr:
        print(f"[SIMULATION-RUNNER] Stderr preview (first 500 chars): {stderr[:500]}", file=sys.stderr)
    
    # Ensure logs directory exists
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save logs
    (logs_dir / "stdout.txt").write_text(stdout)
    (logs_dir / "stderr.txt").write_text(stderr)
    
    # Collect artifacts
    artifacts = collect_artifacts(sim_work_dir, artifacts_dir)
    
    if artifacts:
        print(f"Captured {len(artifacts)} artifact(s)", file=sys.stderr)
    
    # Return structured result based on success/failure
    print(f"[SIMULATION-RUNNER] Preparing return result...", file=sys.stderr)
    if exit_code == 0:
        print(f"[SIMULATION-RUNNER] Simulation succeeded (exit_code=0)", file=sys.stderr)
        success_result = {
            "status": "success",
            "run_id": run_id,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "artifacts": artifacts,
            "results_path": str(run_dir)
        }
        
        print(f"[SIMULATION-RUNNER] Success result keys: {list(success_result.keys())}", file=sys.stderr)
        print(f"[SIMULATION-RUNNER] Artifacts count: {len(artifacts)}", file=sys.stderr)
        
        # Save result report for ReportAgent (always save, success or error)
        result_report_path = run_dir / "simulation_result.json"
        with open(result_report_path, 'w') as f:
            json.dump(success_result, f, indent=2)
        
        # Also save as error_report.json for backward compatibility (ReportAgent can use either)
        error_report_path = run_dir / "error_report.json"
        with open(error_report_path, 'w') as f:
            json.dump(success_result, f, indent=2)
        
        print(f"[SIMULATION-RUNNER] Result report saved to: {result_report_path}", file=sys.stderr)
        print(f"[SIMULATION-RUNNER] Returning success result", file=sys.stderr)
        print("="*80, file=sys.stderr)
        
        return success_result
    else:
        error_summary = generate_error_summary(stderr, exit_code)
        error_result = {
            "status": "error",
            "run_id": run_id,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "error_summary": error_summary,
            "artifacts": artifacts,
            "results_path": str(run_dir)
        }
        
        # Save error report for ErrorFeedbackAgent and ReportAgent
        error_report_path = run_dir / "error_report.json"
        with open(error_report_path, 'w') as f:
            json.dump(error_result, f, indent=2)
        
        # Also save as simulation_result.json for consistency
        result_report_path = run_dir / "simulation_result.json"
        with open(result_report_path, 'w') as f:
            json.dump(error_result, f, indent=2)
        
        print(f"[SIMULATION-RUNNER] Error result keys: {list(error_result.keys())}", file=sys.stderr)
        print(f"[SIMULATION-RUNNER] Error summary: {error_summary}", file=sys.stderr)
        print(f"[SIMULATION-RUNNER] Error report saved to: {error_report_path}", file=sys.stderr)
        print(f"[SIMULATION-RUNNER] Returning error result", file=sys.stderr)
        print("="*80, file=sys.stderr)
        
        return error_result


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
        description="SimulationRunnerAgent - Execute Python simulations and collect results"
    )
    parser.add_argument(
        "--python-file",
        type=str,
        required=True,
        help="Path to the Python simulation file"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Path to JSON file containing dataset paths (dict or list)"
    )
    parser.add_argument(
        "--workdir",
        type=str,
        help="Working directory for simulation (default: parent of python_file)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically trigger ErrorFeedbackAgent on errors (requires --simulation-plan)"
    )
    parser.add_argument(
        "--simulation-plan",
        type=str,
        help="Path to simulation plan JSON (required for --auto-fix)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of auto-fix retry attempts (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Load dataset paths if provided
    dataset_paths = None
    if args.datasets:
        try:
            datasets_data = load_json_from_file(args.datasets)
            if isinstance(datasets_data, dict):
                # If it's a dict with "datasets" key, extract it
                if "datasets" in datasets_data:
                    dataset_paths = datasets_data["datasets"]
                else:
                    dataset_paths = datasets_data
            elif isinstance(datasets_data, list):
                dataset_paths = datasets_data
            else:
                print(f"Warning: datasets file format not recognized, ignoring", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to load datasets: {e}", file=sys.stderr)
    
    try:
        max_retries = args.max_retries if args.auto_fix else 0
        retry_count = 0
        result = None
        
        # Retry loop: keep fixing and re-running until success or max retries
        while retry_count <= max_retries:
            # Run the simulation agent
            result = asyncio.run(run_simulation_agent(
                python_file_path=args.python_file,
                dataset_paths=dataset_paths,
                simulation_workdir=args.workdir
            ))
            
            # If successful, break out of retry loop
            if result.get("status") == "success":
                print(f"\n✓ Simulation completed successfully!", file=sys.stderr)
                break
            
            # If error occurred and auto-fix is enabled, try to fix
            if result.get("status") == "error" and args.auto_fix and retry_count < max_retries:
                if not args.simulation_plan:
                    print("Error: --simulation-plan required when using --auto-fix", file=sys.stderr)
                    sys.exit(1)
                
                print("\n" + "="*60, file=sys.stderr)
                print(f"Auto-triggering ErrorFeedbackAgent (attempt {retry_count + 1}/{max_retries})...", file=sys.stderr)
                print("="*60 + "\n", file=sys.stderr)
                
                # Import and call ErrorFeedbackAgent
                try:
                    # Add parent directory to path for imports (sys is already imported at top)
                    parent_dir = str(Path(__file__).parent.parent)
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    from agents.ErrorFeedbackAgent import generate_fix_request
                
                    error_report_path = Path(result["results_path"]) / "error_report.json"
                    if not error_report_path.exists():
                        # Create error report if it doesn't exist
                        with open(error_report_path, 'w') as f:
                            json.dump(result, f, indent=2)
                    
                    # Load simulation plan using local function
                    simulation_plan = load_json_from_file(args.simulation_plan)
                    
                    # Try to load original code
                    original_code = None
                    code_path = Path(args.python_file)
                    if code_path.exists():
                        original_code = code_path.read_text()
                    
                    # Generate fix request
                    fix_request = generate_fix_request(
                        error_report=result,
                        simulation_plan=simulation_plan,
                        original_python_code=original_code
                    )
                    
                    # Save fix request
                    fix_request_path = Path(result["results_path"]) / "fix_request.json"
                    with open(fix_request_path, 'w') as f:
                        json.dump(fix_request, f, indent=2)
                    
                    print(f"Fix request saved to: {fix_request_path}", file=sys.stderr)
                    
                    # Automatically trigger CodeGeneratorAgent to regenerate code
                    print("\n" + "="*60, file=sys.stderr)
                    print("Auto-triggering CodeGeneratorAgent to regenerate code...", file=sys.stderr)
                    print("="*60 + "\n", file=sys.stderr)
                    
                    try:
                        from agents.CodeGeneratorAgent import generate_python_code
                        # asyncio is already imported at the top
                        
                        # Extract simulation plan from fix request
                        plan_with_fixes = fix_request["simulation_plan"].copy()
                        # Add fix instructions to the plan so CodeGeneratorAgent can use them
                        plan_with_fixes["_fix_instructions"] = fix_request["fix_instructions"]
                        plan_with_fixes["_error_context"] = fix_request["error_context"]
                        plan_with_fixes["_error_summary"] = fix_request["error_summary"]
                        plan_with_fixes["_explanation"] = fix_request["explanation"]
                        
                        # Load datasets manifest if available
                        datasets_manifest = None
                        if args.datasets:
                            try:
                                datasets_manifest = load_json_from_file(args.datasets)
                            except Exception as e:
                                print(f"Warning: Failed to load datasets manifest for code regeneration: {e}", file=sys.stderr)
                        
                        # Generate corrected code with datasets manifest
                        code_result = asyncio.run(generate_python_code(plan_with_fixes, datasets_manifest))
                        
                        # Save regenerated code
                        code_path = Path(args.python_file)
                        output_path = code_path.parent / "simulation.py"
                        with open(output_path, 'w') as f:
                            f.write(code_result["python_code"])
                        
                        print(f"Regenerated code saved to: {output_path}", file=sys.stderr)
                        print(f"\nRetrying simulation (attempt {retry_count + 2}/{max_retries + 1})...", file=sys.stderr)
                        retry_count += 1
                        # Continue loop to re-run with fixed code
                        continue
                        
                    except Exception as e:
                        print(f"Warning: Failed to auto-regenerate code: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                        print("\nFix request (manual regeneration required):", file=sys.stderr)
                        print(json.dumps(fix_request, indent=2))
                        # Break out of retry loop if code regeneration fails
                        break
                
                except Exception as e:
                    print(f"Warning: Failed to auto-trigger ErrorFeedbackAgent: {e}", file=sys.stderr)
                    print("Error result:", file=sys.stderr)
                    print(json.dumps(result, indent=2))
                    # Break out of retry loop if ErrorFeedbackAgent fails
                    break
            else:
                # If auto-fix is disabled or max retries reached, break
                break
        
        # Output final result (success or final error after all retries)
        if result:
            if result.get("status") == "error" and retry_count >= max_retries:
                print(f"\n✗ Simulation failed after {max_retries + 1} attempts. Final error:", file=sys.stderr)
            print(json.dumps(result, indent=2))
        else:
            print("Error: No simulation result available", file=sys.stderr)
            sys.exit(1)
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

