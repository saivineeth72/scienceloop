#!/usr/bin/env python3
"""
Flask API server for ScienceLoop backend
Handles PDF uploads and processes them with PaperUnderstandingAgent
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

# Import the PaperUnderstandingAgent
# Using direct method with SpoonOS PDF tool (allows temperature control)
from agents.PaperUnderstandingAgent import analyze_pdf_direct
from agents.KnowledgeGraphAgent import build_knowledge_graph
from agents.HypothesisAgent import generate_hypothesis
from agents.SimulationPlanAgent import generate_simulation_plan
from agents.CodeGeneratorAgent import generate_python_code
from agents.DatasetAgent import generate_datasets
from agents.SimulationRunnerAgent import run_simulation_agent
from agents.ReportAgent import generate_report

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
OUTPUT_FOLDER = Path(__file__).parent / 'outputs'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'pdf'}

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/api/analyze-pdf', methods=['POST'])
def analyze_pdf():
    """
    Endpoint to upload and analyze a PDF file
    
    Expects: multipart/form-data with 'file' field
    Returns: JSON with analysis results and output file path
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        file_path = Path(app.config['UPLOAD_FOLDER']) / safe_filename
        file.save(str(file_path))
        
        print(f"File uploaded: {file_path}", flush=True)
        
        # Check for OpenAI API key before processing
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({
                'error': 'OPENAI_API_KEY not found. Please set it in your .env file or environment variables.'
            }), 500
        
        # Process PDF with PaperUnderstandingAgent (using SpoonOS PDF tool)
        print(f"Processing PDF with PaperUnderstandingAgent...", flush=True)
        try:
            # Run the async function (uses SpoonOS PDF tool with temperature=0.3)
            result = asyncio.run(analyze_pdf_direct(str(file_path)))
        except Exception as e:
            print(f"Error processing PDF: {e}", flush=True)
            error_msg = str(e)
            # Provide more helpful error messages
            if 'OPENAI_API_KEY' in error_msg or 'api key' in error_msg.lower():
                error_msg = 'OPENAI_API_KEY not configured. Please set it in your .env file or environment variables.'
            elif 'Provider service unavailable' in error_msg:
                error_msg = 'OpenAI API service unavailable. Please check your API key and network connection.'
            return jsonify({
                'error': f'Failed to process PDF: {error_msg}'
            }), 500
        
        # Save output to outputs folder
        output_filename = f"{timestamp}_{Path(filename).stem}_understand.json"
        output_path = OUTPUT_FOLDER / output_filename
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Output saved to: {output_path}", flush=True)
        
        # Return result
        return jsonify({
            'success': True,
            'result': result,
            'output_path': str(output_path),
            'filename': output_filename
        }), 200
        
    except Exception as e:
        print(f"Error in analyze_pdf endpoint: {e}", flush=True)
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/build-knowledge-graph', methods=['POST'])
def build_knowledge_graph_endpoint():
    """
    Endpoint to build knowledge graph from PaperUnderstandingAgent results
    
    Expects: JSON with 'result' field containing the analysis results
    Returns: JSON with knowledge graph (nodes and edges)
    """
    try:
        data = request.get_json()
        
        if not data or 'result' not in data:
            return jsonify({'error': 'No result data provided'}), 400
        
        result = data['result']
        
        # Check for OpenAI API key before processing
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({
                'error': 'OPENAI_API_KEY not found. Please set it in your .env file or environment variables.'
            }), 500
        
        # Build knowledge graph
        print(f"Building knowledge graph...", flush=True)
        try:
            kg_result = asyncio.run(build_knowledge_graph(result))
        except Exception as e:
            print(f"Error building knowledge graph: {e}", flush=True)
            error_msg = str(e)
            if 'OPENAI_API_KEY' in error_msg or 'api key' in error_msg.lower():
                error_msg = 'OPENAI_API_KEY not configured. Please set it in your .env file or environment variables.'
            return jsonify({
                'error': f'Failed to build knowledge graph: {error_msg}'
            }), 500
        
        # Return result
        return jsonify({
            'success': True,
            'result': kg_result
        }), 200
        
    except Exception as e:
        print(f"Error in build_knowledge_graph endpoint: {e}", flush=True)
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/generate-hypothesis', methods=['POST'])
def generate_hypothesis_endpoint():
    """
    Endpoint to generate hypothesis from PaperUnderstandingAgent results and knowledge graph
    
    Expects: JSON with 'step1_output' and 'knowledge_graph' fields
    Returns: JSON with hypothesis and justification
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'step1_output' not in data:
            return jsonify({'error': 'No step1_output data provided'}), 400
        
        if 'knowledge_graph' not in data:
            return jsonify({'error': 'No knowledge_graph data provided'}), 400
        
        step1_output = data['step1_output']
        knowledge_graph = data['knowledge_graph']
        
        # Validate knowledge graph structure
        if not isinstance(knowledge_graph, dict) or 'nodes' not in knowledge_graph or 'edges' not in knowledge_graph:
            return jsonify({'error': 'Invalid knowledge_graph structure. Must have nodes and edges fields.'}), 400
        
        # Check for Anthropic API key before processing
        if not os.getenv('ANTHROPIC_API_KEY'):
            return jsonify({
                'error': 'ANTHROPIC_API_KEY not found. Please set it in your .env file or environment variables.'
            }), 500
        
        # Generate hypothesis
        print(f"Generating hypothesis from paper understanding and knowledge graph...", flush=True)
        try:
            hypothesis_result = asyncio.run(generate_hypothesis(step1_output, knowledge_graph))
        except Exception as e:
            print(f"Error generating hypothesis: {e}", flush=True)
            error_msg = str(e)
            if 'ANTHROPIC_API_KEY' in error_msg or 'api key' in error_msg.lower():
                error_msg = 'ANTHROPIC_API_KEY not configured. Please set it in your .env file or environment variables.'
            return jsonify({
                'error': f'Failed to generate hypothesis: {error_msg}'
            }), 500
        
        # Return result
        return jsonify({
            'success': True,
            'result': hypothesis_result
        }), 200
        
    except Exception as e:
        print(f"Error in generate_hypothesis endpoint: {e}", flush=True)
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/generate-simulation-plan', methods=['POST'])
def generate_simulation_plan_endpoint():
    """
    Endpoint to generate simulation plan from hypothesis, paper understanding, and knowledge graph
    
    Expects: JSON with 'hypothesis', 'step1_output', and 'knowledge_graph' fields
    Returns: JSON with simulation plan
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'hypothesis' not in data:
            return jsonify({'error': 'No hypothesis data provided'}), 400
        
        if 'step1_output' not in data:
            return jsonify({'error': 'No step1_output data provided'}), 400
        
        if 'knowledge_graph' not in data:
            return jsonify({'error': 'No knowledge_graph data provided'}), 400
        
        hypothesis = data['hypothesis']
        step1_output = data['step1_output']
        knowledge_graph = data['knowledge_graph']
        
        # Validate inputs
        if not isinstance(hypothesis, dict) or 'hypothesis' not in hypothesis:
            return jsonify({'error': 'Invalid hypothesis structure. Must have hypothesis field.'}), 400
        
        if not isinstance(knowledge_graph, dict) or 'nodes' not in knowledge_graph or 'edges' not in knowledge_graph:
            return jsonify({'error': 'Invalid knowledge_graph structure. Must have nodes and edges fields.'}), 400
        
        # Check for Anthropic API key before processing
        if not os.getenv('ANTHROPIC_API_KEY'):
            return jsonify({
                'error': 'ANTHROPIC_API_KEY not found. Please set it in your .env file or environment variables.'
            }), 500
        
        # Generate simulation plan
        print(f"Generating simulation plan from hypothesis, paper understanding, and knowledge graph...", flush=True)
        try:
            simulation_plan_result = asyncio.run(generate_simulation_plan(hypothesis, step1_output, knowledge_graph))
        except Exception as e:
            print(f"Error generating simulation plan: {e}", flush=True)
            error_msg = str(e)
            if 'ANTHROPIC_API_KEY' in error_msg or 'api key' in error_msg.lower():
                error_msg = 'ANTHROPIC_API_KEY not configured. Please set it in your .env file or environment variables.'
            return jsonify({
                'error': f'Failed to generate simulation plan: {error_msg}'
            }), 500
        
        # Return result
        return jsonify({
            'success': True,
            'result': simulation_plan_result
        }), 200
        
    except Exception as e:
        print(f"Error in generate_simulation_plan endpoint: {e}", flush=True)
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/generate-datasets', methods=['POST'])
def generate_datasets_endpoint():
    """
    Endpoint to generate datasets from simulation plan
    
    Expects: JSON with 'simulation_plan' field
    Returns: JSON with dataset_type, datasets, and reasoning
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'simulation_plan' not in data:
            return jsonify({'error': 'No simulation_plan data provided'}), 400
        
        simulation_plan = data['simulation_plan']
        
        # Validate simulation plan structure
        required_fields = ["simulation_equations", "constants_required", "variables_to_vary", "procedure_steps"]
        for field in required_fields:
            if field not in simulation_plan:
                return jsonify({'error': f'Invalid simulation_plan structure. Missing required field: {field}'}), 400
        
        # Check for Anthropic API key before processing
        if not os.getenv('ANTHROPIC_API_KEY'):
            return jsonify({
                'error': 'ANTHROPIC_API_KEY not found. Please set it in your .env file or environment variables.'
            }), 500
        
        # Generate datasets
        print(f"[API] DEBUG: Generating datasets from simulation plan...", flush=True)
        print(f"[API] DEBUG: Simulation plan keys: {list(simulation_plan.keys())}", flush=True)
        print(f"[API] DEBUG: OUTPUT_FOLDER: {OUTPUT_FOLDER}", flush=True)
        
        try:
            # Use OUTPUT_FOLDER as the output directory for datasets
            datasets_result = asyncio.run(generate_datasets(simulation_plan, OUTPUT_FOLDER))
            
            print(f"[API] DEBUG: Dataset generation completed", flush=True)
            print(f"[API] DEBUG: Dataset type: {datasets_result.get('dataset_type', 'unknown')}", flush=True)
            print(f"[API] DEBUG: Number of datasets: {len(datasets_result.get('datasets', {}))}", flush=True)
            print(f"[API] DEBUG: Dataset names: {list(datasets_result.get('datasets', {}).keys())}", flush=True)
            
            # Verify files exist before returning
            for name, filepath in datasets_result.get('datasets', {}).items():
                from pathlib import Path
                if Path(filepath).exists():
                    file_size = Path(filepath).stat().st_size
                    print(f"[API] DEBUG: ✓ Dataset '{name}': {filepath} ({file_size} bytes)", flush=True)
                else:
                    print(f"[API] WARNING: ✗ Dataset '{name}': File not found at {filepath}", flush=True)
            
        except Exception as e:
            print(f"[API] ERROR: Error generating datasets: {e}", flush=True)
            import traceback
            print(f"[API] ERROR: Traceback: {traceback.format_exc()}", flush=True)
            error_msg = str(e)
            if 'ANTHROPIC_API_KEY' in error_msg or 'api key' in error_msg.lower():
                error_msg = 'ANTHROPIC_API_KEY not configured. Please set it in your .env file or environment variables.'
            return jsonify({
                'error': f'Failed to generate datasets: {error_msg}'
            }), 500
        
        # Return result
        print(f"[API] DEBUG: Returning datasets result to frontend", flush=True)
        return jsonify({
            'success': True,
            'result': datasets_result
        }), 200
        
    except Exception as e:
        print(f"Error in generate_datasets endpoint: {e}", flush=True)
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/generate-code', methods=['POST'])
def generate_code_endpoint():
    """
    Endpoint to generate Python code from simulation plan
    
    Expects: JSON with 'simulation_plan' field
    Returns: JSON with python_code field containing the complete Python script
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'simulation_plan' not in data:
            return jsonify({'error': 'No simulation_plan data provided'}), 400
        
        simulation_plan = data['simulation_plan']
        
        # Validate simulation plan structure
        required_fields = ["simulation_equations", "constants_required", "variables_to_vary", "procedure_steps", "expected_outcomes"]
        for field in required_fields:
            if field not in simulation_plan:
                return jsonify({'error': f'Invalid simulation_plan structure. Missing required field: {field}'}), 400
        
        # Check for OpenAI API key before processing
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({
                'error': 'OPENAI_API_KEY not found. Please set it in your .env file or environment variables.'
            }), 500
        
        # Generate Python code
        print(f"Generating Python code from simulation plan...", flush=True)
        try:
            code_result = asyncio.run(generate_python_code(simulation_plan))
        except Exception as e:
            print(f"Error generating code: {e}", flush=True)
            error_msg = str(e)
            if 'OPENAI_API_KEY' in error_msg or 'api key' in error_msg.lower():
                error_msg = 'OPENAI_API_KEY not configured. Please set it in your .env file or environment variables.'
            return jsonify({
                'error': f'Failed to generate code: {error_msg}'
            }), 500
        
        # Return result
        return jsonify({
            'success': True,
            'result': code_result
        }), 200
        
    except Exception as e:
        print(f"Error in generate_code endpoint: {e}", flush=True)
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/validate-code', methods=['POST'])
def validate_code_endpoint():
    """
    Endpoint to validate code using OpenAI and return corrected code if errors found
    
    Expects: JSON with 'python_code' and 'simulation_plan' fields
    Returns: JSON with corrected_code (if errors found) or original code
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'python_code' not in data:
            return jsonify({'error': 'No python_code provided'}), 400
        
        if 'simulation_plan' not in data:
            return jsonify({'error': 'No simulation_plan provided'}), 400
        
        python_code = data['python_code']
        simulation_plan = data['simulation_plan']
        dataset_metadata = data.get('dataset_metadata')  # Optional dataset metadata
        
        # Check for OpenAI API key before processing
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({
                'error': 'OPENAI_API_KEY not found. Please set it in your .env file or environment variables.'
            }), 500
        
        # Call Claude to validate code
        print(f"Validating code with Claude...", flush=True)
        try:
            from anthropic import AsyncAnthropic
            import asyncio
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return jsonify({
                    'error': 'ANTHROPIC_API_KEY not found. Please set it in your .env file or environment variables.'
                }), 500
            
            client = AsyncAnthropic(api_key=api_key)
            model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
            
            # Build dataset metadata section if available
            dataset_info = ""
            if dataset_metadata:
                dataset_info = f"""

AVAILABLE DATASETS METADATA:
{json.dumps(dataset_metadata, indent=2)}

IMPORTANT: The code should correctly reference these datasets. Check:
- File paths match the dataset filenames/paths
- Column names (if CSV) match what the code expects
- Row counts are reasonable for the operations
- Data types are appropriate for the operations"""
            
            prompt = f"""You are a code validator. Review the following Python code and simulation plan.

SIMULATION PLAN:
{json.dumps(simulation_plan, indent=2)}
{dataset_info}

PYTHON CODE:
```python
{python_code}
```

Your task:
1. Check if the code has any errors (syntax errors, logical errors, missing imports, incorrect implementations)
2. Check if the code correctly implements the simulation plan
3. Check if the code correctly references datasets (if datasets are provided):
   - CRITICAL: If datasets are provided in the metadata, the code MUST load and use them
   - Verify file paths match the dataset metadata
   - Verify column names match (if CSV files)
   - Verify data loading operations are correct
   - If the code has hardcoded data but datasets are provided, REPLACE hardcoded data with dataset loading

4. CRITICAL RULES:
   - If ANY errors are found (including missing dataset usage), you MUST provide CORRECTED code that fixes ALL errors
   - The corrected code MUST be different from the original if errors exist
   - Do NOT return the same code if errors were found - you must fix them
   - If datasets are provided but not used, you MUST modify the code to load and use the datasets
   - Only return the original code unchanged if NO errors are found
   - IMPORTANT: Check for deprecated/removed functions:
     * `scipy.integrate.cumtrapz` is deprecated/removed - use `scipy.integrate.cumulative_trapezoid` instead
     * `scipy.misc` functions may be deprecated - check for alternatives
     * Ensure all imports use current, non-deprecated function names

5. If the code is correct (no errors), return the original code unchanged

Respond with ONLY valid JSON in this exact format:
{{
  "has_errors": true/false,
  "python_code": "corrected code here (MUST be different from input if has_errors is true)",
  "errors_found": ["list of errors if any", "or empty array if none"]
}}

IMPORTANT: If has_errors is true, the python_code field MUST contain corrected code that fixes all the errors listed in errors_found. Do NOT return the same code if errors exist.

Return ONLY the JSON, no additional text or explanation."""
            
            # Print the prompt being sent to Claude
            print("="*80, flush=True)
            print("[VALIDATE-CODE] PROMPT SENT TO CLAUDE:", flush=True)
            print("="*80, flush=True)
            print(f"Model: {model}", flush=True)
            print(f"Temperature: 0.2", flush=True)
            print(f"Max tokens: 8192", flush=True)
            print(f"\nPrompt length: {len(prompt)} characters", flush=True)
            print(f"\nFull prompt:", flush=True)
            print(prompt, flush=True)
            print("="*80, flush=True)
            
            async def validate_code():
                response = await client.messages.create(
                    model=model,
                    max_tokens=8192,  # Claude supports larger context
                    temperature=0.2,
                    system="You are a code validator that checks Python code for errors and provides corrections. Always respond with valid JSON only.",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                return response.content[0].text
            
            llm_response = asyncio.run(validate_code())
            
            # Print the response received from Claude
            print("="*80, flush=True)
            print("[VALIDATE-CODE] RESPONSE RECEIVED FROM CLAUDE:", flush=True)
            print("="*80, flush=True)
            print(f"Response length: {len(llm_response)} characters", flush=True)
            print(f"\nFull response:", flush=True)
            print(llm_response, flush=True)
            print("="*80, flush=True)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = json.loads(llm_response)
            
            # Validate response structure
            if "python_code" not in result:
                raise Exception("LLM response missing required field: python_code")
            
            if "has_errors" not in result:
                result["has_errors"] = False
            
            if "errors_found" not in result:
                result["errors_found"] = []
            
            # Validate that if errors are found, the code should be different
            if result["has_errors"] and len(result["errors_found"]) > 0:
                # Check if the returned code is the same as input (after normalizing whitespace)
                returned_code_normalized = result["python_code"].strip()
                input_code_normalized = python_code.strip()
                
                if returned_code_normalized == input_code_normalized:
                    print(f"WARNING: Claude reported errors but returned the same code. Errors: {result['errors_found']}", flush=True)
                    print("This may indicate the model didn't properly correct the code.", flush=True)
                    # We'll still return it, but log the warning
                    # Optionally, we could retry or mark it differently
            
            # Return result
            return jsonify({
                'success': True,
                'result': result
            }), 200
            
        except Exception as e:
            print(f"Error validating code: {e}", flush=True)
            import traceback
            print(f"Traceback: {traceback.format_exc()}", flush=True)
            error_msg = str(e)
            if 'ANTHROPIC_API_KEY' in error_msg or 'api key' in error_msg.lower():
                error_msg = 'ANTHROPIC_API_KEY not configured. Please set it in your .env file or environment variables.'
            return jsonify({
                'error': f'Failed to validate code: {error_msg}'
            }), 500
        
    except Exception as e:
        print(f"Error in validate_code endpoint: {e}", flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/run-simulation', methods=['POST'])
def run_simulation_endpoint():
    """
    Endpoint to run simulation using SimulationRunnerAgent
    
    Expects: JSON with 'python_code', 'simulation_plan', and 'dataset_paths' fields
    Returns: JSON with simulation result (status, stdout, stderr, exit_code, etc.)
    """
    print("="*80, flush=True)
    print("[RUN-SIMULATION] ENDPOINT CALLED", flush=True)
    print("="*80, flush=True)
    try:
        data = request.get_json()
        
        print(f"[RUN-SIMULATION] Request received", flush=True)
        print(f"[RUN-SIMULATION] Data keys: {list(data.keys()) if data else 'None'}", flush=True)
        
        if not data:
            print("[RUN-SIMULATION] ERROR: No data provided", flush=True)
            return jsonify({'error': 'No data provided'}), 400
        
        if 'python_code' not in data:
            print("[RUN-SIMULATION] ERROR: No python_code provided", flush=True)
            return jsonify({'error': 'No python_code provided'}), 400
        
        python_code = data['python_code']
        simulation_plan = data.get('simulation_plan')
        dataset_paths = data.get('dataset_paths')  # Dict or list of dataset file paths
        
        print(f"[RUN-SIMULATION] Python code length: {len(python_code) if python_code else 0}", flush=True)
        print(f"[RUN-SIMULATION] Has simulation_plan: {simulation_plan is not None}", flush=True)
        print(f"[RUN-SIMULATION] Has dataset_paths: {dataset_paths is not None}", flush=True)
        if dataset_paths:
            if isinstance(dataset_paths, dict):
                print(f"[RUN-SIMULATION] Dataset paths (dict): {list(dataset_paths.keys())}", flush=True)
            elif isinstance(dataset_paths, list):
                print(f"[RUN-SIMULATION] Dataset paths (list): {len(dataset_paths)} items", flush=True)
        
        # Save Python code to a file in outputs directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        code_filename = f"{timestamp}_simulation.py"
        code_path = OUTPUT_FOLDER / code_filename
        
        with open(code_path, 'w') as f:
            f.write(python_code)
        
        print(f"[RUN-SIMULATION] Python code saved to: {code_path}", flush=True)
        print(f"[RUN-SIMULATION] Code preview (first 500 chars):\n{python_code[:500]}", flush=True)
        
        # Check for potential infinite loops or blocking operations
        if 'while True' in python_code and 'break' not in python_code.split('while True')[1][:500]:
            print(f"[RUN-SIMULATION] WARNING: Potential infinite loop detected (while True without break nearby)", flush=True)
        if 'input(' in python_code:
            print(f"[RUN-SIMULATION] WARNING: Code contains input() calls which will block", flush=True)
        if 'plt.show()' in python_code:
            print(f"[RUN-SIMULATION] WARNING: Code contains plt.show() which will block", flush=True)
        
        # Run simulation using SimulationRunnerAgent
        print(f"[RUN-SIMULATION] Calling SimulationRunnerAgent...", flush=True)
        print(f"[RUN-SIMULATION] Parameters:", flush=True)
        print(f"  - python_file_path: {code_path}", flush=True)
        print(f"  - dataset_paths: {dataset_paths}", flush=True)
        print(f"  - simulation_workdir: None", flush=True)
        
        try:
            result = asyncio.run(run_simulation_agent(
                python_file_path=str(code_path),
                dataset_paths=dataset_paths,
                simulation_workdir=None
            ))
            
            print(f"[RUN-SIMULATION] SimulationRunnerAgent returned successfully", flush=True)
            print(f"[RUN-SIMULATION] Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}", flush=True)
            print(f"[RUN-SIMULATION] Result status: {result.get('status') if isinstance(result, dict) else 'N/A'}", flush=True)
            print(f"[RUN-SIMULATION] Result exit_code: {result.get('exit_code') if isinstance(result, dict) else 'N/A'}", flush=True)
            print(f"[RUN-SIMULATION] Result stdout length: {len(str(result.get('stdout', ''))) if isinstance(result, dict) else 'N/A'}", flush=True)
            print(f"[RUN-SIMULATION] Result stderr length: {len(str(result.get('stderr', ''))) if isinstance(result, dict) else 'N/A'}", flush=True)
            
        except Exception as e:
            print(f"[RUN-SIMULATION] ERROR running simulation: {e}", flush=True)
            import traceback
            print(f"[RUN-SIMULATION] Traceback: {traceback.format_exc()}", flush=True)
            return jsonify({
                'error': f'Failed to run simulation: {str(e)}'
            }), 500
        
        # Return result
        print(f"[RUN-SIMULATION] Returning success response", flush=True)
        print("="*80, flush=True)
        return jsonify({
            'success': True,
            'result': result
        }), 200
        
    except Exception as e:
        print(f"Error in run_simulation endpoint: {e}", flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/generate-report', methods=['POST'])
def generate_report_endpoint():
    """
    Endpoint to generate report using ReportAgent
    
    Expects: JSON with 'paper_understanding', 'knowledge_graph', 'hypothesis', 
             'simulation_plan', and 'simulation_result' fields
    Returns: JSON with report_markdown and summary
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['paper_understanding', 'knowledge_graph', 'hypothesis', 
                          'simulation_plan', 'simulation_result']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'No {field} data provided'}), 400
        
        paper_understanding = data['paper_understanding']
        knowledge_graph = data['knowledge_graph']
        hypothesis = data['hypothesis']
        simulation_plan = data['simulation_plan']
        simulation_result = data['simulation_result']
        error_history = data.get('error_history')  # Optional
        
        # Read only outputs/results.csv file and add it to simulation_result
        results_csv_path = OUTPUT_FOLDER / 'results.csv'
        if results_csv_path.exists() and results_csv_path.is_file():
            try:
                file_size = results_csv_path.stat().st_size
                if file_size <= 1024 * 1024:  # 1MB limit
                    csv_content = results_csv_path.read_text(encoding='utf-8', errors='ignore')
                    # Add results.csv content to simulation_result
                    if isinstance(simulation_result, dict):
                        simulation_result = simulation_result.copy()
                        if 'artifacts' not in simulation_result:
                            simulation_result['artifacts'] = []
                        # Add results.csv as a single artifact with content
                        simulation_result['artifacts'] = [{
                            "filename": "results.csv",
                            "path": str(results_csv_path),
                            "content": csv_content,
                            "size_bytes": file_size
                        }]
                        print(f"Read results.csv file: {file_size} bytes", flush=True)
                    else:
                        print(f"Warning: simulation_result is not a dict, cannot add CSV content", flush=True)
                else:
                    print(f"Skipped large results.csv file: {file_size} bytes", flush=True)
            except Exception as e:
                print(f"Error reading results.csv file: {e}", flush=True)
        else:
            print(f"results.csv not found at: {results_csv_path}", flush=True)
        
        # Print what's being sent to ReportAgent
        print("="*80, flush=True)
        print("DATA BEING SENT TO REPORTAGENT:", flush=True)
        print("="*80, flush=True)
        print(f"\npaper_understanding keys: {list(paper_understanding.keys()) if isinstance(paper_understanding, dict) else type(paper_understanding)}", flush=True)
        print(f"paper_understanding (first 500 chars): {str(paper_understanding)[:500]}...", flush=True)
        print(f"\nknowledge_graph keys: {list(knowledge_graph.keys()) if isinstance(knowledge_graph, dict) else type(knowledge_graph)}", flush=True)
        print(f"knowledge_graph nodes count: {len(knowledge_graph.get('nodes', [])) if isinstance(knowledge_graph, dict) else 'N/A'}", flush=True)
        print(f"knowledge_graph edges count: {len(knowledge_graph.get('edges', [])) if isinstance(knowledge_graph, dict) else 'N/A'}", flush=True)
        print(f"\nhypothesis keys: {list(hypothesis.keys()) if isinstance(hypothesis, dict) else type(hypothesis)}", flush=True)
        print(f"hypothesis (first 500 chars): {str(hypothesis)[:500]}...", flush=True)
        print(f"\nsimulation_plan keys: {list(simulation_plan.keys()) if isinstance(simulation_plan, dict) else type(simulation_plan)}", flush=True)
        print(f"simulation_plan (first 500 chars): {str(simulation_plan)[:500]}...", flush=True)
        print(f"\nsimulation_result keys: {list(simulation_result.keys()) if isinstance(simulation_result, dict) else type(simulation_result)}", flush=True)
        print(f"simulation_result.status: {simulation_result.get('status', 'N/A') if isinstance(simulation_result, dict) else 'N/A'}", flush=True)
        print(f"simulation_result.exit_code: {simulation_result.get('exit_code', 'N/A') if isinstance(simulation_result, dict) else 'N/A'}", flush=True)
        print(f"simulation_result.artifacts count: {len(simulation_result.get('artifacts', [])) if isinstance(simulation_result, dict) else 'N/A'}", flush=True)
        if isinstance(simulation_result, dict) and 'artifacts' in simulation_result:
            print(f"Artifact files: {[art.get('filename') for art in simulation_result.get('artifacts', [])]}", flush=True)
            # Show which artifacts have content
            artifacts_with_content = [art for art in simulation_result.get('artifacts', []) if 'content' in art]
            if artifacts_with_content:
                print(f"Artifacts WITH CONTENT: {[art.get('filename') for art in artifacts_with_content]}", flush=True)
                for art in artifacts_with_content:
                    content_preview = art.get('content', '')[:200] if art.get('content') else ''
                    print(f"  - {art.get('filename')}: {len(art.get('content', ''))} chars, preview: {content_preview}...", flush=True)
            else:
                print("No artifacts have content included", flush=True)
        print(f"simulation_result.stdout length: {len(str(simulation_result.get('stdout', ''))) if isinstance(simulation_result, dict) else 'N/A'} chars", flush=True)
        print(f"simulation_result.stderr length: {len(str(simulation_result.get('stderr', ''))) if isinstance(simulation_result, dict) else 'N/A'} chars", flush=True)
        print(f"\nerror_history: {error_history}", flush=True)
        print("="*80, flush=True)
        print("FULL JSON DATA (first 2000 chars):", flush=True)
        print(json.dumps(data, indent=2)[:2000] + "...", flush=True)
        print("="*80, flush=True)
        
        # Check for Anthropic API key before processing
        if not os.getenv('ANTHROPIC_API_KEY'):
            return jsonify({
                'error': 'ANTHROPIC_API_KEY not found. Please set it in your .env file or environment variables.'
            }), 500
        
        # Generate report
        print(f"Generating report with ReportAgent...", flush=True)
        try:
            report_result = asyncio.run(generate_report(
                paper_understanding=paper_understanding,
                knowledge_graph=knowledge_graph,
                hypothesis=hypothesis,
                simulation_plan=simulation_plan,
                simulation_result=simulation_result,
                error_history=error_history
            ))
        except Exception as e:
            print(f"Error generating report: {e}", flush=True)
            error_msg = str(e)
            if 'ANTHROPIC_API_KEY' in error_msg or 'api key' in error_msg.lower():
                error_msg = 'ANTHROPIC_API_KEY not configured. Please set it in your .env file or environment variables.'
            return jsonify({
                'error': f'Failed to generate report: {error_msg}'
            }), 500
        
        # Return result
        return jsonify({
            'success': True,
            'result': report_result
        }), 200
        
    except Exception as e:
        print(f"Error in generate_report endpoint: {e}", flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/datasets/<path:filepath>', methods=['GET'])
def serve_dataset_file(filepath):
    """
    Serve dataset files from the outputs/datasets directory.
    
    Expects: filepath relative to OUTPUT_FOLDER/datasets/
    Returns: File content with appropriate content-type
    """
    try:
        # Security: Only allow files from datasets directory
        # Remove any path traversal attempts
        safe_path = filepath.replace('..', '').lstrip('/')
        
        # Construct full path
        full_path = OUTPUT_FOLDER / "datasets" / safe_path
        
        # Verify file exists and is within OUTPUT_FOLDER
        if not full_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # Verify it's actually within OUTPUT_FOLDER/datasets
        try:
            full_path.resolve().relative_to((OUTPUT_FOLDER / "datasets").resolve())
        except ValueError:
            return jsonify({'error': 'Invalid file path'}), 403
        
        # Determine content type
        content_type = 'text/plain'
        if filepath.endswith('.csv'):
            content_type = 'text/csv'
        elif filepath.endswith('.json'):
            content_type = 'application/json'
        elif filepath.endswith('.txt'):
            content_type = 'text/plain'
        
        print(f"[API] DEBUG: Serving dataset file: {full_path}", flush=True)
        return send_file(str(full_path), mimetype=content_type)
        
    except Exception as e:
        print(f"[API] ERROR: Error serving dataset file: {e}", flush=True)
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


if __name__ == '__main__':
    # Get port from environment or default to 5001 (5000 is often used by AirPlay on macOS)
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting ScienceLoop backend server on port {port}", flush=True)
    app.run(host='0.0.0.0', port=port, debug=debug)

