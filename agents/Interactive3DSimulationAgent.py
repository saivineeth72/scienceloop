#!/usr/bin/env python3
"""
Interactive3DSimulationAgent - SpoonOS Agent

Analyzes research paper outputs and generates toggleable interactive simulations.

Process:
1. Analyzes all paper files (images, report, simulation.py, hypothesis, knowledge graph)
2. Identifies potential simulations based on the outputs
3. Creates multiple toggleable interactive simulations
4. Works for any domain (physics, biology, code, ML)

Uses Gemini 2.5 Pro for analysis and HTML generation.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import base64

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os


def collect_paper_artifacts(paper_dir: Path) -> Dict[str, Any]:
    """Collect all artifacts from the paper folder."""
    artifacts = {
        "png_images": [],
        "simulation_code": None,
        "report_md": None,
        "report_json": None,
        "hypothesis": None,
        "knowledge_graph": None,
        "understand": None,
        "simulation_plan": None,
        "simulation_3d_plan": None,
        "datasets_manifest": None,
        "embedded_datasets": {}
    }
    
    # Find reference images recursively (png, jpg, jpeg, svg, webp)
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.svg", "*.webp"]:
        for img_file in paper_dir.rglob(ext):
            try:
                rel = img_file.relative_to(paper_dir)
            except Exception:
                rel = img_file.name
            artifacts["png_images"].append({
                "name": img_file.name,
                "path": str(rel)
            })
    artifacts["png_images"].sort(key=lambda x: x["name"]) 
    
    # Read simulation.py if exists
    sim_py = paper_dir / "simulation.py"
    if sim_py.exists():
        with open(sim_py, 'r') as f:
            artifacts["simulation_code"] = f.read()
    
    # Read report.md if exists
    report_md = paper_dir / "report.md"
    if report_md.exists():
        with open(report_md, 'r') as f:
            artifacts["report_md"] = f.read()
    
    # Read report.json if exists
    report_json = paper_dir / "report.json"
    if report_json.exists():
        with open(report_json, 'r') as f:
            artifacts["report_json"] = json.load(f)
    
    # Read hypothesis if exists
    hypothesis_json = paper_dir / "hypothesis.json"
    if hypothesis_json.exists():
        with open(hypothesis_json, 'r') as f:
            artifacts["hypothesis"] = json.load(f)
    
    # Read knowledge graph if exists
    kg_json = paper_dir / "knowledge_graph.json"
    if kg_json.exists():
        with open(kg_json, 'r') as f:
            artifacts["knowledge_graph"] = json.load(f)
    
    # Read understand.json if exists
    understand_json = paper_dir / "understand.json"
    if understand_json.exists():
        with open(understand_json, 'r') as f:
            artifacts["understand"] = json.load(f)
    
    # Read simulation_plan.json if exists
    sim_plan_json = paper_dir / "simulation_plan.json"
    if sim_plan_json.exists():
        with open(sim_plan_json, 'r') as f:
            artifacts["simulation_plan"] = json.load(f)

    sim_3d_plan_json = paper_dir / "3d_simulation_plan.json"
    if sim_3d_plan_json.exists():
        with open(sim_3d_plan_json, 'r') as f:
            artifacts["simulation_3d_plan"] = json.load(f)

    datasets_manifest = paper_dir / "datasets_manifest.json"
    if datasets_manifest.exists():
        with open(datasets_manifest, 'r') as f:
            dm = json.load(f)
            artifacts["datasets_manifest"] = dm
            try:
                if isinstance(dm.get("datasets"), dict):
                    for name, rel_path in dm["datasets"].items():
                        candidates = []
                        rp = Path(rel_path)
                        candidates.append(rp)  # project-root relative or absolute
                        candidates.append(paper_dir / rp)  # relative to paper dir
                        candidates.append(Path(__file__).parent.parent / rp)  # relative to project root
                        # also try just the basename inside paper_dir
                        candidates.append(paper_dir / rp.name)
                        target = None
                        for c in candidates:
                            try:
                                if c.exists():
                                    target = c
                                    break
                            except Exception:
                                continue
                        if target and target.exists():
                            try:
                                import pandas as pd
                                df = pd.read_csv(target)
                                artifacts["embedded_datasets"][name] = df.to_dict(orient="list")
                            except Exception:
                                pass
            except Exception:
                pass
    
    return artifacts


def identify_potential_simulations(artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Analyze paper artifacts and identify potential interactive simulations.
    
    Returns:
        List of simulation specifications
    """
    from google import generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not found")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Prepare context
    png_images = artifacts.get("png_images", [])
    hypothesis = artifacts.get("hypothesis", {})
    understand = artifacts.get("understand", {})
    knowledge_graph = artifacts.get("knowledge_graph", {})
    report = artifacts.get("report_json", {}) or artifacts.get("report_md", "")
    simulation_code = artifacts.get("simulation_code", "")
    
    prompt = f"""You are analyzing a research paper's outputs to identify ADVANCED, RELEVANT interactive simulations.

PAPER CONTEXT:
- PNG Images (final outputs - these show what was actually visualized): {[img['name'] for img in png_images]}
- Hypothesis: {json.dumps(hypothesis, indent=2)[:800] if hypothesis else 'Not available'}
- Summary: {json.dumps(understand, indent=2)[:800] if understand else 'Not available'}
- Knowledge Graph: {json.dumps(knowledge_graph, indent=2)[:1500] if knowledge_graph else 'Not available'}
- Report: {str(report)[:1500] if report else 'Not available'}
- Simulation Code (equations and constants): {simulation_code[:3000] if simulation_code else 'Not available'}

CRITICAL REQUIREMENTS:
1. Each simulation must be UNIQUE and NON-OVERLAPPING
2. Must be ADVANCED - use actual equations from simulation.py, not generic animations
3. Must be RELEVANT - directly based on the PNG images and research findings
4. Must be HIGHLY INTERACTIVE - parameter sliders, real-time updates, zoom, pan, drag
5. Must use ACTUAL DATA - extract equations, constants, and data processing from simulation.py
6. Each simulation should visualize a DIFFERENT aspect of the research

Your task: Identify 2-3 UNIQUE, ADVANCED interactive simulations. Each simulation should:
- Be based on a SPECIFIC PNG image or research finding
- Use ACTUAL EQUATIONS from simulation.py (not approximations)
- Have MULTIPLE interactive controls (temperature, concentration, time, etc.)
- Show REAL-TIME calculations and updates
- Be visually sophisticated (not just basic plots)
- Be educational and help understand the research

Return ONLY valid JSON array in this format:
[
  {{
    "id": "sim1",
    "title": "Specific Simulation Title",
    "description": "Detailed description of what this simulation shows and why it's important",
    "type": "arrhenius_plot|molecular_dynamics|energy_landscape|network_analysis|data_exploration|etc",
    "based_on_png": "exact PNG image filename this is based on",
    "equations_used": ["list of specific equations from simulation.py"],
    "interactive_features": ["temperature_slider", "concentration_slider", "zoom", "pan", "real_time_calculation", "parameter_sweep"],
    "data_source": "simulation.py|knowledge_graph|report",
    "visualization_type": "2d_plot|3d_surface|animated_chain|force_graph|etc",
    "key_insights": "What key research insight this simulation helps verify/understand"
  }}
]

Limit to 2-3 UNIQUE simulations. Avoid overlap - each should show a different aspect."""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000,
                response_mime_type="application/json"
            )
        )
        
        simulations = json.loads(response.text.strip())
        return simulations if isinstance(simulations, list) else []
    except Exception as e:
        print(f"Warning: Could not identify simulations: {e}", file=sys.stderr)
        return []


def generate_interactive_html(artifacts: Dict[str, Any], simulations: List[Dict[str, Any]]) -> str:
    """
    Generate complete HTML with toggleable interactive simulations.
    
    Args:
        artifacts: All collected artifacts
        simulations: List of identified simulations
    
    Returns:
        Complete HTML code
    """
    from google import generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not found")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Prepare context
    png_images = artifacts.get("png_images", [])
    hypothesis = artifacts.get("hypothesis", {})
    understand = artifacts.get("understand", {})
    knowledge_graph = artifacts.get("knowledge_graph", {})
    report = artifacts.get("report_json", {}) or artifacts.get("report_md", "")
    simulation_code = artifacts.get("simulation_code", "")
    
    prompt = f"""You are creating an ADVANCED, HIGHLY INTERACTIVE visualization page for a research paper.

PAPER CONTEXT:
- PNG Images: {[img['name'] for img in png_images]}
- Hypothesis: {json.dumps(hypothesis, indent=2)[:1000] if hypothesis else 'Not available'}
- Summary: {json.dumps(understand, indent=2)[:1000] if understand else 'Not available'}
- Knowledge Graph: {json.dumps(knowledge_graph, indent=2)[:3000] if knowledge_graph else 'Not available'}
- Report: {str(report)[:2000] if report else 'Not available'}
- Simulation Code (USE THESE EXACT EQUATIONS): 
```python
{simulation_code[:5000] if simulation_code else 'Not available'}
```

IDENTIFIED SIMULATIONS:
{json.dumps(simulations, indent=2)}

CRITICAL REQUIREMENTS FOR ADVANCED INTERACTIVE SIMULATIONS:

1. **Use ACTUAL EQUATIONS from simulation.py**:
   - Extract ALL constants (R, T0, DH, DS, DCp, m, etc.)
   - Implement EXACT functions (k_fold, k_unfold, DG_fold, etc.)
   - Use the SAME equations as the Python code
   - Calculate values in real-time using these equations

2. **Each Simulation Must Be ADVANCED**:
   - For Arrhenius plots: Show actual Arrhenius plots with data points calculated from equations
   - For protein folding: Show protein chain with actual folding/unfolding kinetics
   - For energy landscapes: Show 3D energy surfaces with actual free energy calculations
   - Use the PNG images as reference for what the visualization should look like
   - Make it match the style and data from the PNG images

3. **Highly Interactive Controls**:
   - Multiple sliders (temperature, concentration, time, etc.)
   - Real-time parameter updates
   - Zoom and pan for plots
   - Drag interactions where relevant
   - Play/pause for animations
   - Reset functionality
   - Parameter sweeps/animations

4. **Visual Sophistication**:
   - Professional plots with axes, labels, legends
   - Smooth animations (60fps)
   - Color-coded data points
   - Multiple data series on same plot
   - Interactive tooltips showing exact values
   - Grid lines, axis scaling
   - Professional styling

5. **Toggleable View System**: 
   - Buttons to switch between simulations + "Reference Images"
   - Each simulation in its own view (hidden/shown with JavaScript)
   - Only one view visible at a time
   - Active button highlighting
   - Smooth transitions

6. **Reference Images View**:
   - Show all PNG images in a clean grid
   - Make images clickable/zoomable
   - Show image names/titles
   - This should be one of the toggleable views

7. **Layout**:
   - Header with title
   - Left sidebar with:
     * Controls for current simulation (sliders, buttons)
     * Hypothesis display
     * Key metrics/stats
   - Main area with toggleable views
   - Use Tailwind CSS
   - Dark theme (#0f172a background, #1e293b panels)

8. **JavaScript Implementation**:
   - Extract ALL constants from simulation.py
   - Implement ALL functions from simulation.py exactly
   - Real-time calculations on parameter changes
   - Smooth animations with requestAnimationFrame
   - Proper event handlers for all controls
   - View switching logic that works correctly

EXAMPLE for Arrhenius Plot Simulation (ADVANCED):
- Extract ALL constants: R=0.001987, T0=298.15, DH_U2TS=20, DS_U2TS=0.06, DCp_U2TS=-0.45, m_U2TS=0.9, etc.
- Implement EXACT functions from simulation.py:
  ```javascript
  function k_fold(T, gnd) {{
    return D_fold * Math.exp(-((DH_U2TS - T*DS_U2TS + DCp_U2TS*(T - T0 - T*Math.log(T/T0)) - m_U2TS*gnd) / (R*T)));
  }}
  ```
- Create professional plot with:
  * Proper axes (1/T on x-axis, ln(k) on y-axis)
  * Grid lines
  * Axis labels and units
  * Legend for multiple curves
  * Data points as circles
  * Smooth curves connecting points
  * Interactive tooltips showing exact values
- Add MULTIPLE interactive controls:
  * Temperature range slider
  * Denaturant concentration slider
  * Toggle between different conditions
  * Zoom and pan
  * Export data button
- Calculate and display:
  * R-squared values
  * Activation energies
  * Rate constants at different temperatures
  * Real-time updates on parameter changes
- Match the PNG image exactly in style and data

CRITICAL REQUIREMENTS:
1. **MUST be SELF-CONTAINED**: All CSS and JavaScript MUST be inline in the HTML file. DO NOT reference external .js files.
2. **Reference Images View MUST be included**: 
   - Create a toggleable "Reference Images" view
   - Display ALL PNG images from the paper in a grid
   - Make images clickable/zoomable
   - This must be one of the toggle buttons
3. **Toggle Functionality MUST work**:
   - Create buttons for each simulation + "Reference Images"
   - JavaScript code to switch between views (all inline, not external)
   - Only one view visible at a time
   - Active button highlighting
4. **Complete Implementation**:
   - All JavaScript code must be in <script> tags within the HTML
   - All CSS must be in <style> tags within the HTML
   - No external file dependencies except CDN libraries (Tailwind, Plotly, etc.)

Return ONLY the complete, self-contained HTML code, starting with <!DOCTYPE html> and ending with </html>.
The HTML must include:
- All CSS inline in <style> tags
- All JavaScript inline in <script> tags
- Reference Images view with toggle button
- All simulation views with toggle buttons
- Working toggle functionality
- All equations from simulation.py implemented exactly
- Advanced, visually sophisticated simulations
- Interactive controls that work in real-time
- No overlapping/duplicate content between simulations"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=16384,
            )
        )
        
        html_code = response.text.strip()
        
        # Clean up markdown code blocks
        if html_code.startswith("```html"):
            html_code = html_code[7:]
        elif html_code.startswith("```"):
            html_code = html_code[3:]
        
        if html_code.endswith("```"):
            html_code = html_code[:-3]
        
        return html_code.strip()
    except Exception as e:
        raise Exception(f"Failed to generate HTML: {e}")


def parse_constants_from_sim_code(sim_code: str) -> Dict[str, float]:
    constants = {}
    pattern = r"^(\w+)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"
    for line in sim_code.splitlines():
        m = re.match(pattern, line.strip())
        if m:
            key = m.group(1)
            try:
                val = float(m.group(2))
                constants[key] = val
            except Exception:
                pass
    return constants


def parse_ranges_from_sim_code(sim_code: str) -> Dict[str, Dict[str, Any]]:
    ranges: Dict[str, Dict[str, Any]] = {}
    linspace_re = r"^(\w+)\s*=\s*np\.linspace\(\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*(\d+)\s*\)"
    logspace_re = r"^(\w+)\s*=\s*np\.logspace\(\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*(\d+)\s*\)"
    array_re = r"^(\w+)\s*=\s*np\.array\(\s*\[(.*?)\]\s*\)"
    for m in re.finditer(linspace_re, sim_code, flags=re.M):
        key = m.group(1)
        ranges[key] = {"type": "linspace", "start": float(m.group(2)), "end": float(m.group(3)), "num": int(m.group(4))}
    for m in re.finditer(logspace_re, sim_code, flags=re.M):
        key = m.group(1)
        ranges[key] = {"type": "logspace", "start": float(m.group(2)), "end": float(m.group(3)), "num": int(m.group(4))}
    for m in re.finditer(array_re, sim_code, flags=re.M|re.S):
        key = m.group(1)
        inner = m.group(2)
        vals: List[float] = []
        for tok in inner.split(','):
            tok = tok.strip()
            try:
                vals.append(float(tok))
            except Exception:
                pass
        if vals:
            ranges[key] = {"type": "array", "values": vals}
    return ranges
def generate_interactive_html_local(artifacts: Dict[str, Any]) -> str:
    pngs = artifacts.get("png_images", [])
    sim_code = artifacts.get("simulation_code") or ""
    consts = parse_constants_from_sim_code(sim_code)
    plan3d = artifacts.get("simulation_3d_plan") or {}
    datasets = artifacts.get("embedded_datasets", {})
    hypothesis = artifacts.get("hypothesis", {})
    report = artifacts.get("report_json", {}) or artifacts.get("report_md", "")

    png_data = [
        {"name": p["name"], "path": p["path"]} for p in pngs
    ]
    ds_json = json.dumps(datasets)
    consts_json = json.dumps(consts)
    plan_json = json.dumps(plan3d)

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\"/>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
<title>Interactive Simulation</title>
<script src=\"https://cdn.tailwindcss.com\"></script>
<script src=\"https://cdn.plot.ly/plotly-2.30.0.min.js\"></script>
</head>
<body class=\"bg-slate-900 text-slate-100\">
<div class=\"flex h-screen\">
  <div class=\"w-80 bg-slate-800 p-4 space-y-4\">
    <div class=\"text-xl font-semibold\">Controls</div>
    <label class=\"block text-sm\">Temperature (K)</label>
    <input id=\"tempSlider\" type=\"range\" min=\"278.15\" max=\"338.15\" step=\"0.1\" value=\"298.15\" class=\"w-full\" />
    <div id=\"tempVal\" class=\"text-sm\">298.15</div>
    <label class=\"block text-sm\">Denaturant (M)</label>
    <input id=\"gndSlider\" type=\"range\" min=\"0\" max=\"3\" step=\"0.1\" value=\"1\" class=\"w-full\" />
    <div id=\"gndVal\" class=\"text-sm\">1.0</div>
    <div class=\"pt-2\">
      <button data-view=\"gallery\" class=\"viewBtn w-full bg-slate-700 py-2 rounded mb-2\">Reference Images</button>
      <button data-view=\"arrhenius\" class=\"viewBtn w-full bg-slate-700 py-2 rounded mb-2\">Arrhenius Plots</button>
      <button data-view=\"surface\" class=\"viewBtn w-full bg-slate-700 py-2 rounded\">DG Surface</button>
    </div>
  </div>
  <div class=\"flex-1 p-4\">
    <div id=\"gallery\" class=\"view grid grid-cols-2 md:grid-cols-3 gap-4\"></div>
    <div id=\"arrhenius\" class=\"view hidden\">
      <div id=\"arrPlot1\" class=\"w-full h-[40vh]\"></div>
      <div id=\"arrPlot2\" class=\"w-full h-[40vh] mt-4\"></div>
    </div>
    <div id=\"surface\" class=\"view hidden\">
      <div id=\"surfacePlot\" class=\"w-full h-[80vh]\"></div>
    </div>
  </div>
</div>
<script>
const PNGS = {json.dumps(png_data)};
const CONSTS = {consts_json};
const PLAN3D = {plan_json};
const DATASETS = {ds_json};

function initGallery(){{
  const g = document.getElementById('gallery');
  g.innerHTML = '';
  PNGS.forEach(p=>{{
    const card = document.createElement('div');
    card.className = 'bg-slate-800 rounded p-2';
    const img = document.createElement('img');
    img.src = p.path;
    img.alt = p.name;
    img.className = 'w-full h-auto rounded';
    const cap = document.createElement('div');
    cap.className = 'mt-2 text-xs';
    cap.textContent = p.name;
    card.appendChild(img);
    card.appendChild(cap);
    g.appendChild(card);
  }});
}}

function k_fold(T,gnd){{
  const R = CONSTS.R||0.001987;
  const T0 = CONSTS.T0||298.15;
  const DH = CONSTS.DH_U2TS||20;
  const DS = CONSTS.DS_U2TS||0.06;
  const DCp = CONSTS.DCp_U2TS||-0.45;
  const m = CONSTS.m_U2TS||0.9;
  const Df = CONSTS.D_fold||1e7;
  const term = (DH - T*DS + DCp*(T - T0 - T*Math.log(T/T0)) - m*gnd)/(R*T);
  return Df*Math.exp(-term);
}}

function dg_fold(T,gnd){{
  const R = CONSTS.R||0.001987;
  const T0 = CONSTS.T0||298.15;
  const DH = CONSTS.DH||-50;
  const DS = CONSTS.DS||-0.15;
  const DCp = CONSTS.DCp||-1.15;
  const m = CONSTS.m||2.25;
  return DH - T*DS + DCp*(T - T0 - T*Math.log(T/T0)) - m*gnd;
}}

function gnd_constant_stability(T){{
  const T0 = CONSTS.T0||298.15;
  const DH = CONSTS.DH||-50;
  const DS = CONSTS.DS||-0.15;
  const DCp = CONSTS.DCp||-1.15;
  const m = CONSTS.m||2.25;
  const DGt = CONSTS.DG_target||-6.5;
  return (DGt - DH + T*DS - DCp*(T - T0 - T*Math.log(T/T0))) / m;
}}

function drawArrhenius(){{
  const Ts = Array.from({{length:25}},(_,i)=>278.15 + i*((338.15-278.15)/24));
  const gnds = [0,1,2,3];
  const traces1 = gnds.map(g=>{{
    const ks = Ts.map(T=>Math.log(k_fold(T,g)));
    return {{x: Ts.map(T=>1/T), y: ks, type:'scatter', mode:'lines', name:`gnd=${{g}} M`}};
  }});
  Plotly.newPlot('arrPlot1', traces1, {{title:'Arrhenius Plot (Constant gnd)', xaxis:{{title:'1/T (1/K)'}}, yaxis:{{title:'ln(k_fold)'}}, template:'plotly_dark'}});

  const gndVar = Ts.map(T=>gnd_constant_stability(T));
  const ks2 = Ts.map((T,i)=>Math.log(k_fold(T,gndVar[i])));
  Plotly.newPlot('arrPlot2', [{{x: Ts.map(T=>1/T), y: ks2, type:'scatter', mode:'lines', name:'Constant DG'}}], {{title:'Arrhenius Plot (Constant DG)', xaxis:{{title:'1/T (1/K)'}}, yaxis:{{title:'ln(k_fold)'}}, template:'plotly_dark'}});
}}

function drawSurface(){{
  const Tr = Array.from({{length:30}},(_,i)=>278.15 + i*((338.15-278.15)/29));
  const Gr = Array.from({{length:31}},(_,i)=>0 + i*(3/30));
  const Z = Tr.map(T=>Gr.map(g=>dg_fold(T,g)));
  Plotly.newPlot('surfacePlot', [{{z: Z, x:Tr, y:Gr, type:'surface', colorscale:'RdBu'}}], {{title:'DG vs T,gnd', scene:{{xaxis:{{title:'T (K)'}}, yaxis:{{title:'gnd (M)'}}, zaxis:{{title:'DG (kcal/mol)'}}}}, template:'plotly_dark'}});
}}

function setView(v){{
  document.querySelectorAll('.view').forEach(el=>el.classList.add('hidden'));
  document.getElementById(v).classList.remove('hidden');
}}

document.querySelectorAll('.viewBtn').forEach(btn=>{{
  btn.addEventListener('click',()=>setView(btn.dataset.view));
}});

document.getElementById('tempSlider').addEventListener('input',e=>{{
  document.getElementById('tempVal').textContent = Number(e.target.value).toFixed(2);
}});
document.getElementById('gndSlider').addEventListener('input',e=>{{
  document.getElementById('gndVal').textContent = Number(e.target.value).toFixed(1);
}});

initGallery();
drawArrhenius();
drawSurface();
setView('gallery');
</script>
</body>
</html>"""
    return html


def generate_js_bundle_local(artifacts: Dict[str, Any]) -> str:
    pngs = artifacts.get("png_images", [])
    sim_code = artifacts.get("simulation_code") or ""
    consts = parse_constants_from_sim_code(sim_code)
    ranges = parse_ranges_from_sim_code(sim_code)
    plan3d = artifacts.get("simulation_3d_plan") or {}
    datasets = artifacts.get("embedded_datasets", {})
    hypothesis = artifacts.get("hypothesis", {})
    report = artifacts.get("report_json", {}) or artifacts.get("report_md", "")
    png_data = [{"name": p["name"], "path": p["path"]} for p in pngs]
    let_mode = 'bio'
    if ('KNeighborsClassifier' in sim_code or 'sklearn' in sim_code or 'scikit-learn' in sim_code):
        let_mode = 'ml'
    elif ('log_D_mom' in sim_code or ' v(' in sim_code or 'beta_model_values' in sim_code):
        let_mode = 'physics'
    mode = let_mode
    head = "const MODE = '" + mode + "';\n" + \
           "const PNGS = " + json.dumps(png_data) + ";\n" + \
           "const CONSTS = " + json.dumps(consts) + ";\n" + \
           "const PLAN3D = " + json.dumps(plan3d) + ";\n" + \
           "const DATASETS = " + json.dumps(datasets) + ";\n" + \
           "const RANGES = " + json.dumps(ranges) + ";\n" + \
           "const HYPOTHESIS = " + json.dumps(hypothesis) + ";\n" + \
           "const REPORT = " + json.dumps(report) + ";\n\n"
    body = """
const THEME = {
  paper_bgcolor: '#0f172a',
  plot_bgcolor: '#0b1220',
  font: {family: 'Inter, system-ui, -apple-system, sans-serif', color: '#e5e7eb'},
  gridcolor: '#334155',
  axis: {linecolor: '#475569', tickcolor: '#475569', zerolinecolor: '#334155'}
};
const COLORS = ['#60a5fa','#34d399','#f59e0b','#ef4444','#22d3ee','#a78bfa'];
let CUR = { T: 298.15, gnd: 1.0 };

function el(tag, cls){
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

function mountUI(){
  const app = document.getElementById('app');
  const shell = el('div', 'flex h-screen');
  const sidebar = el('div', 'w-80 bg-slate-800 p-4 space-y-4 ring-1 ring-slate-700');
  const main = el('div', 'flex-1 p-4');
  const title = el('div', 'text-xl font-semibold');
  title.textContent = 'Controls';
  const tempLabel = el('label', 'block text-sm');
  tempLabel.textContent = 'Temperature (K)';
  const tempSlider = el('input', 'w-full');
  tempSlider.type = 'range';
  tempSlider.min = '278.15';
  tempSlider.max = '338.15';
  tempSlider.step = '0.1';
  tempSlider.value = '298.15';
  const tempVal = el('div', 'text-sm');
  tempVal.id = 'tempVal';
  tempVal.textContent = '298.15';
  const gndLabel = el('label', 'block text-sm');
  gndLabel.textContent = 'Denaturant (M)';
  const gndSlider = el('input', 'w-full');
  gndSlider.type = 'range';
  gndSlider.min = '0';
  gndSlider.max = '3';
  gndSlider.step = '0.1';
  gndSlider.value = '1';
  const gndVal = el('div', 'text-sm');
  gndVal.id = 'gndVal';
  gndVal.textContent = '1.0';
  const btns = el('div', 'pt-2');
  function mkBtn(name, view){
    const b = el('button', 'viewBtn w-full bg-slate-700 py-2 rounded mb-2');
    b.dataset.view = view;
    b.textContent = name;
    return b;
  }
  const b1 = mkBtn('Reference Images','gallery');
  const b2 = mkBtn(MODE === 'ml' ? 'KNN Performance' : (MODE === 'physics' ? 'Velocity Profile' : 'Arrhenius Plots'), MODE === 'ml' ? 'knn' : (MODE === 'physics' ? 'phys_vel' : 'arrhenius'));
  const b3 = mkBtn(MODE === 'ml' ? 'Confusion Matrix' : (MODE === 'physics' ? 'Wind Momentum' : 'DG Surface'), MODE === 'ml' ? 'cm' : (MODE === 'physics' ? 'phys_dmom' : 'surface'));
  const b4 = mkBtn(MODE === 'ml' ? '3D Scatter' : (MODE === 'physics' ? 'Beta Surface' : '3D DG Surface'), MODE === 'ml' ? 'ml3d' : (MODE === 'physics' ? 'phys_beta' : 'surface'));
  const b5 = mkBtn(MODE === 'ml' ? 'Metric Surface' : (MODE === 'physics' ? '—' : '—'), MODE === 'ml' ? 'mlMetric3d' : (MODE === 'physics' ? 'phys_beta' : 'surface'));
  btns.appendChild(b1); btns.appendChild(b2); btns.appendChild(b3);
  if (MODE === 'ml') { btns.appendChild(b4); btns.appendChild(b5); }
  if (MODE === 'physics') { btns.appendChild(b4); }
  sidebar.appendChild(title);
  sidebar.appendChild(tempLabel);
  sidebar.appendChild(tempSlider);
  sidebar.appendChild(tempVal);
  sidebar.appendChild(gndLabel);
  sidebar.appendChild(gndSlider);
  sidebar.appendChild(gndVal);
  sidebar.appendChild(btns);

  const gallery = el('div', 'view grid grid-cols-2 md:grid-cols-3 gap-4');
  gallery.id = 'gallery';
  const arrhenius = el('div', 'view hidden');
  arrhenius.id = 'arrhenius';
  const arrPlot1 = el('div', 'w-full h-[40vh]'); arrPlot1.id = 'arrPlot1';
  const arrPlot2 = el('div', 'w-full h-[40vh] mt-4'); arrPlot2.id = 'arrPlot2';
  arrhenius.appendChild(arrPlot1); arrhenius.appendChild(arrPlot2);
  const surface = el('div', 'view hidden');
  surface.id = 'surface';
  const surfacePlot = el('div', 'w-full h-[80vh]'); surfacePlot.id = 'surfacePlot';
  surface.appendChild(surfacePlot);
  const physVel = el('div', 'view hidden'); physVel.id = 'phys_vel';
  const physVelPlot = el('div', 'w-full h-[60vh]'); physVelPlot.id = 'physVelPlot';
  physVel.appendChild(physVelPlot);
  const physDmom = el('div', 'view hidden'); physDmom.id = 'phys_dmom';
  const physDmomPlot = el('div', 'w-full h-[60vh]'); physDmomPlot.id = 'physDmomPlot';
  physDmom.appendChild(physDmomPlot);
  const knn = el('div', 'view hidden'); knn.id = 'knn';
  const knPlot = el('div', 'w-full h-[60vh]'); knPlot.id = 'knPlot';
  knn.appendChild(knPlot);
  const cm = el('div', 'view hidden'); cm.id = 'cm';
  const cmPlot = el('div', 'w-full h-[60vh]'); cmPlot.id = 'cmPlot';
  cm.appendChild(cmPlot);
  const ml3d = el('div', 'view hidden'); ml3d.id = 'ml3d';
  const ml3dPlot = el('div', 'w-full h-[70vh]'); ml3dPlot.id = 'ml3dPlot';
  ml3d.appendChild(ml3dPlot);
  const mlMetric3d = el('div', 'view hidden'); mlMetric3d.id = 'mlMetric3d';
  const mlMetric3dPlot = el('div', 'w-full h-[70vh]'); mlMetric3dPlot.id = 'mlMetric3dPlot';
  mlMetric3d.appendChild(mlMetric3dPlot);
  const physBeta = el('div', 'view hidden'); physBeta.id = 'phys_beta';
  const physBetaPlot = el('div', 'w-full h-[70vh]'); physBetaPlot.id = 'physBetaPlot';
  physBeta.appendChild(physBetaPlot);
  main.appendChild(gallery);
  if (MODE === 'ml') { main.appendChild(knn); main.appendChild(cm); main.appendChild(ml3d); main.appendChild(mlMetric3d); } 
  else if (MODE === 'physics') { main.appendChild(physVel); main.appendChild(physDmom); main.appendChild(physBeta); }
  else { main.appendChild(arrhenius); main.appendChild(surface); }
  shell.appendChild(sidebar);
  shell.appendChild(main);
  app.appendChild(shell);

  tempSlider.addEventListener('input',e=>{ CUR.T = Number(e.target.value); tempVal.textContent = CUR.T.toFixed(2); updatePlots(); } );
  gndSlider.addEventListener('input',e=>{ CUR.gnd = Number(e.target.value); gndVal.textContent = CUR.gnd.toFixed(1); updatePlots(); } );
  document.querySelectorAll('.viewBtn').forEach(btn=>{btn.addEventListener('click',()=>setView(btn.dataset.view));});
}

function initGallery(){
  const g = document.getElementById('gallery');
  g.innerHTML = '';
  if (!PNGS || PNGS.length === 0){
    const empty = el('div','col-span-2 md:col-span-3 bg-slate-800 rounded p-6 text-center ring-1 ring-slate-700');
    const h = el('div','text-lg font-semibold'); h.textContent = 'No reference images found';
    const p = el('div','mt-2 text-sm text-slate-300'); p.textContent = 'Place PNG/JPG/SVG/WebP images in this paper folder to populate the gallery.';
    empty.appendChild(h); empty.appendChild(p); g.appendChild(empty);
    return;
  }
  PNGS.forEach(p=>{
    const card = el('div','bg-slate-800 rounded p-2 shadow-lg ring-1 ring-slate-700');
    const img = el('img','w-full h-auto rounded');
    img.src = p.path;
    img.alt = p.name;
    const cap = el('div','mt-2 text-xs');
    cap.textContent = p.name;
    card.appendChild(img);
    card.appendChild(cap);
    g.appendChild(card);
  });
}

 

function k_fold(T,gnd){
  const R = CONSTS.R||0.001987;
  const T0 = CONSTS.T0||298.15;
  const DH = CONSTS.DH_U2TS||20;
  const DS = CONSTS.DS_U2TS||0.06;
  const DCp = CONSTS.DCp_U2TS||-0.45;
  const m = CONSTS.m_U2TS||0.9;
  const Df = CONSTS.D_fold||1e7;
  const term = (DH - T*DS + DCp*(T - T0 - T*Math.log(T/T0)) - m*gnd)/(R*T);
  return Df*Math.exp(-term);
}

function dg_fold(T,gnd){
  const T0 = CONSTS.T0||298.15;
  const DH = CONSTS.DH||-50;
  const DS = CONSTS.DS||-0.15;
  const DCp = CONSTS.DCp||-1.15;
  const m = CONSTS.m||2.25;
  return DH - T*DS + DCp*(T - T0 - T*Math.log(T/T0)) - m*gnd;
}

function gnd_constant_stability(T){
  const T0 = CONSTS.T0||298.15;
  const DH = CONSTS.DH||-50;
  const DS = CONSTS.DS||-0.15;
  const DCp = CONSTS.DCp||-1.15;
  const m = CONSTS.m||2.25;
  const DGt = CONSTS.DG_target||-6.5;
  return (DGt - DH + T*DS - DCp*(T - T0 - T*Math.log(T/T0))) / m;
}

function drawArrhenius(){
  const Ts = Array.from({length:25},(_,i)=>278.15 + i*((338.15-278.15)/24));
  const gnds = [CUR.gnd];
  const X1 = Ts.map(T=>1/T);
  const Yall = gnds.flatMap(g=>Ts.map(T=>Math.log(k_fold(T,g))));
  const xMin = Math.min(...X1), xMax = Math.max(...X1);
  const yMin = Math.min(...Yall), yMax = Math.max(...Yall);
  const yPad = 0.05 * (yMax - yMin || 1);
  const traces1 = gnds.map(g=>{
    const ks = Ts.map(T=>Math.log(k_fold(T,g)));
    return {
      x: X1,
      y: ks,
      type:'scatter',
      mode:'lines+markers',
      name:`gnd=${g} M`,
      line: {width: 3, color: COLORS[g % COLORS.length]},
      marker: {size: 7, color: COLORS[g % COLORS.length]},
      hovertemplate: '1/T=%{x:.4f}<br>ln(k)=%{y:.3f}<br>gnd=' + g + ' M'
    };
  });
  const xSel = 1/CUR.T;
  const ySel = Math.log(k_fold(CUR.T, CUR.gnd));
  traces1.push({ x:[xSel], y:[ySel], type:'scatter', mode:'markers', name:'current', marker:{size:10, color:'#22d3ee', symbol:'diamond'} });
  Plotly.newPlot('arrPlot1', traces1, {
    title: {text:'Arrhenius Plot (Constant gnd)', font:{size:16}},
    xaxis:{title:'1/T (1/K)', gridcolor: THEME.gridcolor, linecolor: THEME.axis.linecolor, range:[xMin, xMax]},
    yaxis:{title:'ln(k_fold)', gridcolor: THEME.gridcolor, linecolor: THEME.axis.linecolor, range:[yMin - yPad, yMax + yPad]},
    template:'plotly_dark',
    paper_bgcolor: THEME.paper_bgcolor,
    plot_bgcolor: THEME.plot_bgcolor,
    font: THEME.font,
    legend:{orientation:'h', x:0, y:1.1},
    margin:{t:50, r:20, b:50, l:55}
  });
  const gndVar = Ts.map(T=>gnd_constant_stability(T));
  const ks2 = Ts.map((T,i)=>Math.log(k_fold(T,gndVar[i])));
  const y2Min = Math.min(...ks2), y2Max = Math.max(...ks2);
  const y2Pad = 0.05 * (y2Max - y2Min || 1);
  const traces2 = [{
    x: X1,
    y: ks2,
    type:'scatter',
    mode:'lines+markers',
    name:'Constant DG',
    line:{width:3, color:'#22d3ee'},
    marker:{size:7, color:'#22d3ee'},
    hovertemplate: '1/T=%{x:.4f}<br>ln(k)=%{y:.3f}'
  }];
  const ySel2 = Math.log(k_fold(CUR.T, gnd_constant_stability(CUR.T)));
  traces2.push({ x:[xSel], y:[ySel2], type:'scatter', mode:'markers', name:'current', marker:{size:10, color:'#ef4444', symbol:'diamond'} });
  Plotly.newPlot('arrPlot2', traces2, {
    title: {text:'Arrhenius Plot (Constant DG)', font:{size:16}},
    xaxis:{title:'1/T (1/K)', gridcolor: THEME.gridcolor, linecolor: THEME.axis.linecolor, range:[xMin, xMax]},
    yaxis:{title:'ln(k_fold)', gridcolor: THEME.gridcolor, linecolor: THEME.axis.linecolor, range:[y2Min - y2Pad, y2Max + y2Pad]},
    template:'plotly_dark',
    paper_bgcolor: THEME.paper_bgcolor,
    plot_bgcolor: THEME.plot_bgcolor,
    font: THEME.font,
    legend:{orientation:'h', x:0, y:1.1},
    margin:{t:50, r:20, b:50, l:55}
  });
}

function drawSurface(){
  const Tr = Array.from({length:30},(_,i)=>278.15 + i*((338.15-278.15)/29));
  const Gr = Array.from({length:31},(_,i)=>0 + i*(3/30));
  const Z = Tr.map(T=>Gr.map(g=>dg_fold(T,g)));
  const zSel = dg_fold(CUR.T, CUR.gnd);
  Plotly.newPlot('surfacePlot', [
    { z: Z, x:Tr, y:Gr, type:'surface', colorscale:'Viridis', showscale: true,
      lighting:{ambient:0.6, diffuse:0.8, specular:0.1, roughness:0.8}, contours: {z:{show:true, usecolormap:true, highlightcolor:'#ffffff', project:{z:true}}} },
    { x:[CUR.T], y:[CUR.gnd], z:[zSel], type:'scatter3d', mode:'markers', name:'current', marker:{size:4, color:'#ff7f0e'} }
  ], {
    title:{text:'DG vs T, gnd', font:{size:16}},
    scene:{
      xaxis:{title:'T (K)', gridcolor: THEME.gridcolor, backgroundcolor: THEME.plot_bgcolor},
      yaxis:{title:'gnd (M)', gridcolor: THEME.gridcolor, backgroundcolor: THEME.plot_bgcolor},
      zaxis:{title:'DG (kcal/mol)', gridcolor: THEME.gridcolor, backgroundcolor: THEME.plot_bgcolor}
    },
    template:'plotly_dark',
    paper_bgcolor: THEME.paper_bgcolor,
    font: THEME.font,
    margin:{t:50, r:20, b:40, l:40}
  });
  window.addEventListener('resize', ()=>{
    Plotly.Plots.resize(document.getElementById('arrPlot1'));
    Plotly.Plots.resize(document.getElementById('arrPlot2'));
    Plotly.Plots.resize(document.getElementById('surfacePlot'));
  });
}

function updatePlots(){
  if (MODE === 'bio'){ drawArrhenius(); drawSurface(); }
}

function drawPhysicsVelocity(){
  const Rstars = (RANGES['R_star_range'] && (RANGES['R_star_range'].type === 'array' ? RANGES['R_star_range'].values : (function(){ const s=RANGES['R_star_range']; if(!s) return [12,13.5,15]; const step=(s.end-s.start)/(s.num-1); const arr=[]; for(let i=0;i<s.num;i++) arr.push(s.start+i*step); return arr; })())) || [12,13.5,15];
  const betas = (RANGES['beta_model_values'] && (RANGES['beta_model_values'].type === 'array' ? RANGES['beta_model_values'].values : [0.8,1.0,1.5,2.0])) || [0.8,1.0,1.5,2.0];
  const Vinfs = (RANGES['V_inf_base_range'] && (RANGES['V_inf_base_range'].type === 'array' ? RANGES['V_inf_base_range'].values : (function(){ const s=RANGES['V_inf_base_range']; if(!s) return [2000,2200,2500]; const step=(s.end-s.start)/(s.num-1); const arr=[]; for(let i=0;i<s.num;i++) arr.push(s.start+i*step); return arr; })())) || [2000,2200,2500];
  const Rstar = Rstars[0];
  const Vinf = Vinfs[0];
  const r = Array.from({length:200}, (_,i)=> Rstar*(1.05 + i*(4.0/199)));
  function v(r, R_star, V_inf, beta){ return V_inf * Math.pow(1 - R_star/r, beta); }
  const traces = betas.map((b,idx)=>{
    const vv = r.map(rv=>v(rv, Rstar, Vinf, b));
    return {x:r, y:vv, type:'scatter', mode:'lines', name:`beta=${b}`, line:{width:3, color: COLORS[idx%COLORS.length]}};
  });
  Plotly.newPlot('physVelPlot', traces, {
    title:{text:'Wind Velocity vs Radius', font:{size:16}},
    xaxis:{title:'r (R_star units)'}, yaxis:{title:'v (km/s)'}, template:'plotly_dark', paper_bgcolor: THEME.paper_bgcolor, plot_bgcolor: THEME.plot_bgcolor, font: THEME.font,
    legend:{orientation:'h'}, margin:{t:50, r:20, b:50, l:55}
  });
}

function drawPhysicsDmom(){
  const Lspec = RANGES['L_star_range'];
  const Lstar = Lspec ? (function(){ if (Lspec.type === 'array') return Lspec.values; if (Lspec.type === 'linspace'){ const arr=[]; const step=(Lspec.end-Lspec.start)/(Lspec.num-1); for(let i=0;i<Lspec.num;i++) arr.push(Math.pow(10, Lspec.start + i*step)); return arr; } if (Lspec.type === 'logspace'){ const arr=[]; const step=(Lspec.end-Lspec.start)/(Lspec.num-1); for(let i=0;i<Lspec.num;i++) arr.push(Math.pow(10, Lspec.start + i*step)); return arr; } return null; })() : null;
  const Mspec = RANGES['M_dot_base_range'];
  const Mdot = Mspec ? (function(){ if (Mspec.type === 'array') return Mspec.values; if (Mspec.type === 'linspace'){ const arr=[]; const step=(Mspec.end-Mspec.start)/(Mspec.num-1); for(let i=0;i<Mspec.num;i++) arr.push(Math.pow(10, Mspec.start + i*step)); return arr; } if (Mspec.type === 'logspace'){ const arr=[]; const step=(Mspec.end-Mspec.start)/(Mspec.num-1); for(let i=0;i<Mspec.num;i++) arr.push(Math.pow(10, Mspec.start + i*step)); return arr; } return null; })() : null;
  const Vspec = RANGES['V_inf_base_range'];
  const Vinf = Vspec ? (function(){ if (Vspec.type === 'array') return Vspec.values; if (Vspec.type === 'linspace'){ const arr=[]; const step=(Vspec.end-Vspec.start)/(Vspec.num-1); for(let i=0;i<Vspec.num;i++) arr.push(Vspec.start + i*step); return arr; } if (Vspec.type === 'logspace'){ const arr=[]; const step=(Vspec.end-Vspec.start)/(Vspec.num-1); for(let i=0;i<Vspec.num;i++) arr.push(Math.pow(10, Vspec.start + i*step)); return arr; } return null; })() : null;
  const Rspec = RANGES['R_star_range'];
  const Rstar = Rspec ? (function(){ if (Rspec.type === 'array') return Rspec.values; if (Rspec.type === 'linspace'){ const arr=[]; const step=(Rspec.end-Rspec.start)/(Rspec.num-1); for(let i=0;i<Rspec.num;i++) arr.push(Rspec.start + i*step); return arr; } if (Rspec.type === 'logspace'){ const arr=[]; const step=(Rspec.end-Rspec.start)/(Rspec.num-1); for(let i=0;i<Rspec.num;i++) arr.push(Math.pow(10, Rspec.start + i*step)); return arr; } return null; })() : null;
  const Ls = Lstar || Array.from({length:30}, (_,i)=> Math.pow(10, 5.5 + i*(0.3/29)));
  const Ms = Mdot || Array.from({length:3}, (_,i)=> Math.pow(10, -7.5 + i*(1.0/2)));
  const Vs = Vinf || [2000, 2200, 2500];
  const Rs = Rstar || [12, 13.5, 15];
  function logDmom(Mdot, Vinf, Rstar){ return Math.log10(Mdot) + Math.log10(Vinf) + 0.5*Math.log10(Rstar); }
  const traces=[];
  for (let m=0;m<Ms.length;m++){
    for (let v=0; v<Vs.length; v++){
      for (let r=0;r<Rs.length;r++){
        const y = Ls.map(L=> logDmom(Ms[m], Vs[v], Rs[r]));
        traces.push({x: Ls.map(L=>Math.log10(L)), y, type:'scatter', mode:'lines', name:`Mdot=${Ms[m].toExponential(1)}, Vinf=${Vs[v]}, R=${Rs[r]}`, line:{width:2}});
      }
    }
  }
  Plotly.newPlot('physDmomPlot', traces, {
    title:{text:'log(D_mom) vs log(L_star)', font:{size:16}},
    xaxis:{title:'log(L_star)'}, yaxis:{title:'log(D_mom)'}, template:'plotly_dark', paper_bgcolor: THEME.paper_bgcolor, plot_bgcolor: THEME.plot_bgcolor, font: THEME.font, legend:{orientation:'h'}, margin:{t:50, r:20, b:50, l:55}
  });
}

// animation removed

function drawMLScatter3D(){
  const ds = getClassificationDataset();
  if (!ds){ Plotly.newPlot('ml3dPlot', [], {title:'No dataset found', template:'plotly_dark'}); return; }
  const keys = Object.keys(DATASETS['classification_data']).filter(k=>k!=='label');
  const f1 = keys[0], f2 = keys[1]||keys[0], f3 = keys[2]||keys[0];
  const X = ds.X;
  const xs = X.map(r=>r[keys.indexOf(f1)]);
  const ys = X.map(r=>r[keys.indexOf(f2)]);
  const zs = X.map(r=>r[keys.indexOf(f3)]);
  const labels = ds.y;
  const trace0 = {x:[], y:[], z:[], type:'scatter3d', mode:'markers', name:'0', marker:{size:3, color:'#34d399'}};
  const trace1 = {x:[], y:[], z:[], type:'scatter3d', mode:'markers', name:'1', marker:{size:3, color:'#ef4444'}};
  for (let i=0;i<labels.length;i++){ (labels[i]===0?trace0:trace1).x.push(xs[i]); (labels[i]===0?trace0:trace1).y.push(ys[i]); (labels[i]===0?trace0:trace1).z.push(zs[i]); }
  Plotly.newPlot('ml3dPlot', [trace0, trace1], {
    title:{text:'3D Feature Scatter', font:{size:16}},
    scene:{xaxis:{title:f1}, yaxis:{title:f2}, zaxis:{title:f3}}, template:'plotly_dark', paper_bgcolor: THEME.paper_bgcolor, font: THEME.font, margin:{t:50, r:20, b:40, l:40}
  });
  window.addEventListener('resize', ()=>{ Plotly.Plots.resize(document.getElementById('ml3dPlot')); });
}

function drawMLMetricSurface(){
  const ds = getClassificationDataset(); if (!ds){ return; }
  const ks = [3,5,7,9,11,13,15];
  const ratios = [0.6,0.65,0.7,0.75,0.8,0.85,0.9];
  const Z = ratios.map(r=>{
    const split = splitData(ds.X, ds.y, r, 42);
    const fit = scalerFit(split.X_train);
    const Xtr = scalerTransform(split.X_train, fit);
    const Xte = scalerTransform(split.X_test, fit);
    return ks.map(k=>{ const y_pred = knnPredict(Xtr, split.y_train, Xte, k); const m = metrics(split.y_test, y_pred); return m.acc; });
  });
  Plotly.newPlot('mlMetric3dPlot', [{z: Z, x: ks, y: ratios, type:'surface', colorscale:'Viridis'}], {
    title:{text:'Accuracy Surface (k, train_ratio)', font:{size:16}},
    scene:{xaxis:{title:'k'}, yaxis:{title:'train_ratio'}, zaxis:{title:'accuracy'}}, template:'plotly_dark', paper_bgcolor: THEME.paper_bgcolor, font: THEME.font, margin:{t:50, r:20, b:40, l:40}
  });
  window.addEventListener('resize', ()=>{ Plotly.Plots.resize(document.getElementById('mlMetric3dPlot')); });
}

function drawPhysicsBetaSurface(){
  function arr(spec, fallback){ if (!spec) return fallback; if (spec.type==='array') return spec.values; if (spec.type==='linspace'){ const step=(spec.end-spec.start)/(spec.num-1); const a=[]; for(let i=0;i<spec.num;i++) a.push(spec.start+i*step); return a; } return fallback; }
  const vrot = arr(RANGES['v_rot_values'], [0,50,100,150,200,250,300,350,400]);
  const incl = arr(RANGES['inclination_values'], [0,10,20,30,40,50,60,70,80,90]).map(d=>d*Math.PI/180);
  const rstars = arr(RANGES['R_star_range'], [12,13.5,15]);
  const mstars = arr(RANGES['M_star_range'], [30,35,40]);
  const vcrit = Math.sqrt((mstars[0])/(rstars[0]));
  const betaBaseSpec = RANGES['beta_intrinsic_range'];
  const betaBase = betaBaseSpec && betaBaseSpec.type==='linspace' ? (betaBaseSpec.start + betaBaseSpec.end)/2 : 0.9;
  const alphaSpec = RANGES['alpha_gravity_darkening_range'];
  const alpha = alphaSpec && alphaSpec.type==='linspace' ? (alphaSpec.start + alphaSpec.end)/2 : 0.5;
  const Z = incl.map(i=> vrot.map(v=> betaBase + 0.8*alpha*Math.pow(v/vcrit,2)*Math.cos(i)*Math.cos(i) ));
  Plotly.newPlot('physBetaPlot', [{z: Z, x: vrot, y: incl.map(a=>a*180/Math.PI), type:'surface', colorscale:'Turbo'}], {
    title:{text:'beta_obs Surface (v_rot, inclination)', font:{size:16}},
    scene:{xaxis:{title:'v_rot (km/s)'}, yaxis:{title:'inclination (deg)'}, zaxis:{title:'beta_obs'}}, template:'plotly_dark', paper_bgcolor: THEME.paper_bgcolor, font: THEME.font, margin:{t:50, r:20, b:40, l:40}
  });
  window.addEventListener('resize', ()=>{ Plotly.Plots.resize(document.getElementById('physBetaPlot')); });
}

function getClassificationDataset(){
  const ds = DATASETS['classification_data'];
  if (!ds) return null;
  const keys = Object.keys(ds);
  if (keys.length === 0) return null;
  const N = ds[keys[0]].length;
  const labelKey = keys.includes('label') ? 'label' : keys[keys.length - 1];
  const featureKeys = keys.filter(k => k !== labelKey);
  const X = Array.from({length:N}, (_,i)=>featureKeys.map(k=>Number(ds[k][i])));
  const y = Array.from({length:N}, (_,i)=>Number(ds[labelKey][i]));
  return {X, y};
}

function scalerFit(X){
  const n = X.length, d = X[0].length;
  const mean = Array(d).fill(0), std = Array(d).fill(0);
  for (let i=0;i<n;i++){
    for (let j=0;j<d;j++) mean[j] += X[i][j];
  }
  for (let j=0;j<d;j++) mean[j] /= n;
  for (let i=0;i<n;i++){
    for (let j=0;j<d;j++) std[j] += Math.pow(X[i][j]-mean[j],2);
  }
  for (let j=0;j<d;j++) std[j] = Math.sqrt(std[j]/n) || 1;
  return {mean, std};
}

function scalerTransform(X, fit){
  const n = X.length, d = X[0].length;
  const Y = Array.from({length:n}, ()=>Array(d).fill(0));
  for (let i=0;i<n;i++){
    for (let j=0;j<d;j++) Y[i][j] = (X[i][j] - fit.mean[j]) / fit.std[j];
  }
  return Y;
}

function mulberry32(a){
  return function(){
    var t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
}

function splitData(X, y, ratio, seed){
  const n = X.length;
  const idx = Array.from({length:n}, (_,i)=>i);
  const rand = mulberry32(seed|0);
  for (let i=n-1;i>0;i--){ const j = Math.floor(rand()*(i+1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const nTrain = Math.floor(ratio*n);
  const trainIdx = idx.slice(0,nTrain), testIdx = idx.slice(nTrain);
  const Xt = trainIdx.map(i=>X[i]), yt = trainIdx.map(i=>y[i]);
  const Xv = testIdx.map(i=>X[i]), yv = testIdx.map(i=>y[i]);
  return {X_train: Xt, y_train: yt, X_test: Xv, y_test: yv};
}

function knnPredict(X_train, y_train, X_test, k){
  function dist(a,b){ let s=0; for(let j=0;j<a.length;j++){ const d=a[j]-b[j]; s+= d*d; } return Math.sqrt(s); }
  return X_test.map(x=>{
    const arr = X_train.map((xi,i)=>({d: dist(x, xi), y: y_train[i]}));
    arr.sort((a,b)=>a.d-b.d);
    const neigh = arr.slice(0,k);
    const votes = {};
    for (const n of neigh){ const key = String(n.y); votes[key] = (votes[key]||0)+1; }
    const best = Object.entries(votes).sort((a,b)=>b[1]-a[1])[0][0];
    return Number(best);
  });
}

function metrics(y_true, y_pred){
  let tp=0, tn=0, fp=0, fn=0;
  for (let i=0;i<y_true.length;i++){
    const yt = y_true[i], yp = y_pred[i];
    if (yt===1 && yp===1) tp++; else if (yt===0 && yp===0) tn++; else if (yt===0 && yp===1) fp++; else if (yt===1 && yp===0) fn++;
  }
  const acc = (tp+tn)/(tp+tn+fp+fn);
  const prec = tp/(tp+fp||1);
  const rec = tp/(tp+fn||1);
  const f1 = 2*((prec*rec)/(prec+rec||1));
  return {acc, prec, rec, f1, tp, tn, fp, fn};
}

function drawKNN(){
  const ds = getClassificationDataset();
  if (!ds){ Plotly.newPlot('knPlot', [], {title:'No dataset found', template:'plotly_dark'}); return; }
  const split = splitData(ds.X, ds.y, 0.75, 42);
  const fit = scalerFit(split.X_train);
  const Xtr = scalerTransform(split.X_train, fit);
  const Xte = scalerTransform(split.X_test, fit);
  const ks = [3,5,7,9,11];
  const accs=[], precs=[], recs=[], f1s=[];
  for (const k of ks){
    const y_pred = knnPredict(Xtr, split.y_train, Xte, k);
    const m = metrics(split.y_test, y_pred);
    accs.push(m.acc); precs.push(m.prec); recs.push(m.rec); f1s.push(m.f1);
  }
  const traces = [
    {x: ks, y: accs, type:'scatter', mode:'lines+markers', name:'Accuracy', line:{width:3, color:'#60a5fa'}, marker:{size:7}},
    {x: ks, y: precs, type:'scatter', mode:'lines+markers', name:'Precision', line:{width:3, color:'#34d399'}, marker:{size:7}},
    {x: ks, y: recs, type:'scatter', mode:'lines+markers', name:'Recall', line:{width:3, color:'#f59e0b'}, marker:{size:7}},
    {x: ks, y: f1s, type:'scatter', mode:'lines+markers', name:'F1', line:{width:3, color:'#a78bfa'}, marker:{size:7}}
  ];
  const yMin = Math.min(...accs, ...precs, ...recs, ...f1s), yMax = Math.max(...accs, ...precs, ...recs, ...f1s);
  const yPad = 0.05 * (yMax - yMin || 1);
  Plotly.newPlot('knPlot', traces, {
    title:{text:'KNN Performance vs k', font:{size:16}},
    xaxis:{title:'k', range:[Math.min(...ks), Math.max(...ks)], dtick:2},
    yaxis:{title:'Score', range:[yMin - yPad, yMax + yPad]},
    template:'plotly_dark', paper_bgcolor: THEME.paper_bgcolor, plot_bgcolor: THEME.plot_bgcolor, font: THEME.font, legend:{orientation:'h'}, margin:{t:50, r:20, b:50, l:55}
  });
}

function drawCM(){
  const ds = getClassificationDataset(); if (!ds){ return; }
  const split = splitData(ds.X, ds.y, 0.75, 42);
  const fit = scalerFit(split.X_train);
  const Xtr = scalerTransform(split.X_train, fit);
  const Xte = scalerTransform(split.X_test, fit);
  const ks = [3,5,7,9,11];
  let bestK = ks[0], bestAcc = -1, bestM=null;
  for (const k of ks){
    const y_pred = knnPredict(Xtr, split.y_train, Xte, k);
    const m = metrics(split.y_test, y_pred);
    if (m.acc > bestAcc){ bestAcc = m.acc; bestK = k; bestM = m; }
  }
  const z = [[bestM.tn, bestM.fp],[bestM.fn, bestM.tp]];
  Plotly.newPlot('cmPlot', [{z, type:'heatmap', colorscale:'Blues'}], {
    title:{text:`Confusion Matrix (k=${bestK})`, font:{size:16}}, template:'plotly_dark', paper_bgcolor: THEME.paper_bgcolor, font: THEME.font,
    xaxis:{title:'Predicted', tickvals:[0,1], ticktext:['0','1']}, yaxis:{title:'Actual', tickvals:[0,1], ticktext:['0','1']}, margin:{t:50, r:20, b:50, l:55}
  });
  window.addEventListener('resize', ()=>{ Plotly.Plots.resize(document.getElementById('knPlot')); Plotly.Plots.resize(document.getElementById('cmPlot')); });
}

function setView(v){
  document.querySelectorAll('.view').forEach(el=>el.classList.add('hidden'));
  document.getElementById(v).classList.remove('hidden');
}

function init(){
  mountUI();
  initGallery();
  if (MODE === 'ml'){ drawKNN(); drawCM(); drawMLScatter3D(); drawMLMetricSurface(); } else if (MODE === 'physics'){ drawPhysicsVelocity(); drawPhysicsDmom(); drawPhysicsBetaSurface(); } else { drawArrhenius(); drawSurface(); }
  setView('gallery');
}

window.addEventListener('DOMContentLoaded', init);
"""
    return head + body


def generate_html_shell_local() -> str:
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\"/>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
<title>Interactive Simulation</title>
<script src=\"https://cdn.tailwindcss.com\"></script>
<script src=\"https://cdn.plot.ly/plotly-2.30.0.min.js\"></script>
</head>
<body class=\"bg-slate-900 text-slate-100\">
<div id=\"app\"></div>
<script src=\"interactive_simulation.js\"></script>
</body>
</html>"""


def main():
    """Main function to run the Interactive3DSimulationAgent"""
    parser = argparse.ArgumentParser(
        description="Generates toggleable interactive simulations from research paper outputs"
    )
    parser.add_argument(
        "--paper-dir",
        type=str,
        required=True,
        help='Path to paper directory (e.g., "papers/bio paper/" or absolute path). Use quotes for paths with spaces.'
    )
    parser.add_argument(
        "--plan",
        type=str,
        required=False,
        help='Optional path to 3D simulation plan JSON; defaults to 3d_simulation_plan.json in paper-dir'
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Generate simulations using Gemini instead of local renderer"
    )
    
    args = parser.parse_args()
    
    # Resolve paper directory path
    paper_dir = Path(args.paper_dir)
    
    if not paper_dir.is_absolute():
        project_root = Path(__file__).parent.parent
        paper_dir = project_root / paper_dir
    
    paper_dir = paper_dir.resolve()
    
    if not paper_dir.exists():
        print(f"Error: Paper directory not found: {paper_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not paper_dir.is_dir():
        print(f"Error: Path is not a directory: {paper_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        print("Collecting paper artifacts...", file=sys.stderr)
        artifacts = collect_paper_artifacts(paper_dir)

        if args.plan:
            plan_path = Path(args.plan)
            if not plan_path.is_absolute():
                plan_path = (paper_dir / plan_path).resolve()
            if plan_path.exists():
                with open(plan_path, 'r') as f:
                    artifacts["simulation_3d_plan"] = json.load(f)
        
        print(f"Found {len(artifacts['png_images'])} PNG images", file=sys.stderr)
        
        use_local = not args.use_llm
        html_code = None
        if use_local:
            print("Generating interactive HTML (local renderer)...", file=sys.stderr)
            html_code = generate_interactive_html_local(artifacts)
            js_code = generate_js_bundle_local(artifacts)
        else:
            print("Identifying potential simulations...", file=sys.stderr)
            simulations = identify_potential_simulations(artifacts)
            print(f"Identified {len(simulations)} potential simulations", file=sys.stderr)
            for sim in simulations:
                print(f"  - {sim.get('title', 'Unknown')}: {sim.get('description', '')}", file=sys.stderr)
            print("Generating interactive HTML (LLM)...", file=sys.stderr)
            html_code = generate_interactive_html(artifacts, simulations)
        
        # Save HTML file
        output_path = paper_dir / "interactive_simulation.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(generate_html_shell_local())
        js_out = paper_dir / "interactive_simulation.js"
        with open(js_out, 'w', encoding='utf-8') as jf:
            jf.write(js_code)
        
        print(f"✅ Interactive simulation saved to: {output_path}", file=sys.stderr)
        print("📊 Includes JS-based Arrhenius and 3D DG views", file=sys.stderr)
        sim_count = 2
        print(f"🌐 Open {output_path} in a web browser to view", file=sys.stderr)
        
        print(json.dumps({
            "status": "success",
            "output_file": str(output_path),
            "simulations": sim_count,
            "message": "Interactive simulation generated successfully"
        }, indent=2))
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
