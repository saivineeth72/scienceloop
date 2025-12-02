# ScienceLoop - AI4Science Project

An AI4Science project using SpoonOS for understanding scientific papers.

## Project Structure

```
ScienceLoop/
├── agents/
│   ├── PaperUnderstandingAgent.py  # Step 1: Analyzes PDFs and extracts structured data
│   └── KnowledgeGraphAgent.py      # Step 2: Builds knowledge graph from Step 1 output
├── tools/
│   └── pdf_reader_tool.py          # SpoonOS PDF reader tool (BaseTool)
├── papers/                          # Place PDF files here
├── spoon.json                       # SpoonOS configuration
└── requirements.txt                 # Python dependencies
```

## Setup

1. Create and activate virtual environment:
```bash
# macOS/Linux
python3 -m venv spoon-env
source spoon-env/bin/activate

# Windows (PowerShell)
python -m venv spoon-env
.\spoon-env\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure LLM integration (choose one option):

   **Option 1: Using .env file (Recommended)**
   
   Create a `.env` file in the project root:
   ```bash
   # Create .env file
   cat > .env << EOF
   OPENAI_API_KEY=your-openai-api-key-here
   OPENAI_MODEL=gpt-4
   EOF
   ```
   
   Or manually create `.env` with:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   OPENAI_MODEL=gpt-4
   ```

   **Option 2: Environment Variables**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export OPENAI_MODEL="gpt-4"  # optional, defaults to gpt-4
   ```

   **Option 2: Custom LLM API**
   ```bash
   export LLM_API_URL="https://your-llm-api-endpoint.com/v1/complete"
   export LLM_API_KEY="your-api-key-here"  # optional
   ```

   **Option 3: SpoonOS LLM Module**
   ```bash
   pip install spoonos
   ```

4. Place your PDF files in the `papers/` directory.

## Usage

**Important:** Make sure the virtual environment is activated before running:

```bash
source spoon-env/bin/activate  # macOS/Linux
# or
.\spoon-env\Scripts\Activate.ps1  # Windows PowerShell
```

## Step 1: Paper Understanding

Run the PaperUnderstandingAgent with a PDF file:

```bash
# Using SpoonOS CLI
spoon run PaperUnderstandingAgent --pdf papers/sample.pdf

# Or run directly (direct mode - default)
python agents/PaperUnderstandingAgent.py --pdf papers/sample.pdf

# Or use agent mode with full tool integration
python agents/PaperUnderstandingAgent.py --pdf papers/sample.pdf --use-agent
```

The agent will:
1. Read the raw PDF text (no extraction or parsing)
2. Send the entire text to the LLM
3. The LLM will understand formulas, variables, relationships, and key ideas
4. Output structured JSON with the analysis

## Step 2: Knowledge Graph Building

Run the KnowledgeGraphAgent with Step 1 output:

```bash
# From a JSON file
python agents/KnowledgeGraphAgent.py --input step1_output.json

# Or pipe from Step 1
python agents/PaperUnderstandingAgent.py --pdf papers/sample.pdf 2>/dev/null | \
  python agents/KnowledgeGraphAgent.py --stdin

# Using SpoonOS CLI
spoon run KnowledgeGraphAgent --input step1_output.json
```

The agent will:
1. Take JSON output from Step 1 (summary, formulas, relationships, variables, key_ideas)
2. Extract scientific entities (variables, concepts, processes, conditions)
3. Identify directional relationships between entities
4. Output a knowledge graph with nodes and edges

## Output Formats

### Step 1 Output (PaperUnderstandingAgent)

```json
{
  "summary": "Brief scientific summary",
  "formulas": ["formula 1", "formula 2"],
  "relationships": ["relationship 1", "relationship 2"],
  "variables": ["variable 1: description", "variable 2: description"],
  "key_ideas": ["idea 1", "idea 2"]
}
```

### Step 2 Output (KnowledgeGraphAgent)

```json
{
  "nodes": ["entity1", "entity2", "entity3"],
  "edges": [
    {
      "source": "entity1",
      "relation": "relationship description",
      "target": "entity2"
    }
  ]
}
```

## Complete Workflow Example

```bash
# Step 1: Extract structured data from PDF
python agents/PaperUnderstandingAgent.py --pdf papers/sample.pdf > step1.json 2>/dev/null

# Step 2: Build knowledge graph
python agents/KnowledgeGraphAgent.py --input step1.json > knowledge_graph.json
```

Or as a pipeline:

```bash
python agents/PaperUnderstandingAgent.py --pdf papers/sample.pdf 2>/dev/null | \
  python agents/KnowledgeGraphAgent.py --stdin
```

## Notes

- **Step 1 (PaperUnderstandingAgent)**:
  - Uses **SpoonOS tools** (`PDFReaderTool`) for PDF reading
  - Raw PDF text extraction - no manual parsing or extraction
  - All understanding is done by the LLM (formulas, variables, relationships are identified by the LLM)
  - Two modes available:
    - **Direct mode** (default): Uses PDF tool directly, then sends to LLM
    - **Agent mode** (`--use-agent`): Uses full SpoonOS agent with tool calling capabilities

- **Step 2 (KnowledgeGraphAgent)**:
  - Takes Step 1 JSON output as input
  - Extracts entities and relationships using LLM
  - Builds a clean knowledge graph with nodes and edges
  - Does not invent concepts not present in Step 1 output

- Python 3.7+ is required
- Both agents can be run directly or via SpoonOS CLI

