#!/bin/bash
# ScienceLoop Workflow: PDF → Knowledge Graph
# Complete workflow from PDF to knowledge graph

# Make sure virtual environment is activated
source spoon-env/bin/activate

# Check if PDF path is provided
if [ -z "$1" ]; then
    echo "Usage: ./workflow.sh papers/your-paper.pdf"
    exit 1
fi

PDF_PATH="$1"

# Get the directory of the PDF file
PDF_DIR=$(dirname "$PDF_PATH")
PDF_BASENAME=$(basename "$PDF_PATH")

echo "=== Step 1: Paper Understanding ===" >&2
echo "Analyzing PDF: $PDF_PATH" >&2

# Step 1: Extract structured data from PDF (saves to understand.json in PDF's directory)
python agents/PaperUnderstandingAgent.py --pdf "$PDF_PATH" >/dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Error: Step 1 failed" >&2
    exit 1
fi

UNDERSTAND_JSON="$PDF_DIR/understand.json"
echo "Step 1 complete. Output saved to $UNDERSTAND_JSON" >&2
echo "" >&2

echo "=== Step 2: Knowledge Graph Building ===" >&2
echo "Building knowledge graph from understand.json..." >&2

# Step 2: Build knowledge graph (saves to knowledge_graph.json in PDF's directory)
python agents/KnowledgeGraphAgent.py --input "$UNDERSTAND_JSON" >/dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Error: Step 2 failed" >&2
    exit 1
fi

KG_JSON="$PDF_DIR/knowledge_graph.json"
echo "" >&2
echo "=== Complete! ===" >&2
echo "Files saved in: $PDF_DIR" >&2
echo "  - understand.json" >&2
echo "  - knowledge_graph.json" >&2

