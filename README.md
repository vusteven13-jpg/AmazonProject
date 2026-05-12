# Dawson Data — Executive Business Intelligence AI Agent

## Overview

Dawson Data is an AI-powered executive business intelligence agent for Amazon sales and customer review analysis. It combines structured data analytics with semantic search over internal reports and customer feedback.

## System Architecture

Dawson Data uses a Retrieval-Augmented Generation (RAG) architecture combined with a LangGraph-based reasoning workflow.

Workflow:
1. User submits a natural language query  
2. Query is classified into structured, unstructured, or hybrid intent  
3. Structured data analysis is performed using Pandas  
4. Relevant documents are retrieved from a ChromaDB vector database  
5. Claude LLM synthesizes insights using both data sources  
6. Output is validated and formatted for business decision-making  

## Project Files

- `dawson_data_agent.py` — main agent script
- `README.md` — project documentation
- `data/` — input CSV datasets
- `chroma_db/` — persistent vector store
- `.venv-2/` — suggested virtual environment directory

## Installation

1. Open the project folder:
```bash
cd /Users/user/AmazonProject
```

2. Create a Python environment:
```bash
python3 -m venv .venv-2
source .venv-2/bin/activate
```

3. Install dependencies:
```bash
pip install langgraph langchain langchain-anthropic langchain-community \
  langchain-chroma chromadb sentence-transformers gradio pandas numpy
```

4. Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

Run the main script:
```bash
python dawson_data_agent.py
```

Open the Gradio UI at:
`http://127.0.0.1:7860`

## Data

Put your dataset files in the `data/` folder:
- `data/amazon_sales.csv`
- `data/amazon_reviews.csv`
- `data/amazon_sales_2025.csv`

If files are missing, the script may use synthetic demo data if that is implemented.

## Example Queries

- What are the top 5 revenue-generating categories?
- Which categories are underperforming and why?
- What is our year-over-year growth in 2025?
- What are customers saying about the Clothing category?

## Dataset Download

Large datasets are not included due to GitHub size limits.

Download them here:
https://drive.google.com/drive/folders/1kAL3AwHoR5rUl_AAlPQ7sR1xhPZhG0dp?usp=drive_link

Place the files into the project folder before running the code.