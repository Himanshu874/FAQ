# RAG-based QA System for CSV Files

A Retrieval-Augmented Generation (RAG) system that allows you to ask questions about CSV data using natural language.

## Tech Stack

- **Python**: Core programming language
- **ChromaDB**: Vector database for storing embeddings
- **LangChain**: Framework for building LLM applications
- **OpenAI Embeddings**: For creating vector embeddings

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. Prepare your CSV file (e.g., `data.csv`)

## Usage

### Option 1: Interactive CLI
```bash
python3 rag_qa_system.py
```
Note: Update the `csv_file` variable in `main()` to point to your CSV file.

### Option 2: Run Example Script
```bash
python3 example_usage.py
```

### Option 3: Use in Your Code
```python
from rag_qa_system import CSVRAGSystem
import os

# Initialize with your CSV file
rag_system = CSVRAGSystem("data.csv")

# Create or load vector store
if not os.path.exists("./chroma_db"):
    rag_system.create_vectorstore()
else:
    rag_system.load_existing_vectorstore()

# Setup QA chain
rag_system.setup_qa_chain()

# Ask questions
result = rag_system.ask_question("Your question here?")
print(result["answer"])
```

## Reset Vector Store

If you need to recreate the vector store (e.g., after updating your CSV):
```bash
python3 reset_vectorstore.py
```

## How It Works

1. **Data Loading**: Reads CSV file and converts each row into text
2. **Embedding**: Creates vector embeddings using OpenAI
3. **Storage**: Stores embeddings in ChromaDB for efficient retrieval
4. **Retrieval**: Finds relevant data based on your question
5. **Generation**: Uses LLM to generate natural language answers

## Features

- Persistent vector storage (no need to re-embed on restart)
- Interactive Q&A interface
- Source document tracking
- Configurable chunk sizes and retrieval parameters
