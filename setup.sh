#!/bin/bash

# Update pip
python -m pip install --upgrade pip

# Install pysqlite3-binary first
pip install pysqlite3-binary

# Install other requirements
pip install "pydantic>=2.0.0,<3.0.0" 
pip install "chromadb>=0.4.17"
pip install "langchain>=0.0.339"
pip install "openai>=1.3.5"
pip install streamlit pandas duckdb python-docx docx2txt requests python-dotenv sentence-transformers tiktoken numpy

# Run the SQLite fix
python sqlite_fix.py

echo "Setup completed!"