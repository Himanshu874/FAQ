"""
Script to reset and recreate the vector store with proper encoding
"""
import shutil
import os

# Remove existing vector store
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("✓ Removed old vector store")

print("\nNow run: python3 rag_qa_system.py")
print("This will create a fresh vector store with proper encoding")
