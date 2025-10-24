"""
Example usage of the CSV RAG QA System
"""
from rag_qa_system import CSVRAGSystem
import os

# Initialize with your CSV file
csv_file = "data.csv"
rag_system = CSVRAGSystem(csv_file)

# Create or load vector store
if not os.path.exists("./chroma_db"):
    print("Creating new vector store...")
    rag_system.create_vectorstore()
else:
    print("Loading existing vector store...")
    rag_system.load_existing_vectorstore()

# Setup QA chain
rag_system.setup_qa_chain(k=5)

# Example questions
questions = [
    "Do you have a refund policy?",
    "Can I take the bootcamp if I have no programming experience?",
    "What is the duration of the bootcamp?",
    "Do you provide job assistance?",
    "What tools will I learn in this bootcamp?"
]

print("\n" + "="*60)
print("CSV RAG QA System - Example Questions")
print("="*60 + "\n")

for question in questions:
    print(f"Q: {question}")
    result = rag_system.ask_question(question)
    print(f"A: {result['answer']}")
    print(f"   (Retrieved {len(result['source_documents'])} relevant documents)")
    print("-" * 60 + "\n")
