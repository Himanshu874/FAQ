import os
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

load_dotenv()


class CSVRAGSystem:
    def __init__(self, csv_path, persist_directory="./chroma_db"):
        self.csv_path = csv_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        
    def load_csv_data(self):
        """Load and process CSV file into text documents"""
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"Successfully loaded CSV with {encoding} encoding")
                break
            except (UnicodeDecodeError, Exception) as e:
                continue
        
        if df is None:
            raise ValueError("Could not read CSV file with any supported encoding")
        
        # Convert each row to a text document
        documents = []
        for idx, row in df.iterrows():
            # Create a text representation of the row
            text = " | ".join([f"{col}: {str(row[col])}" for col in df.columns])
            documents.append(text)
        
        print(f"Loaded {len(documents)} documents from CSV")
        return documents
    
    def create_vectorstore(self):
        """Create vector embeddings and store in ChromaDB"""
        print("Loading CSV data...")
        documents = self.load_csv_data()
        
        # Split documents if they're too large
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.create_documents(documents)
        
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = OpenAIEmbeddings()
        
        # Create and persist ChromaDB vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        
        print("Vector store created successfully!")
        
    def load_existing_vectorstore(self):
        """Load existing ChromaDB vectorstore"""
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        print("Loaded existing vector store!")
        
    def setup_qa_chain(self, k=5):
        """Setup the QA retrieval chain"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore() first.")
        
        llm = OpenAI(temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            ),
            return_source_documents=True
        )
        
    def ask_question(self, question):
        """Ask a question and get an answer"""
        if self.qa_chain is None:
            self.setup_qa_chain()
        
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }


def main():
    # Example usage
    csv_file = "data.csv"  # Replace with your CSV file path
    
    # Initialize the RAG system
    rag_system = CSVRAGSystem(csv_file)
    
    # Create vector store (do this once)
    if not os.path.exists("./chroma_db"):
        rag_system.create_vectorstore()
    else:
        rag_system.load_existing_vectorstore()
    
    # Setup QA chain
    rag_system.setup_qa_chain()
    
    # Interactive Q&A loop
    print("\n=== CSV RAG QA System ===")
    print("Ask questions about your CSV data (type 'quit' to exit)\n")
    
    while True:
        question = input("Question: ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = rag_system.ask_question(question)
        print(f"\nAnswer: {result['answer']}\n")
        print(f"Sources: {len(result['source_documents'])} documents retrieved\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
