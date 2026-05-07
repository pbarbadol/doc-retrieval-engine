from hybrid_retriever import HybridRetriever
from document_loader import load_and_split_docs
from vector_store import create_vector_store
from rag_chain import create_rag_chain
from dotenv import load_dotenv

load_dotenv()

def main():
    # Load and split documents
    print(" Loading and splitting documents...")
    chunks = load_and_split_docs("data/")

    # Create vector store
    print(" Creating vector store...")
    vector_db = create_vector_store(chunks)

    print(" Creating retriever...")
    retriever = HybridRetriever(chunks, vector_db)

    # Create RAG chain
    print(" Creating RAG chain...")
    rag_answer = create_rag_chain(retriever)

    print("\nRAG system is ready! Ask your questions about the company policy.\n")

    try:
        while True:
            question = input("Question: ")
            if question.lower() in ["exit", "quit"]:
                print("Exiting RAG system. Goodbye!")
                break
            
            answer, source = rag_answer(question)
            print(f"Answer: {answer}\n")
            print(f"{source}\n")
    except KeyboardInterrupt:
        print("\nExiting RAG system. Goodbye!")
        
if __name__ == "__main__":
    main()