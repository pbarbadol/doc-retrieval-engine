from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You are a helpful AI assistant.
Anwer the question ONLY using the provided context.
if the answer is not in the context, say "I have no answer for that question.
"""

def create_rag_chain(vector_db):
    llm = OllamaLLM(model="mistral",
                 temperature=0.0)
    

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ]
    )

    def rag_answer(question: str):
        docs = vector_db.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        #source
        
        
        messages = prompt.format_messages(
            context=context,
            question=question
        )

        source_set = {doc.metadata.get("source", "Unknown") for doc in docs}
        source = "\n".join([f"Source: {f}" for f in source_set])

        return llm.invoke(messages), source
    
    return rag_answer
