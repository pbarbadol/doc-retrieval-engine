from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
Always respond in the same language as the user's question.
You are a helpful AI assistant.
Answer the question ONLY using the provided context.
if the answer is not in the context, say "I have no answer for that question.
"""

def create_rag_chain(vector_db: Chroma):
    llm = OllamaLLM(model="mistral",
                 temperature=0.0)
    

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ]
    )

    def rag_answer(question: str):

        results: tuple[list, list] = vector_db.similarity_search_with_score(question, k=5)
        docs: list = [doc for doc, score in results]
        scores: list = [score for doc, score in results]
        context = "\n\n".join([doc.page_content for doc in docs])
        
        
        messages = prompt.format_messages(
            context=context,
            question=question
        )

        source_set: set = {doc.metadata.get("source", "Unknown") for doc in docs}
        source = "\n".join([f"Source: {f}" for f in source_set])
        print(f"Retrieved {len(docs)} documents with scores: {[s for s in scores]}")

        if(scores[0] > 1.4):
            return "I have no answer for that question.", source
        
        return llm.invoke(messages), source
    
    return rag_answer
