from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "all-MiniLM-L6-v2"

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return Chroma.from_documents(chunks, embeddings)
