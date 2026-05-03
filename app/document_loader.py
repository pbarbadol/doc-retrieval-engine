import pathlib
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_docs(path: str):
    file_paths = pathlib.Path(path).glob("*")

    documents = []

    for file_path in file_paths:
        if file_path.suffix.lower() == ".pdf":
            loaderPdf = PyMuPDFLoader(str(file_path))
            documents.extend(loaderPdf.load())
        elif file_path.suffix.lower() == ".txt":
            loaderTxt = TextLoader(str(file_path))
            documents.extend(loaderTxt.load())
        else:
            continue

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400
    )

    return splitter.split_documents(documents)