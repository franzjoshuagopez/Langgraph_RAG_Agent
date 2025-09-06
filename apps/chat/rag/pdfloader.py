import os
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from config.logger import get_logger

logger = get_logger(__name__)

#create paths for pdf source and chromadb source folders
PDF_DIR = Path(__file__).resolve().parent.parent.parent.parent / "media" / "pdfs"
CHROMA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "chromadb"


def init_retriever():
    """
        initialzes the retriever:
        -loading pdfs
        -splitting into chunks
        -creating embeddings
        -building ChromaDB vector store
        
        returns:
            retriever: Vectorstore retriever
    """
    logger.info("Initializing retriever...")

    PDF_DIR.mkdir(parents=True, exist_ok=True) #checks if pdf directory exists if not create
    CHROMA_DIR.mkdir(parents=True, exist_ok=True) #checks if chromadb directory exists if not create

    #read and load pdfs
    pdfloader = PyPDFDirectoryLoader(str(PDF_DIR))
    documents = pdfloader.load()

    if not documents:
        logger.warning("No documents found in media/pdfs")
    
    #split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splitted_docs = splitter.split_documents(documents)

    #create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    #create ChromaDB vectorstore
    vectorstore = Chroma.from_documents(
        documents=splitted_docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    #create retriever instance
    retriever = vectorstore.as_retriever()
    logger.info("Retriever initialization complete")

    return retriever