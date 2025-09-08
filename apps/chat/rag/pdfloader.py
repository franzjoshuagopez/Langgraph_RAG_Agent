import os
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from config.logger import get_logger

logger = get_logger(__name__)

#create paths for pdf source and chromadb source folders
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


#create variables here for pdf directories
LISTING_PDF_DIR = BASE_DIR / "media" / "pdfs" / "market_listing"
MARKET_PDF_DIR = BASE_DIR / "media" / "pdfs" / "stock_market"

#create variables here for chromaDB directories
CHROMA_LISTING_DIR = BASE_DIR / "chromadb" / "listing_db"
CHROMA_MARKET_DIR = BASE_DIR / "chromadb" / "market_db"


def build_vectorstore(pdf_dir : Path, chroma_dir: Path):
    """
        This is a helper function to build a vector store from PDFs
    """

    pdf_dir.mkdir(parents=True, exist_ok=True) #checks if pdf directory exists if not create

    chroma_dir.mkdir(parents=True, exist_ok=True) #checks if chromadb directory exists if not create

    #load all pdfs in directory
    loader = PyPDFDirectoryLoader(str(pdf_dir))

    documents = loader.load()

    if not documents:
        logger.warning("No documents found in media/pdfs")
    
    #split documents    
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
        persist_directory=str(chroma_dir)
    )

    return vectorstore.as_retriever()



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

    listing_retriever = build_vectorstore(LISTING_PDF_DIR, CHROMA_LISTING_DIR)
    market_retriever = build_vectorstore(MARKET_PDF_DIR, CHROMA_MARKET_DIR)

    logger.info("Retriever initialization complete")

    return listing_retriever, market_retriever