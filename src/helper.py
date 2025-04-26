import os
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document

# === Load environment variables ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_pdf_text(pdf_docs):
    """
    Extracts and concatenates text from a list of PDF files.

    Args:
        pdf_docs (List[BinaryIO]): List of PDF files.

    Returns:
        str: Combined text from all PDF pages.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text

def get_text_chunks(text, chunk_size=1000, chunk_overlap=20):
    """
    Splits large text into smaller chunks.

    Args:
        text (str): The text to split.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]  # Avoid empty ones

def get_vector_store(text_chunks, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Converts text chunks into a FAISS vector store.

    Args:
        text_chunks (List[str]): List of text chunks.
        embedding_model (str): HuggingFace embedding model to use.

    Returns:
        FAISS: Vector store object.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def get_conversational_chain(vector_store, llm_model="gemini-1.5-pro", temperature=0.2):
    """
    Creates a conversational retrieval chain using a vector store.

    Args:
        vector_store (FAISS): The FAISS vector store.
        llm_model (str): Name of the Gemini model to use.
        temperature (float): LLM creativity level.

    Returns:
        ConversationalRetrievalChain: The conversation chain.
    """
    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return chain
