# VectorstoreAPI.py

import os
import uuid
from typing import Any, Optional, List, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, PrivateAttr, validator

from cerebras.cloud.sdk import Cerebras
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.document_loaders import PyPDFLoader  # Use LangChain's built-in loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, BaseRetriever

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store vector stores
VECTOR_STORE_DIR = "vector_stores"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Initialize Cerebras client
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-e2e8kypw838rwmpjxd9nx2vn5jrertm339fnrcnt9c6p8hmx")
client = Cerebras(api_key=CEREBRAS_API_KEY)

# Define Pydantic Models
class QueryRequest(BaseModel):
    vector_store_ids: List[str]
    question: str
    chat_history: Optional[List[List[str]]] = []

    @validator('chat_history', pre=True, always=True)
    def convert_chat_history(cls, v):
        if not v:
            return []
        return [tuple(pair) for pair in v]

class UploadResponse(BaseModel):
    message: str
    vector_store_id: str
    persist_directory: str

class QueryResponse(BaseModel):
    answer: str

# Define the custom Cloud LLM
class CloudLLM(LLM):
    _client: Any = PrivateAttr()
    _model_name: str = PrivateAttr()

    def __init__(self, client: Any, model_name: str):
        super().__init__()
        self._client = client
        self._model_name = model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._model_name,
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"model_name": self._model_name}

    @property
    def _llm_type(self):
        return "cloud_llm"

# Define the MultiVectorRetriever class within the same file
class MultiVectorRetriever(BaseRetriever):
    _retrievers: List[BaseRetriever] = PrivateAttr()

    def __init__(self, retrievers: List[BaseRetriever]):
        super().__init__()
        self._retrievers = retrievers

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        all_docs = []
        for retriever in self._retrievers:
            docs = retriever.get_relevant_documents(query, **kwargs)
            all_docs.extend(docs)
        # Optional: Implement ranking or deduplication here
        return all_docs

    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        all_docs = []
        for retriever in self._retrievers:
            docs = await retriever.aget_relevant_documents(query, **kwargs)
            all_docs.extend(docs)
        # Optional: Implement ranking or deduplication here
        return all_docs

# Function to convert PDF to vector store
def convert_pdf_to_vector_store(pdf_path: str, persist_directory: str) -> str:
    """
    Converts a PDF to a vector store and persists it to disk.

    Args:
        pdf_path (str): Path to the PDF file.
        persist_directory (str): Directory to save the vector store.

    Returns:
        str: Path to the persisted vector store.
    """
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        # Create embeddings using Ollama
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory  # Specify directory here
        )

        # Persist to disk
        vectorstore.persist()

        return persist_directory
    except Exception as e:
        raise e

# Function to load an existing vector store
def load_vector_store(persist_directory: str, embeddings: OllamaEmbeddings) -> Chroma:
    """
    Loads an existing vector store from the specified directory.

    Args:
        persist_directory (str): Directory where the vector store is persisted.
        embeddings (OllamaEmbeddings): The embedding function.

    Returns:
        Chroma: The loaded vector store.
    """
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        # Removed the load() call as Chroma doesn't have a load method
        return vectorstore
    except Exception as e:
        raise e

# Function to create QA chain from multiple vector stores
def create_qa_chain(selected_vector_store_dirs: List[str], embeddings: OllamaEmbeddings, cloud_llm: CloudLLM) -> ConversationalRetrievalChain:
    """
    Creates a ConversationalRetrievalChain from multiple vector stores.

    Args:
        selected_vector_store_dirs (List[str]): List of directories for the selected vector stores.
        embeddings (OllamaEmbeddings): The embedding function.
        cloud_llm (CloudLLM): The custom LLM.

    Returns:
        ConversationalRetrievalChain: The QA chain.
    """
    retrievers = []
    for dir_path in selected_vector_store_dirs:
        vectorstore = load_vector_store(dir_path, embeddings)
        retrievers.append(vectorstore.as_retriever())

    # Combine retrievers
    combined_retriever = MultiVectorRetriever(retrievers)

    # Create QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=cloud_llm,
        retriever=combined_retriever
    )
    return qa_chain

# Endpoint to upload PDF and create vector store
@app.post("/upload_pdf/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), name: Optional[str] = None):
    """
    Endpoint to upload a PDF file and convert it to a vector store.

    Args:
        file (UploadFile): The PDF file to upload.
        name (Optional[str]): Optional name for the vector store.

    Returns:
        UploadResponse: Confirmation with the vector store path.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are supported.")

    try:
        # Generate a unique identifier for the vector store
        vector_store_id = name if name else str(uuid.uuid4())
        persist_directory = os.path.join(VECTOR_STORE_DIR, vector_store_id)

        os.makedirs(persist_directory, exist_ok=True)

        # Save the uploaded file to a temporary location
        temp_pdf_path = os.path.join(persist_directory, file.filename)
        with open(temp_pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Convert PDF to vector store
        convert_pdf_to_vector_store(temp_pdf_path, persist_directory)

        # Optionally, remove the temporary PDF file after processing
        os.remove(temp_pdf_path)

        return UploadResponse(
            message="PDF successfully converted to vector store.",
            vector_store_id=vector_store_id,
            persist_directory=persist_directory
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to list all vector stores
@app.get("/list_vector_stores/")
def list_vector_stores():
    """
    Endpoint to list all available vector stores.

    Returns:
        dict: List of vector stores with their IDs and paths.
    """
    try:
        vector_stores = []
        for dir_name in os.listdir(VECTOR_STORE_DIR):
            dir_path = os.path.join(VECTOR_STORE_DIR, dir_name)
            if os.path.isdir(dir_path):
                vector_stores.append({
                    "vector_store_id": dir_name,
                    "persist_directory": dir_path
                })
        return {"vector_stores": vector_stores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to query multiple vector stores
@app.post("/query_vector_store/", response_model=QueryResponse)
def query_vector_store(request: QueryRequest):
    """
    Endpoint to query multiple vector stores.

    Args:
        request (QueryRequest): The query request containing vector_store_ids, question, and chat_history.

    Returns:
        dict: The answer from the QA chain.
    """
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Initialize the custom cloud LLM
        cloud_llm = CloudLLM(client=client, model_name="llama3.1-8b")

        # Create the QA chain with multiple vector stores
        qa_chain = create_qa_chain(request.vector_store_ids, embeddings, cloud_llm)

        # Perform the query
        result = qa_chain({"question": request.question, "chat_history": request.chat_history})

        return QueryResponse(
            answer=result['answer']
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))