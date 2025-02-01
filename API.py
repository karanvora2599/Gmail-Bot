# # main.py

# import os
# import uuid
# import json
# from datetime import datetime
# from typing import Any, Optional, List, Tuple

# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel, PrivateAttr, ConfigDict
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_chroma import Chroma  # Updated import
# from langchain.chains import ConversationalRetrievalChain
# from langchain_community.llms import Ollama

# # Import the configured logger
# from logger_config import logger

# from cerebras.cloud.sdk import Cerebras
# from langchain.llms.base import LLM

# app = FastAPI()

# # Define directories
# VECTORSTORE_BASE_DIR = "vectorstores"
# TEMP_DIR = "temp"

# # Create directories if they don't exist
# os.makedirs(VECTORSTORE_BASE_DIR, exist_ok=True)
# os.makedirs(TEMP_DIR, exist_ok=True)
# logger.info(f"Ensured directories '{VECTORSTORE_BASE_DIR}' and '{TEMP_DIR}' exist.")

# # Initialize Cerebras client
# cerebras_client = Cerebras(
#     api_key=os.getenv("CEREBRAS_API_KEY", "csk-e2e8kypw838rwmpjxd9nx2vn5jrertm339fnrcnt9c6p8hmx"),
# )

# # Initialize the custom CloudLLM
# class CloudLLM(LLM):
#     _client: Any = PrivateAttr()
#     model_name: str

#     # Configure Pydantic to allow arbitrary types
#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     def __init__(self, client: Any, model_name: str, **kwargs):
#         # Pass model_name to the parent LLM initializer
#         super().__init__(model_name=model_name, **kwargs)
#         self._client = client

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         response = self._client.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model=self.model_name,
#         )
#         return response.choices[0].message.content

#     @property
#     def _identifying_params(self):
#         return {"model_name": self.model_name}

#     @property
#     def _llm_type(self):
#         return "cloud_llm"

# cloud_llm = CloudLLM(client=cerebras_client, model_name="llama3.1-8b")

# # In-memory store for vectorstores
# vectorstores = {}

# class QueryRequest(BaseModel):
#     vectorstore_id: str
#     query: str
#     chat_history: List[List[str]] = []

# @app.post("/upload_pdf/")
# async def upload_pdf(file: UploadFile = File(...)):
#     """
#     Upload a PDF file, process it into a vector store, and persist it.
#     Returns a unique vectorstore_id for querying.
#     """
#     logger.info("Received request to upload PDF.")
    
#     if file.content_type != 'application/pdf':
#         logger.warning(f"Unsupported file type: {file.content_type}")
#         raise HTTPException(status_code=400, detail="Only PDF files are supported.")

#     # Save the uploaded PDF to a temporary file
#     temp_pdf_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.pdf")
#     try:
#         with open(temp_pdf_path, "wb") as f:
#             content = await file.read()
#             f.write(content)
#         logger.debug(f"Saved uploaded PDF to temporary path: {temp_pdf_path}")
#     except Exception as e:
#         logger.error(f"Failed to save uploaded PDF: {e}")
#         raise HTTPException(status_code=500, detail="Failed to save uploaded PDF.")

#     try:
#         # Load PDF
#         logger.info("Loading PDF using PyPDFLoader.")
#         loader = PyPDFLoader(temp_pdf_path)
#         documents = loader.load()
#         logger.debug(f"Loaded {len(documents)} documents from PDF.")

#         # Split text
#         logger.info("Splitting text into chunks.")
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=7500,
#             chunk_overlap=100
#         )
#         chunks = text_splitter.split_documents(documents)
#         logger.debug(f"Split documents into {len(chunks)} chunks.")

#         # Create embeddings using Ollama
#         logger.info("Creating embeddings using OllamaEmbeddings.")
#         embeddings = OllamaEmbeddings(model="nomic-embed-text")
#         logger.debug("Embeddings created successfully.")

#         # Generate a unique ID for the vector store
#         vectorstore_id = str(uuid.uuid4())
#         persist_directory = os.path.join(VECTORSTORE_BASE_DIR, vectorstore_id)
#         logger.debug(f"Generated vectorstore_id: {vectorstore_id}")
#         logger.debug(f"Persist directory for vector store: {persist_directory}")

#         # Create and persist vector store by specifying persist_directory during initialization
#         logger.info("Creating Chroma vector store from documents.")
#         vectorstore = Chroma.from_documents(
#             documents=chunks,
#             embedding=embeddings,
#             persist_directory=persist_directory  # Specify directory here
#         )
#         logger.debug("Chroma vector store created and persisted successfully.")

#         # Create QA chain using CloudLLM
#         logger.info("Creating ConversationalRetrievalChain with CloudLLM.")
#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=cloud_llm,  # Use the initialized CloudLLM
#             retriever=vectorstore.as_retriever()
#         )
#         logger.debug("ConversationalRetrievalChain created successfully.")

#         # Save metadata
#         metadata = {
#             "vectorstore_id": vectorstore_id,
#             "original_filename": file.filename,
#             "upload_timestamp": datetime.utcnow().isoformat()
#         }
#         metadata_path = os.path.join(persist_directory, "metadata.json")
#         with open(metadata_path, "w") as meta_file:
#             json.dump(metadata, meta_file)
#         logger.debug(f"Metadata saved at {metadata_path}")

#         # Load the persisted vector store and store in memory
#         logger.info("Loading persisted vector store into memory.")
#         loaded_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#         vectorstores[vectorstore_id] = {
#             "vectorstore": loaded_vectorstore,
#             "qa_chain": qa_chain
#         }
#         logger.info(f"Vector store '{vectorstore_id}' loaded into memory.")

#         return {"vectorstore_id": vectorstore_id, "metadata": metadata}

#     except Exception as e:
#         logger.error(f"Error processing PDF: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         # Clean up the temporary PDF file
#         if os.path.exists(temp_pdf_path):
#             os.remove(temp_pdf_path)
#             logger.debug(f"Temporary PDF file '{temp_pdf_path}' removed.")

# @app.get("/list_vectorstores/")
# def list_vectorstores():
#     """
#     List all available vector stores with their metadata.
#     """
#     logger.info("Received request to list all vector stores.")
#     try:
#         vectorstore_ids = os.listdir(VECTORSTORE_BASE_DIR)
#         logger.debug(f"Found vectorstore_ids: {vectorstore_ids}")

#         vectorstores_list = []
#         for vs_id in vectorstore_ids:
#             persist_directory = os.path.join(VECTORSTORE_BASE_DIR, vs_id)
#             metadata_path = os.path.join(persist_directory, "metadata.json")
#             if os.path.exists(metadata_path):
#                 with open(metadata_path, "r") as meta_file:
#                     metadata = json.load(meta_file)
#             else:
#                 metadata = {
#                     "vectorstore_id": vs_id,
#                     "original_filename": "Unknown",
#                     "upload_timestamp": "Unknown"
#                 }
#             vectorstores_list.append(metadata)
#             logger.debug(f"Loaded metadata for vectorstore_id {vs_id}: {metadata}")

#         logger.info(f"Returning list of {len(vectorstores_list)} vector stores.")
#         return {"vectorstores": vectorstores_list}

#     except Exception as e:
#         logger.error(f"Error listing vector stores: {e}")
#         raise HTTPException(status_code=500, detail="Failed to list vector stores.")

# @app.post("/query/")
# def query_vectorstore(request: QueryRequest):
#     """
#     Query a specific vector store using the provided query and chat history.
#     Returns the answer generated by the language model.
#     """
#     logger.info(f"Received query for vectorstore_id: {request.vectorstore_id}")
#     try:
#         # Retrieve the vector store info from in-memory store
#         vectorstore_info = vectorstores.get(request.vectorstore_id)

#         if not vectorstore_info:
#             logger.warning(f"Vector store '{request.vectorstore_id}' not found in memory. Attempting to load from disk.")
#             # If not in memory, attempt to load from disk
#             persist_directory = os.path.join(VECTORSTORE_BASE_DIR, request.vectorstore_id)
#             if not os.path.exists(persist_directory):
#                 logger.error(f"Vector store '{request.vectorstore_id}' not found on disk.")
#                 raise HTTPException(status_code=404, detail="Vector store not found.")

#             # Recreate embeddings and load vector store
#             logger.info("Loading vector store from disk.")
#             embeddings = OllamaEmbeddings(model="nomic-embed-text")
#             loaded_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#             logger.debug("Vector store loaded from disk.")

#             # Create QA chain using CloudLLM
#             logger.info("Creating ConversationalRetrievalChain with CloudLLM.")
#             qa_chain = ConversationalRetrievalChain.from_llm(
#                 llm=cloud_llm,  # Use the initialized CloudLLM
#                 retriever=loaded_vectorstore.as_retriever()
#             )
#             logger.debug("ConversationalRetrievalChain created successfully.")

#             # Store back in memory for future requests
#             vectorstores[request.vectorstore_id] = {
#                 "vectorstore": loaded_vectorstore,
#                 "qa_chain": qa_chain
#             }
#             vectorstore_info = vectorstores[request.vectorstore_id]
#             logger.info(f"Vector store '{request.vectorstore_id}' loaded into memory.")

#         # Get the QA chain
#         qa_chain = vectorstore_info["qa_chain"]
#         logger.debug("Retrieved QA chain from vector store info.")

#         # Convert chat_history from list of lists to list of tuples
#         formatted_chat_history: List[Tuple[str, str]] = [tuple(message) for message in request.chat_history]
#         logger.debug(f"Formatted chat_history: {formatted_chat_history}")

#         # Run the query using the 'invoke' method
#         logger.info(f"Executing query: {request.query}")
#         response = qa_chain.invoke({"question": request.query, "chat_history": formatted_chat_history})
#         logger.debug(f"Query executed successfully. Answer: {response['answer']}")

#         return {"answer": response['answer']}

#     except HTTPException as he:
#         logger.error(f"HTTPException occurred: {he.detail}")
#         raise he
#     except Exception as e:
#         logger.error(f"Error during query processing: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# main.py

import os
import uuid
import json
import pickle
import base64
from datetime import datetime
from typing import Any, Optional, List, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, PrivateAttr, ConfigDict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings  # Replace if switching to another provider
from langchain_chroma import Chroma  # Updated import
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Import the configured logger
from logger_config import logger

from cerebras.cloud.sdk import Cerebras
from langchain.llms.base import LLM

app = FastAPI()

# Define directories
VECTORSTORE_BASE_DIR = "vectorstores"
TEMP_DIR = "temp"
CREDENTIALS_DIR = "credentials"  # Directory to store Gmail credentials securely

# Create directories if they don't exist
os.makedirs(VECTORSTORE_BASE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CREDENTIALS_DIR, exist_ok=True)
logger.info(f"Ensured directories '{VECTORSTORE_BASE_DIR}', '{TEMP_DIR}', and '{CREDENTIALS_DIR}' exist.")

# Initialize Cerebras client
cerebras_client = Cerebras(
    api_key=os.getenv("CEREBRAS_API_KEY"),
)

# Initialize the custom CloudLLM
class CloudLLM(LLM):
    _client: Any = PrivateAttr()
    model_name: str

    # Configure Pydantic to allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, client: Any, model_name: str, **kwargs):
        # Pass model_name to the parent LLM initializer
        super().__init__(model_name=model_name, **kwargs)
        self._client = client

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    @property
    def _llm_type(self):
        return "cloud_llm"

cloud_llm = CloudLLM(client=cerebras_client, model_name="llama3.1-8b")

# In-memory store for vectorstores
vectorstores = {}

# -------------------- Gmail API Setup --------------------

# Define the scope for Gmail API
GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send'
]

def authenticate_gmail():
    """Authenticate the user and return the Gmail service."""
    creds = None
    token_path = os.path.join(CREDENTIALS_DIR, 'token.pickle')
    credentials_path = os.path.join(CREDENTIALS_DIR, 'client_secret.json')

    # Load credentials from token.pickle if it exists
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    # If no valid credentials, prompt the user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            logger.info("Gmail credentials refreshed.")
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0, access_type='offline', prompt='consent')
            logger.info("Gmail credentials obtained via OAuth flow.")
        # Save the credentials for future use
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
            logger.debug(f"Gmail credentials saved to {token_path}.")
    # Build the Gmail service
    service = build('gmail', 'v1', credentials=creds)
    logger.info("Gmail service built successfully.")
    return service

def get_latest_email(service):
    """Retrieve the most recent email from the user's Primary inbox."""
    try:
        # Fetch the latest email from the Primary category
        results = service.users().messages().list(
            userId='me',
            maxResults=1,
            q="category:primary"
        ).execute()
        messages = results.get('messages', [])
        if not messages:
            logger.warning("No messages found in the Primary category.")
            return None
        # Get the message details
        message = service.users().messages().get(
            userId='me',
            id=messages[0]['id'],
            format='full'
        ).execute()
        logger.info("Latest email fetched successfully.")
        return message
    except Exception as error:
        logger.error(f"An error occurred while fetching the latest email: {error}")
        return None

def decode_message_body(encoded_body):
    """Decode the base64url-encoded message body."""
    decoded_bytes = base64.urlsafe_b64decode(encoded_body + '==')  # Adding padding if necessary
    return decoded_bytes.decode('utf-8')

def extract_email_details(message):
    """Extract details from the email message."""
    headers = message['payload'].get('headers', [])
    subject = next((header['value'] for header in headers if header['name'] == 'Subject'), None)
    sender = next((header['value'] for header in headers if header['name'] == 'From'), None)
    recipient = next((header['value'] for header in headers if header['name'] == 'To'), None)
    body = ""
    if 'parts' in message['payload']:
        for part in message['payload']['parts']:
            if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                body = decode_message_body(part['body']['data'])
                break
    elif 'body' in message['payload'] and 'data' in message['payload']['body']:
        body = decode_message_body(message['payload']['body']['data'])
    logger.debug("Email details extracted successfully.")
    return {
        "Subject": subject,
        "From": sender,
        "To": recipient,
        "Body": body
    }

def create_reply_message(service, original_message, reply_body):
    """Create a reply message for the given original email."""
    try:
        # Extract necessary details from the original message
        thread_id = original_message['threadId']
        message_id = None
        headers = original_message['payload'].get('headers', [])
        for header in headers:
            if header['name'] == 'Message-ID':
                message_id = header['value']
            elif header['name'] == 'From':
                sender = header['value']
            elif header['name'] == 'To':
                recipient = header['value']
            elif header['name'] == 'Subject':
                subject = header['value']

        if not message_id:
            logger.error("Original message does not contain a Message-ID header.")
            return None

        # Create the reply email
        reply = MIMEText(reply_body)
        reply['To'] = sender  # Reply to the original sender
        reply['From'] = recipient  # Your email address
        reply['Subject'] = f"Re: {subject}"
        reply['In-Reply-To'] = message_id
        reply['References'] = message_id

        # Encode the message
        raw_message = base64.urlsafe_b64encode(reply.as_bytes()).decode()

        # Create the message object with threadId
        return {
            'raw': raw_message,
            'threadId': thread_id
        }
    except Exception as error:
        logger.error(f"An error occurred while creating the reply message: {error}")
        return None

def send_reply(service, original_message, reply_body):
    """Send a reply to an email message."""
    try:
        reply = create_reply_message(service, original_message, reply_body)
        if reply:
            sent_message = service.users().messages().send(userId='me', body=reply).execute()
            logger.info(f"Reply sent to thread ID: {sent_message['threadId']}")
        else:
            logger.warning("Failed to create reply message.")
    except Exception as error:
        logger.error(f"An error occurred while sending the reply: {error}")

# -------------------- Cerebras API Setup --------------------

def classify_email_body(email_body):
    """Classify the email body as 'FAQ', 'Inquiry', or 'Other'."""
    try:
        # Define the system prompt
        system_prompt = """You are an email classifier. Classify the following email body into one of the following categories: 'FAQ', 'Inquiry', or 'Other'. Respond with the classification only."""
        
        # Create the completion request
        completion = cerebras_client.chat.completions.create(
            model="llama-3.3-70b",  # Specify the model you wish to use
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": email_body}
            ],
            temperature=0.5,
            max_tokens=10,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract and return the classification
        parsed_content = completion.choices[0].message.content.strip()
        logger.debug(f"Email classified as: {parsed_content}")
        return parsed_content

    except Exception as e:
        logger.error(f"Error classifying email body: {e}")
        return "Error"

# -------------------- FastAPI Endpoints --------------------

class QueryRequest(BaseModel):
    vectorstore_id: str
    query: str
    chat_history: List[List[str]] = []

class GmailReplyRequest(BaseModel):
    reply_body: str

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, process it into a vector store, and persist it.
    Returns a unique vectorstore_id for querying.
    """
    logger.info("Received request to upload PDF.")
    
    if file.content_type != 'application/pdf':
        logger.warning(f"Unsupported file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded PDF to a temporary file
    temp_pdf_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.pdf")
    try:
        with open(temp_pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.debug(f"Saved uploaded PDF to temporary path: {temp_pdf_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded PDF.")

    try:
        # Load PDF
        logger.info("Loading PDF using PyPDFLoader.")
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        logger.debug(f"Loaded {len(documents)} documents from PDF.")

        # Split text
        logger.info("Splitting text into chunks.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        logger.debug(f"Split documents into {len(chunks)} chunks.")

        # Create embeddings using Ollama (Replace if switching to another provider)
        logger.info("Creating embeddings using OllamaEmbeddings.")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        logger.debug("Embeddings created successfully.")

        # Generate a unique ID for the vector store
        vectorstore_id = str(uuid.uuid4())
        persist_directory = os.path.join(VECTORSTORE_BASE_DIR, vectorstore_id)
        logger.debug(f"Generated vectorstore_id: {vectorstore_id}")
        logger.debug(f"Persist directory for vector store: {persist_directory}")

        # Create and persist vector store by specifying persist_directory during initialization
        logger.info("Creating Chroma vector store from documents.")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory  # Specify directory here
        )
        logger.debug("Chroma vector store created and persisted successfully.")

        # Create QA chain using CloudLLM
        logger.info("Creating ConversationalRetrievalChain with CloudLLM.")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=cloud_llm,  # Use the initialized CloudLLM
            retriever=vectorstore.as_retriever()
        )
        logger.debug("ConversationalRetrievalChain created successfully.")

        # Save metadata
        metadata = {
            "vectorstore_id": vectorstore_id,
            "original_filename": file.filename,
            "upload_timestamp": datetime.utcnow().isoformat()
        }
        metadata_path = os.path.join(persist_directory, "metadata.json")
        with open(metadata_path, "w") as meta_file:
            json.dump(metadata, meta_file)
        logger.debug(f"Metadata saved at {metadata_path}")

        # Load the persisted vector store and store in memory
        logger.info("Loading persisted vector store into memory.")
        loaded_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vectorstores[vectorstore_id] = {
            "vectorstore": loaded_vectorstore,
            "qa_chain": qa_chain
        }
        logger.info(f"Vector store '{vectorstore_id}' loaded into memory.")

        return {"vectorstore_id": vectorstore_id, "metadata": metadata}

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary PDF file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            logger.debug(f"Temporary PDF file '{temp_pdf_path}' removed.")

@app.get("/list_vectorstores/")
def list_vectorstores():
    """
    List all available vector stores with their metadata.
    """
    logger.info("Received request to list all vector stores.")
    try:
        vectorstore_ids = os.listdir(VECTORSTORE_BASE_DIR)
        logger.debug(f"Found vectorstore_ids: {vectorstore_ids}")

        vectorstores_list = []
        for vs_id in vectorstore_ids:
            persist_directory = os.path.join(VECTORSTORE_BASE_DIR, vs_id)
            metadata_path = os.path.join(persist_directory, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as meta_file:
                    metadata = json.load(meta_file)
            else:
                metadata = {
                    "vectorstore_id": vs_id,
                    "original_filename": "Unknown",
                    "upload_timestamp": "Unknown"
                }
            vectorstores_list.append(metadata)
            logger.debug(f"Loaded metadata for vectorstore_id {vs_id}: {metadata}")

        logger.info(f"Returning list of {len(vectorstores_list)} vector stores.")
        return {"vectorstores": vectorstores_list}

    except Exception as e:
        logger.error(f"Error listing vector stores: {e}")
        raise HTTPException(status_code=500, detail="Failed to list vector stores.")

@app.post("/query/")
def query_vectorstore(request: QueryRequest):
    """
    Query a specific vector store using the provided query and chat history.
    Returns the answer generated by the language model.
    """
    logger.info(f"Received query for vectorstore_id: {request.vectorstore_id}")
    try:
        # Retrieve the vector store info from in-memory store
        vectorstore_info = vectorstores.get(request.vectorstore_id)

        if not vectorstore_info:
            logger.warning(f"Vector store '{request.vectorstore_id}' not found in memory. Attempting to load from disk.")
            # If not in memory, attempt to load from disk
            persist_directory = os.path.join(VECTORSTORE_BASE_DIR, request.vectorstore_id)
            if not os.path.exists(persist_directory):
                logger.error(f"Vector store '{request.vectorstore_id}' not found on disk.")
                raise HTTPException(status_code=404, detail="Vector store not found.")

            # Recreate embeddings and load vector store
            logger.info("Loading vector store from disk.")
            embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Replace if switching to another provider
            loaded_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            logger.debug("Vector store loaded from disk.")

            # Create QA chain using CloudLLM
            logger.info("Creating ConversationalRetrievalChain with CloudLLM.")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=cloud_llm,  # Use the initialized CloudLLM
                retriever=loaded_vectorstore.as_retriever()
            )
            logger.debug("ConversationalRetrievalChain created successfully.")

            # Store back in memory for future requests
            vectorstores[request.vectorstore_id] = {
                "vectorstore": loaded_vectorstore,
                "qa_chain": qa_chain
            }
            vectorstore_info = vectorstores[request.vectorstore_id]
            logger.info(f"Vector store '{request.vectorstore_id}' loaded into memory.")

        # Get the QA chain
        qa_chain = vectorstore_info["qa_chain"]
        logger.debug("Retrieved QA chain from vector store info.")

        # Convert chat_history from list of lists to list of tuples
        formatted_chat_history: List[Tuple[str, str]] = [tuple(message) for message in request.chat_history]
        logger.debug(f"Formatted chat_history: {formatted_chat_history}")

        # Run the query using the 'invoke' method
        logger.info(f"Executing query: {request.query}")
        response = qa_chain.invoke({"question": request.query, "chat_history": formatted_chat_history})
        logger.debug(f"Query executed successfully. Answer: {response['answer']}")

        return {"answer": response['answer']}

    except HTTPException as he:
        logger.error(f"HTTPException occurred: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latest_email/")
def latest_email():
    """
    Fetch and display the latest email from the user's Primary inbox.
    """
    logger.info("Received request to fetch the latest email.")
    try:
        service = authenticate_gmail()
        message = get_latest_email(service)
        if not message:
            raise HTTPException(status_code=404, detail="No emails found in the Primary category.")
        email_details = extract_email_details(message)
        logger.info("Latest email details retrieved successfully.")
        return {"latest_email": email_details}
    except HTTPException as he:
        logger.error(f"HTTPException occurred: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error fetching latest email: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch the latest email.")

@app.post("/reply_email/")
def reply_email(reply_request: GmailReplyRequest):
    """
    Classify the latest email and send an appropriate reply.
    """
    logger.info("Received request to reply to the latest email.")
    try:
        service = authenticate_gmail()
        message = get_latest_email(service)
        if not message:
            raise HTTPException(status_code=404, detail="No emails found in the Primary category.")

        email_details = extract_email_details(message)
        email_body = email_details.get('Body', '')
        if not email_body:
            classification = 'No Body Content'
            logger.warning("No body content found in the email.")
        else:
            classification = classify_email_body(email_body)
        
        logger.info(f"Email classified as: {classification}")

        # Determine reply based on classification
        if classification == 'FAQ':
            reply_body = reply_request.reply_body or "Thank you for reaching out with your frequently asked question. Here's the information you requested..."
        elif classification == 'Inquiry':
            reply_body = reply_request.reply_body or "Thank you for your inquiry. We will get back to you shortly with the information you need."
        else:
            reply_body = reply_request.reply_body or "Thank you for your email. We will review your message and respond accordingly."

        # Send the reply
        send_reply(service, message, reply_body)
        logger.info("Reply sent successfully.")

        return {"detail": f"Reply sent successfully. Classification: {classification}"}

    except HTTPException as he:
        logger.error(f"HTTPException occurred: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error replying to email: {e}")
        raise HTTPException(status_code=500, detail="Failed to send reply to the email.")