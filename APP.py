# frontend/app.py

import streamlit as st
import requests
import os
from typing import List

# Backend URL (adjust if your backend is hosted elsewhere)
BACKEND_URL = "http://localhost:8000"

# Set Streamlit page configuration
st.set_page_config(
    page_title="📄 Retrieval-Augmented Generation (RAG) Interface",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("📄 Retrieval-Augmented Generation (RAG) Interface")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Option", ["Upload Documents", "Query Documents"])

# Function to upload PDF
def upload_pdf(file, name=None):
    files = {"file": (file.name, file, "application/pdf")}
    data = {}
    if name:
        data["name"] = name
    try:
        response = requests.post(f"{BACKEND_URL}/upload_pdf/", files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Function to list vector stores
def list_vector_stores():
    try:
        response = requests.get(f"{BACKEND_URL}/list_vector_stores/")
        if response.status_code == 200:
            return response.json()["vector_stores"]
        else:
            st.error("Failed to fetch vector stores.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Function to query vector stores
def query_vector_store(vector_store_ids: List[str], question: str, chat_history: List = []):
    headers = {"Content-Type": "application/json"}
    payload = {
        "vector_store_ids": vector_store_ids,
        "question": question,
        "chat_history": chat_history
    }
    try:
        response = requests.post(f"{BACKEND_URL}/query_vector_store/", json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"Error: {response.json()['detail']}"
    except Exception as e:
        return f"An error occurred: {e}"

# Upload Documents Page
if app_mode == "Upload Documents":
    st.header("📤 Upload PDF Documents")
    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        vector_store_name = st.text_input("Optional: Name for the Vector Store")
        submit_button = st.form_submit_button("Upload and Create Vector Store")
    
    if submit_button:
        if uploaded_file is not None:
            with st.spinner("Uploading and processing..."):
                result = upload_pdf(uploaded_file, vector_store_name)
                if result:
                    st.success("PDF successfully converted to vector store!")
                    st.json(result)
        else:
            st.warning("Please upload a PDF file.")

# Query Documents Page
elif app_mode == "Query Documents":
    st.header("❓ Query Your Documents")
    
    # Fetch vector stores
    vector_stores = list_vector_stores()
    if vector_stores:
        # Display vector stores with multi-select
        st.subheader("Select Documents to Query")
        vector_store_ids = [vs["vector_store_id"] for vs in vector_stores]
        selected_vector_stores = st.multiselect(
            "Choose vector stores:",
            options=vector_store_ids,
            default=[]
        )
        
        if selected_vector_stores:
            # Input for question
            question = st.text_input("Enter your question:")
            if st.button("Get Answer"):
                if question.strip() == "":
                    st.warning("Please enter a valid question.")
                else:
                    with st.spinner("Generating answer..."):
                        answer = query_vector_store(selected_vector_stores, question, st.session_state.chat_history)
                        if not isinstance(answer, str):
                            st.write("**Answer:**")
                            st.write(answer)
                        else:
                            st.error(answer)
                        
                        # Update chat history
                        st.session_state.chat_history.append((question, answer))
                        
                        # Display chat history
                        st.subheader("Chat History")
                        for q, a in st.session_state.chat_history:
                            st.markdown(f"**You:** {q}")
                            st.markdown(f"**Bot:** {a}")
        else:
            st.info("Please select at least one document to query.")
    else:
        st.info("No vector stores found. Please upload documents first.")