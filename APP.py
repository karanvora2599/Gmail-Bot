# # frontend/app.py

# import streamlit as st
# import requests
# import json

# # Configure the Streamlit page
# st.set_page_config(
#     page_title="📄 PDF Vector Store Manager",
#     page_icon="📄",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Backend API URL
# API_URL = "http://127.0.0.1:8000"  # Adjust if your FastAPI is hosted elsewhere

# # Helper function to upload PDF
# def upload_pdf(file):
#     files = {"file": (file.name, file, "application/pdf")}
#     response = requests.post(f"{API_URL}/upload_pdf/", files=files)
#     if response.status_code == 200:
#         data = response.json()
#         return data.get("vectorstore_id"), data.get("metadata")
#     else:
#         st.error(f"Failed to upload PDF: {response.text}")
#         return None, None

# # Helper function to list vector stores
# def list_vectorstores():
#     response = requests.get(f"{API_URL}/list_vectorstores/")
#     if response.status_code == 200:
#         return response.json().get("vectorstores", [])
#     else:
#         st.error(f"Failed to retrieve vector stores: {response.text}")
#         return []

# # Helper function to query vector store
# def query_vectorstore(vectorstore_id, query, chat_history):
#     payload = {
#         "vectorstore_id": vectorstore_id,
#         "query": query,
#         "chat_history": chat_history
#     }
#     headers = {"Content-Type": "application/json"}
#     response = requests.post(f"{API_URL}/query/", data=json.dumps(payload), headers=headers)
#     if response.status_code == 200:
#         return response.json()["answer"]
#     else:
#         st.error(f"Failed to get answer: {response.text}")
#         return None

# # Sidebar for navigation
# st.sidebar.title("📄 PDF Vector Store Manager")
# app_mode = st.sidebar.selectbox("Choose the app mode", ["Upload PDF", "Query Vector Store", "About"])

# # Upload PDF Section
# if app_mode == "Upload PDF":
#     st.title("📤 Upload PDF to Create Vector Store")
#     st.markdown("---")
#     st.write("Upload your PDF file to process and create a searchable vector store.")

#     uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=False)

#     if uploaded_file is not None:
#         st.success(f"Uploaded file: `{uploaded_file.name}`")
#         if st.button("Upload and Process"):
#             with st.spinner("Processing PDF..."):
#                 vectorstore_id, metadata = upload_pdf(uploaded_file)
#                 if vectorstore_id:
#                     st.success("PDF uploaded and processed successfully!")
#                     st.markdown(f"""
#                     ### 📌 Vector Store ID
#                     `{vectorstore_id}`
#                     """)
#                     st.markdown(f"""
#                     ### 📄 Original Filename
#                     `{metadata.get("original_filename", "Unknown")}`
#                     """)
#                     st.markdown(f"""
#                     ### 🕒 Upload Timestamp
#                     `{metadata.get("upload_timestamp", "Unknown")}`
#                     """)
#                     st.info("You can now query this vector store from the 'Query Vector Store' section.")

# # Query Vector Store Section
# elif app_mode == "Query Vector Store":
#     st.title("🔍 Query Your Vector Store")
#     st.markdown("---")
#     st.write("Select a vector store and ask your questions.")

#     # Fetch available vector stores
#     vectorstores = list_vectorstores()

#     if vectorstores:
#         # Create a selection list with meaningful labels
#         vectorstore_options = [
#             f"{vs['original_filename']} (ID: {vs['vectorstore_id']})" for vs in vectorstores
#         ]
#         selected_option = st.selectbox("Select a Vector Store", vectorstore_options)

#         # Extract the vectorstore_id from the selected option
#         if selected_option:
#             # Assuming the format is "filename (ID: vectorstore_id)"
#             vectorstore_id = selected_option.split(" (ID: ")[-1].rstrip(")")

#             st.subheader("💬 Chat with the Vector Store")
#             if 'chat_history' not in st.session_state:
#                 st.session_state['chat_history'] = []

#             query = st.text_input("Ask a question about the PDF", "")
#             if st.button("Get Answer"):
#                 if query.strip() == "":
#                     st.warning("Please enter a valid question.")
#                 else:
#                     with st.spinner("Fetching answer..."):
#                         answer = query_vectorstore(vectorstore_id, query, st.session_state['chat_history'])
#                         if answer:
#                             st.session_state['chat_history'].append((query, answer))
#                             st.markdown(f"""
#                             **🧑‍💼 Q:** {query}

#                             **🤖 A:** {answer}
#                             """)
#             # Display chat history in a scrollable container
#             st.subheader("📝 Conversation History")
#             if st.session_state['chat_history']:
#                 for i, (q, a) in enumerate(st.session_state['chat_history'], 1):
#                     st.markdown(f"**{i}. 🧑‍💼 Q:** {q}")
#                     st.markdown(f"**{i}. 🤖 A:** {a}")
#                     st.markdown("---")
#             else:
#                 st.write("No conversation history yet.")
#     else:
#         st.info("No vector stores available. Please upload a PDF to create one.")

# # About Section
# elif app_mode == "About":
#     st.title("ℹ️ About PDF Vector Store Manager")
#     st.markdown("---")
#     st.markdown("""
#     **PDF Vector Store Manager** is a tool that allows you to upload PDF documents, convert them into searchable vector stores, and perform intelligent queries against their content.

#     ### 🌟 Features
#     - **📤 Upload PDFs:** Easily upload and process PDF files.
#     - **🗄️ Vector Stores:** Convert PDFs into vector stores for efficient retrieval.
#     - **🔍 Query Interface:** Ask questions and get answers based on the content of your PDFs.
#     - **💬 Interactive Chat:** Maintain conversation history for contextual queries.
#     - **🎨 User-Friendly UI:** Attractive and intuitive interface built with Streamlit.

#     ### 🛠️ Technologies Used
#     - **Streamlit:** For building the frontend.
#     - **FastAPI:** Backend API for processing and querying PDFs.
#     - **LangChain, Chroma, Ollama:** For document processing, embeddings, and language model interactions.

#     ### 🚀 Getting Started
#     1. **Run the FastAPI Backend:**
#         ```bash
#         cd backend
#         uvicorn main:app --reload
#         ```
#     2. **Run the Streamlit Frontend:**
#         ```bash
#         cd frontend
#         streamlit run app.py
#         ```

#     ### 📝 License
#     MIT License

#     ### 📞 Contact
#     For any questions or support, feel free to reach out!

#     **Developed by [Your Name]**
#     """)

# frontend/app.py

import streamlit as st
import requests
import json

# Configure the Streamlit page
st.set_page_config(
    page_title="📄 PDF Vector Store Manager",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend API URL
API_URL = "http://127.0.0.1:8000"  # Adjust if your FastAPI is hosted elsewhere

# Helper function to upload PDF
def upload_pdf(file):
    files = {"file": (file.name, file, "application/pdf")}
    response = requests.post(f"{API_URL}/upload_pdf/", files=files)
    if response.status_code == 200:
        data = response.json()
        return data.get("vectorstore_id"), data.get("metadata")
    else:
        st.error(f"Failed to upload PDF: {response.text}")
        return None, None

# Helper function to list vector stores
def list_vectorstores():
    response = requests.get(f"{API_URL}/list_vectorstores/")
    if response.status_code == 200:
        return response.json().get("vectorstores", [])
    else:
        st.error(f"Failed to retrieve vector stores: {response.text}")
        return []

# Helper function to query vector store
def query_vectorstore(vectorstore_id, query, chat_history):
    payload = {
        "vectorstore_id": vectorstore_id,
        "query": query,
        "chat_history": chat_history
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{API_URL}/query/", data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        return response.json()["answer"]
    else:
        st.error(f"Failed to get answer: {response.text}")
        return None

# Helper function to fetch the latest email
def fetch_latest_email():
    response = requests.get(f"{API_URL}/latest_email/")
    if response.status_code == 200:
        return response.json().get("latest_email", {})
    else:
        st.error(f"Failed to fetch latest email: {response.text}")
        return {}

# Helper function to send a reply to the latest email
def send_email_reply(reply_body):
    payload = {
        "reply_body": reply_body
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{API_URL}/reply_email/", data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        st.success(response.json().get("detail", "Reply sent successfully."))
    else:
        st.error(f"Failed to send reply: {response.text}")

# Sidebar for navigation
st.sidebar.title("📄 PDF Vector Store Manager")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Upload PDF", "Query Vector Store", "Email Manager", "About"])

# Upload PDF Section
if app_mode == "Upload PDF":
    st.title("📤 Upload PDF to Create Vector Store")
    st.markdown("---")
    st.write("Upload your PDF file to process and create a searchable vector store.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=False)

    if uploaded_file is not None:
        st.success(f"Uploaded file: `{uploaded_file.name}`")
        if st.button("Upload and Process"):
            with st.spinner("Processing PDF..."):
                vectorstore_id, metadata = upload_pdf(uploaded_file)
                if vectorstore_id:
                    st.success("PDF uploaded and processed successfully!")
                    st.markdown(f"""
                    ### 📌 Vector Store ID
                    `{vectorstore_id}`
                    """)
                    st.markdown(f"""
                    ### 📄 Original Filename
                    `{metadata.get("original_filename", "Unknown")}`
                    """)
                    st.markdown(f"""
                    ### 🕒 Upload Timestamp
                    `{metadata.get("upload_timestamp", "Unknown")}`
                    """)
                    st.info("You can now query this vector store from the 'Query Vector Store' section.")

# Query Vector Store Section
elif app_mode == "Query Vector Store":
    st.title("🔍 Query Your Vector Store")
    st.markdown("---")
    st.write("Select a vector store and ask your questions.")

    # Fetch available vector stores
    vectorstores = list_vectorstores()

    if vectorstores:
        # Create a selection list with meaningful labels
        vectorstore_options = [
            f"{vs['original_filename']} (ID: {vs['vectorstore_id']})" for vs in vectorstores
        ]
        selected_option = st.selectbox("Select a Vector Store", vectorstore_options)

        # Extract the vectorstore_id from the selected option
        if selected_option:
            # Assuming the format is "filename (ID: vectorstore_id)"
            vectorstore_id = selected_option.split(" (ID: ")[-1].rstrip(")")

            st.subheader("💬 Chat with the Vector Store")
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []

            query = st.text_input("Ask a question about the PDF", "")
            if st.button("Get Answer"):
                if query.strip() == "":
                    st.warning("Please enter a valid question.")
                else:
                    with st.spinner("Fetching answer..."):
                        answer = query_vectorstore(vectorstore_id, query, st.session_state['chat_history'])
                        if answer:
                            st.session_state['chat_history'].append((query, answer))
                            st.markdown(f"""
                            **🧑‍💼 Q:** {query}

                            **🤖 A:** {answer}
                            """)
            # Display chat history in a scrollable container
            st.subheader("📝 Conversation History")
            if st.session_state['chat_history']:
                for i, (q, a) in enumerate(st.session_state['chat_history'], 1):
                    st.markdown(f"**{i}. 🧑‍💼 Q:** {q}")
                    st.markdown(f"**{i}. 🤖 A:** {a}")
                    st.markdown("---")
            else:
                st.write("No conversation history yet.")
    else:
        st.info("No vector stores available. Please upload a PDF to create one.")

# Email Manager Section
elif app_mode == "Email Manager":
    st.title("📧 Email Manager")
    st.markdown("---")
    st.write("Fetch the latest email, review its details, and send an appropriate reply based on its classification.")

    if st.button("Fetch Latest Email"):
        with st.spinner("Fetching the latest email..."):
            latest_email = fetch_latest_email()
            if latest_email:
                st.subheader("🗂️ Email Details")
                st.markdown(f"""
                **📄 Subject:** {latest_email.get('Subject', 'No Subject')}
                **📨 From:** {latest_email.get('From', 'Unknown Sender')}
                **📥 To:** {latest_email.get('To', 'Unknown Recipient')}
                """)
                st.markdown("### ✉️ Body:")
                st.write(latest_email.get('Body', 'No Content'))

                classification = latest_email.get('Classification', 'Not Classified')
                st.markdown(f"**📝 Classification:** `{classification}`")

                st.markdown("---")
                st.write("### 💬 Send a Reply")

                # Text area for custom reply or use default based on classification
                default_reply = ""
                if classification == 'FAQ':
                    default_reply = "Thank you for reaching out with your frequently asked question. Here's the information you requested..."
                elif classification == 'Inquiry':
                    default_reply = "Thank you for your inquiry. We will get back to you shortly with the information you need."
                else:
                    default_reply = "Thank you for your email. We will review your message and respond accordingly."

                reply_body = st.text_area("Reply Body", value=default_reply, height=150)

                if st.button("Send Reply"):
                    if reply_body.strip() == "":
                        st.warning("Reply body cannot be empty.")
                    else:
                        with st.spinner("Sending reply..."):
                            send_email_reply(reply_body)
    else:
        st.info("Click the button above to fetch and manage your latest email.")

# About Section
elif app_mode == "About":
    st.title("ℹ️ About PDF Vector Store Manager")
    st.markdown("---")
    st.markdown("""
    **PDF Vector Store Manager** is a tool that allows you to upload PDF documents, convert them into searchable vector stores, and perform intelligent queries against their content.

    ### 🌟 Features
    - **📤 Upload PDFs:** Easily upload and process PDF files.
    - **🗄️ Vector Stores:** Convert PDFs into vector stores for efficient retrieval.
    - **🔍 Query Interface:** Ask questions and get answers based on the content of your PDFs.
    - **📧 Email Manager:** Fetch, review, classify, and respond to your latest emails.
    - **💬 Interactive Chat:** Maintain conversation history for contextual queries.
    - **🎨 User-Friendly UI:** Attractive and intuitive interface built with Streamlit.

    ### 🛠️ Technologies Used
    - **Streamlit:** For building the frontend.
    - **FastAPI:** Backend API for processing and querying PDFs and managing emails.
    - **LangChain, Chroma, Ollama:** For document processing, embeddings, and language model interactions.
    - **Google Gmail API:** For fetching and replying to emails.
    - **Cerebras Cloud SDK:** For interacting with Cerebras' language models.

    ### 🚀 Getting Started
    1. **Run the FastAPI Backend:**
        ```bash
        cd backend
        uvicorn main:app --reload
        ```
    2. **Run the Streamlit Frontend:**
        ```bash
        cd frontend
        streamlit run app.py
        ```

    ### 📝 License
    MIT License

    ### 📞 Contact
    For any questions or support, feel free to reach out!

    **Developed by [Your Name]**
    """)