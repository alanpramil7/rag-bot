import streamlit as st
import requests
import json
from typing import Dict
import sseclient

API_BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/documents/upload"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat/stream"
HISTORY_ENDPOINT = f"{API_BASE_URL}/chat/history"

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

def upload_file(uploaded_file) -> Dict:
    try:
        if uploaded_file.name in st.session_state.processed_files:
            return None

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        params = {}
        if st.session_state.session_id:
            params["session_id"] = st.session_state.session_id

        with st.spinner(f"Processing {uploaded_file.name}..."):
            response = requests.post(
                UPLOAD_ENDPOINT,
                files=files,
                params=params
            )

            if response.status_code != 200:
                st.error(f"Upload failed: {response.text}")
                return None

            response_data = response.json()

            st.session_state.processed_files.add(uploaded_file.name)

            return response_data

    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def get_chat_history():
    try:
        if st.session_state.session_id:
            response = requests.post(
                HISTORY_ENDPOINT,
                params={"session_id": st.session_state.session_id}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Error fetching chat history: {str(e)}")
        return []

def stream_chat(question: str):
    try:
        headers = {'Content-Type': 'application/json'}
        data = {
            "question": question,
            "session_id": st.session_state.session_id
        }

        response = requests.post(
            CHAT_ENDPOINT,
            headers=headers,
            json=data,
            stream=True
        )
        response.raise_for_status()

        client = sseclient.SSEClient(response)

        # Create a placeholder for the streaming response
        message_placeholder = st.empty()
        full_response = ""

        for event in client.events():
            if event.data:
                try:
                    chunk_data = json.loads(event.data)
                    full_response += chunk_data.get("answer", "")
                    # Update the message placeholder with the accumulated response
                    message_placeholder.markdown(full_response + "▌")
                except json.JSONDecodeError:
                    continue

        message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(f"Error in chat stream: {str(e)}")

def main():
    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session_state()

    st.title("RAG Chat Assistant")

    # Sidebar for file upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=['pdf', 'txt', 'docx'],
            help="Supported formats: PDF, TXT, DOCX"
        )

        if uploaded_file:
            result = upload_file(uploaded_file)
            if result:
                st.session_state.session_id = result.get("session_id")
                st.success(f"✅ {uploaded_file.name} processed successfully!")
                st.write(f"Created {result.get('chunks_created')} chunks")

        if st.session_state.session_id:
            st.write("---")
            st.write("📝 Session Information")
            st.write(f"Session ID: `{st.session_state.session_id}`")

            if st.session_state.processed_files:
                st.write("Processed Files:")
                for file in st.session_state.processed_files:
                    st.write(f"- {file}")

            if st.button("Clear Session", key="clear_session"):
                st.session_state.session_id = None
                st.session_state.messages = []
                st.session_state.processed_files = set()
                st.rerun()

    # Main chat interface
    chat_container = st.container()

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            stream_chat(prompt)


if __name__ == "__main__":
    main()
