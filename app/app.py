import streamlit as st
import requests
from datetime import datetime

# API endpoint configuration
API_BASE_URL = "http://localhost:8000"


def upload_file(file, session_id=None):
    """Upload file to the API"""
    files = {"file": file}
    params = {"session_id": session_id} if session_id else {}

    response = requests.post(
        f"{API_BASE_URL}/documents/upload",
        files=files,
        params=params
    )
    return response.json()


def send_message(question, session_id):
    """Send message to chat API"""
    response = requests.post(
        f"{API_BASE_URL}/chat/",
        json={"question": question, "session_id": session_id}
    )
    return response.json()


def get_chat_history(session_id):
    """Get chat history for session"""
    response = requests.post(
        f"{API_BASE_URL}/chat/history",
        params={"session_id": session_id}
    )
    return response.json()


def initialize_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False


def display_chat_message(role, content, timestamp=None):
    """Display chat message with styling"""
    if role == "user":
        st.write(f'👤 **You**: {content}')
    else:
        st.write(f'🤖 **Assistant**: {content}')
    if timestamp:
        st.caption(f"Sent at: {timestamp}")


def main():
    st.set_page_config(page_title="Document Chat Assistant", layout="wide")
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a document", type=['pdf', 'txt', 'docx'])

        if uploaded_file and not st.session_state.file_uploaded:
            with st.spinner("Processing document..."):
                response = upload_file(
                    uploaded_file, st.session_state.session_id)
                if response.get("status") == "success":
                    st.session_state.session_id = response.get("session_id")
                    st.session_state.file_uploaded = True
                    st.success("Document uploaded successfully!")
                    st.write(f"Session ID: {st.session_state.session_id}")
                else:
                    st.error("Failed to upload document")

        if st.session_state.session_id:
            if st.button("Clear Session"):
                st.session_state.session_id = None
                st.session_state.messages = []
                st.session_state.file_uploaded = False
                st.experimental_rerun()

    # Main chat interface
    st.title("Document Chat Assistant")

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        if st.session_state.session_id:
            # Load chat history
            history = get_chat_history(st.session_state.session_id)
            if history:
                for message in history:
                    display_chat_message(
                        message["role"],
                        message["content"],
                        message.get("timestamp")
                    )

    # Chat input
    if st.session_state.session_id:
        user_input = st.chat_input("Ask a question about your document...")
        if user_input:
            # Send message to API
            with st.spinner("Thinking..."):
                response = send_message(
                    user_input, st.session_state.session_id)

                if response.get("answer"):
                    # Display new messages
                    with chat_container:
                        display_chat_message("user", user_input)
                        display_chat_message(
                            "assistant",
                            response["answer"],
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )

                        if response.get("processing_time"):
                            st.caption(
                                f"Processing time: {response['processing_time']:.2f} seconds")
                else:
                    st.error("Failed to get response from assistant")
    else:
        st.info("Please upload a document to start chatting!")


if __name__ == "__main__":
    main()
