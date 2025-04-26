import streamlit as st
from src.helper import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain
)

def handle_user_input(user_question):
    """
    Handles the chat interaction and displays conversation history.
    """
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        
        for i, message in enumerate(st.session_state.chat_history):
            speaker = "**User:**" if i % 2 == 0 else "**Bot:**"
            st.markdown(f"{speaker} {message.content}")
    else:
        st.warning("⚠️ Please upload and process PDFs first.")

def main():
    st.set_page_config(page_title="Information Retrieval", layout="wide")
    st.title("📄🔍 PDF Q&A Assistant")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.header("📁 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
        
        st.markdown("Customize processing options:")
        chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=100, value=20, step=5)
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2"
            ],
            index=0
        )

        if st.button("📥 Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    vector_store = get_vector_store(text_chunks, embedding_model=embedding_model)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("✅ PDFs processed! You can now ask questions.")
            else:
                st.error("⚠️ Please upload at least one PDF.")

    # Chat input field
    user_question = st.text_input("💬 Ask a question about your PDFs:")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()
