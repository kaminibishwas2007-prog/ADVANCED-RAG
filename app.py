import streamlit as st
import sys
import os
import tempfile

sys.path.append("src")

from pipeline import AdvancedRAGPipeline
from ingestion import run_ingestion
from vector_database import create_vector_database

from document_loader import load_document
from text_splitter import split_documents


st.set_page_config(
    page_title="Advanced RAG AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ADVANCED RAG AI")
st.caption("Intelligent Document Retrieval & Generation System")


@st.cache_resource
def initialize_rag():

    return AdvancedRAGPipeline()


rag = initialize_rag()


with st.sidebar:

    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):

        if uploaded_files:

            all_documents = []

            for uploaded_file in uploaded_files:

                extension = os.path.splitext(
                    uploaded_file.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())

                    temp_path = temp_file.name

                try:

                    content = load_document(temp_path)

                    all_documents.append({
                        "filename": uploaded_file.name,
                        "content": content
                    })

                finally:

                    os.unlink(temp_path)

            chunks = split_documents(all_documents)

            create_vector_database(chunks)

            st.success(
                f"Successfully indexed {len(uploaded_files)} documents!"
            )

    st.divider()

    st.subheader("System Information")

    st.write("LLM: Llama 3.1")

    st.write("Embeddings: Nomic Embed Text")

    st.write("Vector DB: ChromaDB")


if "messages" not in st.session_state:

    st.session_state.messages = []


if st.button("Clear Chat"):

    st.session_state.messages = []

    st.rerun()


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


question = st.chat_input(
    "Ask something about your documents..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            try:

                result = rag.ask(question)

                answer = result["answer"]

                st.markdown(answer)

                with st.expander("View Sources"):

                    for source in result["sources"]:

                        st.write(source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as error:

                st.error(f"Error: {error}")