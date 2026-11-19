from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import os

VECTOR_DATABASE_PATH = "vectorstore/chroma_db"

COLLECTION_NAME = "advanced_rag_documents"


def get_embedding_function():

    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


def create_vector_database(chunks):

    embeddings = get_embedding_function()

    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DATABASE_PATH
    )

    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        chunk["metadata"] for chunk in chunks
    ]

    ids = [
        f"{metadata['filename']}_{metadata['chunk_id']}"
        for metadata in metadatas
    ]

    if texts:

        vector_db.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )

    print("\nVector Database Created Successfully!")

    print("Total Stored Documents:", vector_db._collection.count())

    return vector_db


if __name__ == "__main__":

    print("Vector Database Module Ready!")