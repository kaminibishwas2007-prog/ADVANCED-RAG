from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# ==========================================
# CONFIGURATION
# ==========================================

VECTOR_DATABASE_PATH = "vectorstore/chroma_db"

COLLECTION_NAME = "advanced_rag_documents"

DEFAULT_TOP_K = 4

# Maximum permitted vector distance.
# Lower distance generally means a closer match.

DEFAULT_MAX_DISTANCE = 1.0


# ==========================================
# RAG RETRIEVER
# ==========================================

class RAGRetriever:

    def __init__(self):

        print("Initializing RAG Retriever...")

        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        self.vector_db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=VECTOR_DATABASE_PATH
        )

        print("RAG Retriever Ready!")


    # ======================================
    # RETRIEVE RELEVANT DOCUMENTS
    # ======================================

    def retrieve(
        self,
        query,
        k=DEFAULT_TOP_K,
        max_distance=DEFAULT_MAX_DISTANCE
    ):

        print("\nSearching Vector Database...")

        results = self.vector_db.similarity_search_with_score(
            query,
            k=k
        )

        retrieved_documents = []

        for document, score in results:

            distance = float(score)

            # Apply similarity threshold

            if distance > max_distance:

                print(
                    f"Filtered document with distance: {distance:.4f}"
                )

                continue

            retrieved_documents.append({

                "content": document.page_content,

                "metadata": document.metadata,

                "similarity_score": distance

            })

        print(
            f"\nRetrieved Documents: {len(retrieved_documents)}"
        )

        print(
            f"Maximum Allowed Distance: {max_distance}"
        )

        return retrieved_documents


# ==========================================
# TEST RETRIEVER
# ==========================================

if __name__ == "__main__":

    retriever = RAGRetriever()

    question = input("\nEnter your question: ")

    results = retriever.retrieve(
        query=question,
        k=4,
        max_distance=1.0
    )

    print("\nRETRIEVED DOCUMENTS")

    if not results:

        print("No sufficiently relevant documents found.")

    else:

        for index, result in enumerate(results, start=1):

            print(f"\nDocument {index}")

            print("Content:", result["content"])

            print("Metadata:", result["metadata"])

            print(
                "Distance:",
                result["similarity_score"]
            )