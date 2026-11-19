from ingestion import run_ingestion

from vector_database import create_vector_database


def build_index():

    print("\n================================")

    print("ADVANCED RAG INDEXING SYSTEM")

    print("================================")

    # Step 1: Load and split documents

    chunks = run_ingestion()

    if not chunks:

        print("No documents found!")

        return

    # Step 2: Generate and store embeddings

    vector_db = create_vector_database(chunks)

    print("\n================================")

    print("INDEXING COMPLETED SUCCESSFULLY!")

    print("================================")


if __name__ == "__main__":

    build_index()