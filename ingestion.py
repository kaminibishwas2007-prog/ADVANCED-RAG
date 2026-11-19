from document_loader import load_all_documents
from text_splitter import split_documents

DOCUMENT_FOLDER = "data/documents"


def run_ingestion():

    print("\nStarting Document Ingestion...\n")

    documents = load_all_documents(DOCUMENT_FOLDER)

    print("\nDocument Loading Completed!")

    chunks = split_documents(documents)

    print("\nIngestion Summary")

    print("Documents:", len(documents))

    print("Chunks:", len(chunks))

    return chunks


if __name__ == "__main__":

    run_ingestion()