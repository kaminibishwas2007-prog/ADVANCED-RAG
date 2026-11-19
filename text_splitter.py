from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200,

        separators=["\n\n", "\n", ".", " ", ""]

    )

    chunks = []

    for document in documents:

        split_texts = splitter.split_text(
            document["content"]
        )

        for index, chunk in enumerate(split_texts):

            chunks.append({

                "text": chunk,

                "metadata": {

                    "filename": document["filename"],

                    "chunk_id": index

                }

            })

    print(f"Total chunks generated: {len(chunks)}")

    return chunks