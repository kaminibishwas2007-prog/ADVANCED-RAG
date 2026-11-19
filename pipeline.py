from retriever import RAGRetriever
from generator import RAGGenerator


class AdvancedRAGPipeline:

    def __init__(self):

        print("Initializing RAG Pipeline...")

        self.retriever = RAGRetriever()

        self.generator = RAGGenerator()

        print("RAG Pipeline Ready!")

    def ask(self, question, top_k=4, max_distance=1.0):

        print("\nSearching Knowledge Base...")

        # Step 1: Retrieve relevant documents

        documents = self.retriever.retrieve(
            query=question,
            k=top_k,
            max_distance=max_distance
        )

        # Step 2: Check retrieval results

        if not documents:

            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "retrieved_chunks": 0
            }

        # Step 3: Optimize context

        context_parts = []

        sources = []

        for index, document in enumerate(documents, start=1):

            metadata = document["metadata"]

            filename = metadata.get(
                "filename",
                "Unknown"
            )

            chunk_id = metadata.get(
                "chunk_id",
                "Unknown"
            )

            context_parts.append(f"""
[SOURCE {index}]

Filename: {filename}

Chunk ID: {chunk_id}

Content:
{document["content"]}
""")

            sources.append({

                "filename": filename,

                "chunk_id": chunk_id,

                "document_id": metadata.get(
                    "document_id",
                    "Original document"
                ),

                "distance_score": document["similarity_score"]

            })

        context = "\n\n".join(context_parts)

        print("\nContext Optimization Complete!")

        print("Retrieved Chunks:", len(documents))

        # Step 4: Generate answer

        print("\nGenerating AI Answer...")

        answer = self.generator.generate(
            question=question,
            context=context
        )

        # Step 5: Return results

        return {

            "answer": answer,

            "sources": sources,

            "retrieved_chunks": len(documents)

        }


# ==========================================
# TERMINAL TESTING
# ==========================================

if __name__ == "__main__":

    rag = AdvancedRAGPipeline()

    while True:

        question = input(
            "\nAsk a question (type exit to stop): "
        )

        if question.lower() == "exit":
            break

        result = rag.ask(
            question=question,
            top_k=4,
            max_distance=1.0
        )

        print("\nAI ANSWER:\n")

        print(result["answer"])

        print("\nSOURCES:\n")

        for source in result["sources"]:

            print(source)

        print(
            "\nRetrieved Chunks:",
            result["retrieved_chunks"]
        )