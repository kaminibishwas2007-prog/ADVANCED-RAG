from langchain_ollama import OllamaEmbeddings


class EmbeddingEngine:

    def __init__(self):

        self.model = OllamaEmbeddings(
            model="nomic-embed-text"
        )

    def generate_embedding(self, text):

        return self.model.embed_query(text)

    def generate_document_embeddings(self, texts):

        return self.model.embed_documents(texts)


if __name__ == "__main__":

    engine = EmbeddingEngine()

    sample_text = "Retrieval Augmented Generation is an AI technology."

    vector = engine.generate_embedding(sample_text)

    print("\nEmbedding Generated Successfully!")

    print("Vector Dimensions:", len(vector))

    print("First 10 Values:", vector[:10])