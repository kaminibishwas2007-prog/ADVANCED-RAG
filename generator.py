from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class RAGGenerator:

    def __init__(self):

        self.llm = ChatOllama(
            model="llama3.1",
            temperature=0.2
        )

        self.prompt = ChatPromptTemplate.from_template("""

You are an advanced Retrieval Augmented Generation AI assistant.

Your task is to answer the user's question using ONLY the
information provided in the retrieved context.

IMPORTANT RULES:

1. Use only the supplied context.
2. Never invent facts or information.
3. If the answer is unavailable, clearly say:
   "The requested information is not available in the uploaded documents."
4. Include source citations in your answers.
5. Use citations in this format: [SOURCE 1], [SOURCE 2].
6. Cite the source that supports each important statement.
7. Explain answers in simple and professional language.
8. Use bullet points when necessary.
9. Do not create fake citations.

RETRIEVED CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:

Provide a clear answer with appropriate source citations.

""")

    def generate(self, question, context):

        chain = self.prompt | self.llm

        response = chain.invoke({
            "question": question,
            "context": context
        })

        return response.content


if __name__ == "__main__":

    generator = RAGGenerator()

    question = "What is RAG?"

    context = """
    SOURCE 1

    Filename: sample.txt

    Content:
    Retrieval Augmented Generation (RAG) is an AI architecture
    that combines information retrieval with large language models.
    It retrieves relevant information from external knowledge
    sources before generating an answer.
    """

    answer = generator.generate(question, context)

    print("\nGENERATED ANSWER:\n")
    print(answer)