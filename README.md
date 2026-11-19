# Advanced Retrieval-Augmented Generation (RAG) Pipeline

### An AI-Powered Document Question Answering and Evaluation System

---

## 1. Project Title

**Advanced Retrieval-Augmented Generation (RAG) Pipeline**

## 2. Short Description of the Project

The Advanced RAG Pipeline is an AI-powered document question-answering system that allows users to upload documents and ask questions based on their content.

Traditional Large Language Models (LLMs) may generate incorrect or outdated information because they do not automatically have access to a user's private documents.

This project addresses that problem by combining document retrieval with a Large Language Model (LLM). It retrieves relevant information from uploaded documents and uses that information to generate context-aware answers.

### Problems It Solves

- Difficulty finding specific information in large documents.
- Time-consuming manual document searching.
- AI-generated answers that may contain unsupported information.
- Lack of source references in document-based question answering.
- Difficulty managing and searching multiple documents.

### Key Features

- Upload PDF, TXT, and DOCX documents.
- Automatic document loading and text extraction.
- Intelligent text chunking.
- Vector-based document retrieval.
- AI-generated answers using Ollama.
- Source-aware responses.
- Similarity-distance filtering.
- Duplicate document detection.
- Document management and deletion.
- REST API using FastAPI.
- Document retrieval evaluation framework.

---

## 3. Dataset Used

This project uses a **custom document dataset** instead of a predefined machine-learning dataset.

Users can upload their own documents, and the system processes them to create a searchable knowledge base.

### Supported Dataset Formats

| Dataset Format | Description |
|---|---|
| TXT | Plain text documents |
| PDF | Portable Document Format files |
| DOCX | Microsoft Word documents |

### Example Dataset

```text
data/
└── documents/
    ├── sample.txt
    ├── document1.pdf
    └── document2.docx
```

### Dataset Processing

1. Load documents from the input directory.
2. Extract text from uploaded files.
3. Split text into smaller chunks.
4. Generate vector embeddings.
5. Store embeddings in ChromaDB.
6. Retrieve relevant chunks when a question is submitted.

**Embedding Model:** `nomic-embed-text`

**Vector Database:** ChromaDB

**Note:** The project uses user-uploaded documents as its knowledge base. The actual dataset depends on the files supplied by the user.

---

## 4. Algorithms and Technologies Used

### 4.1 Retrieval-Augmented Generation (RAG)

RAG combines information retrieval with generative AI.

It retrieves relevant document information before passing the context to an LLM to generate an answer.

### 4.2 Recursive Character Text Splitting

The Recursive Character Text Splitter divides large documents into smaller, manageable chunks.

Current configuration:

- Chunk Size: 1000 characters
- Chunk Overlap: 200 characters

### 4.3 Text Embedding

The `nomic-embed-text` model converts text chunks into numerical vector representations.

These embeddings allow the system to search for semantically relevant information.

### 4.4 Vector Similarity Search

ChromaDB stores and retrieves document embeddings.

When a user submits a question, the system searches for the most relevant document chunks.

### 4.5 Similarity Distance Filtering

The retrieval system applies a maximum-distance threshold to filter less relevant document chunks.

This helps reduce irrelevant context passed to the language model.

### 4.6 Large Language Model Generation

The project uses the `llama3.1` model through Ollama to generate answers using retrieved document context.

The system uses a context-based prompt to reduce unsupported responses and provide source references.

### Algorithms Summary

| Algorithm / Technique | Purpose |
|---|---|
| Retrieval-Augmented Generation | Combines retrieval and AI generation |
| Recursive Character Splitting | Divides documents into chunks |
| Text Embedding | Converts text into vector representations |
| Vector Similarity Search | Retrieves relevant document chunks |
| Distance Threshold Filtering | Filters less relevant results |
| LLM-Based Generation | Generates answers from retrieved context |
| SHA-256 Hashing | Detects duplicate uploaded files |

---

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Main programming language |
| LangChain | RAG pipeline development |
| Ollama | Local AI model execution |
| Llama 3.1 | Language generation model |
| Nomic Embed Text | Embedding model |
| ChromaDB | Vector database |
| FastAPI | Backend API development |
| Uvicorn | API server |
| HTML, CSS, JavaScript | Frontend development |
| PyPDF | PDF text extraction |
| python-docx | DOCX text extraction |

---

## 6. Project Architecture

```text
                 USER
                   |
                   v
          Upload Documents / Ask
                   |
                   v
             FASTAPI SERVER
                   |
          +--------+--------+
          |                 |
          v                 v
    DOCUMENT UPLOAD     USER QUESTION
          |                 |
          v                 v
    TEXT EXTRACTION    QUERY EMBEDDING
          |                 |
          v                 v
     TEXT CHUNKING     CHROMADB SEARCH
          |                 |
          v                 v
    TEXT EMBEDDINGS    RELEVANT CHUNKS
          |                 |
          v                 |
       CHROMADB <-----------+
                            |
                            v
                     CONTEXT BUILDING
                            |
                            v
                      LLAMA 3.1 LLM
                            |
                            v
                     GENERATED ANSWER
                            |
                            v
                      SOURCE REFERENCES
                            |
                            v
                           USER
```

---

## 7. Steps to Install and Run the Project

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd ADVANCED-RAG-PIPELINE
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Install and Configure Ollama

Download and install Ollama from:

https://ollama.com/

Pull the required language model:

```bash
ollama pull llama3.1
```

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Check the installed models:

```bash
ollama list
```

### Step 5: Add Documents

Place your documents inside:

```text
data/documents/
```

Supported formats:

- PDF
- TXT
- DOCX

### Step 6: Build the Vector Database

Run the indexing script:

```bash
python src/index_documents.py
```

This processes the documents, generates embeddings, and stores them in ChromaDB.

### Step 7: Start the FastAPI Server

From the project root directory, execute:

```bash
python -m uvicorn api:app --reload
```

The API server will start at:

```text
http://127.0.0.1:8000
```

### Step 8: Open the Web Application

Open the following address in your browser:

```text
http://127.0.0.1:8000/
```

### Step 9: Access API Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You can test endpoints such as:

- POST `/ask`
- POST `/upload`
- GET `/documents`
- DELETE `/documents/{document_id}`
- GET `/health`

### Step 10: Ask Questions

Example request for the `/ask` endpoint:

```json
{
    "question": "What is Retrieval Augmented Generation?",
    "top_k": 4,
    "max_distance": 1.0
}
```

The system retrieves relevant document chunks and generates an answer using Llama 3.1.

---

## 8. Results

The project demonstrates an end-to-end document question-answering workflow.

### Implemented Results

- Successfully loads supported document formats.
- Splits documents into smaller text chunks.
- Generates embeddings using `nomic-embed-text`.
- Stores document embeddings in ChromaDB.
- Retrieves relevant document chunks.
- Generates answers using Llama 3.1.
- Returns source information with generated answers.
- Filters irrelevant retrieval results using a distance threshold.
- Detects duplicate document uploads using SHA-256 hashing.

### Evaluation Metrics

The project evaluation framework is designed to measure:

| Metric | Purpose |
|---|---|
| Retrieval Precision | Measures the relevance of retrieved results |
| Source Recall | Measures whether expected source documents are retrieved |
| Keyword Coverage | Measures the presence of expected keywords in generated answers |
| Retrieved Chunks | Counts the number of chunks retrieved |

### Accuracy and Performance

**Actual evaluation results should be added after running the evaluation script.**

| Evaluation Metric | Actual Result |
|---|---|
| Retrieval Precision | To be measured |
| Source Recall | To be measured |
| Keyword Coverage | To be measured |
| Overall Accuracy | Not yet established |

Note: RAG retrieval metrics are not equivalent to conventional machine-learning classification accuracy. The evaluation results should be calculated using actual test cases and their expected outputs.

### Evaluation Command

```bash
python src/evaluator.py
```

If the evaluation script is configured and the test cases are available, the report is generated at:

```text
evaluation/reports/evaluation_report.json
```

---

## 9. Graphs and Screenshots

Add screenshots of your actual project execution in this section.

Suggested screenshots:

### 1. Web Application Interface

Show the document upload and question-answering interface.

### 2. Document Upload

Show a successfully uploaded document.

### 3. AI-Generated Answer

Show the answer generated by the RAG pipeline.

### 4. Source Retrieval

Show the filename, chunk ID, and distance score returned by the system.

### 5. FastAPI Documentation

Show the available API endpoints at `/docs`.

### 6. Evaluation Graphs

Add graphs generated from actual evaluation results, such as:

- Retrieval precision comparison.
- Source recall comparison.
- Keyword coverage comparison.
- Performance comparison across test cases.

Example screenshot folder:

```text
screenshots/
├── dashboard.png
├── document_upload.png
├── generated_answer.png
├── source_retrieval.png
├── api_documentation.png
└── evaluation_results.png
```

To display screenshots in this README after adding them to the repository, use:

```markdown
![Dashboard](screenshots/dashboard.png)

![Generated Answer](screenshots/generated_answer.png)

![Evaluation Results](screenshots/evaluation_results.png)
```

---

## 10. Future Enhancements

- Support additional document formats.
- Improve retrieval using hybrid search.
- Add reranking models.
- Implement conversation memory.
- Add multilingual document question answering.
- Develop advanced evaluation dashboards.
- Integrate user authentication.
- Deploy the application to a cloud platform.
- Improve source-grounded answer verification.

---

## 11. Conclusion

The Advanced RAG Pipeline demonstrates how Retrieval-Augmented Generation can be used to build an intelligent document question-answering system.

By combining LangChain, ChromaDB, Ollama, FastAPI, and Llama 3.1, the project provides a practical solution for searching documents and generating context-aware answers.

The system reduces manual document searching and provides source references to help users understand where retrieved information originated.

---

## 12. Author

**K. Bishwas**

B.Tech – Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning

**Project:** Advanced Retrieval-Augmented Generation (RAG) Pipeline

---

## 13. License

This project is intended for educational and academic purposes. A suitable open-source license can be added when publishing the repository.
