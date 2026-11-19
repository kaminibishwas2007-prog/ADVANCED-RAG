import sys
import uuid
import hashlib

from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR / "src"))

from pipeline import AdvancedRAGPipeline
from document_loader import load_document
from text_splitter import split_documents


DOCUMENT_FOLDER = BASE_DIR / "data" / "documents"

DOCUMENT_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

MAX_FILE_SIZE = 20 * 1024 * 1024


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Advanced RAG Pipeline API",
    description="AI-powered Retrieval Augmented Generation System",
    version="3.0"
)

rag = None


# ==========================================
# REQUEST MODEL
# ==========================================

class QuestionRequest(BaseModel):

    question: str

    top_k: int = Field(
        default=4,
        ge=1,
        le=20
    )

    max_distance: float = Field(
        default=1.0,
        gt=0
    )


# ==========================================
# FILE HASH FUNCTION
# ==========================================

def calculate_file_hash(file_content):

    return hashlib.sha256(file_content).hexdigest()


# ==========================================
# DUPLICATE DETECTION
# ==========================================

def find_duplicate_file(file_hash):

    for existing_file in DOCUMENT_FOLDER.iterdir():

        if not existing_file.is_file():
            continue

        try:

            existing_content = existing_file.read_bytes()

            existing_hash = calculate_file_hash(
                existing_content
            )

            if existing_hash == file_hash:

                return existing_file.name

        except Exception:
            continue

    return None


# ==========================================
# STARTUP
# ==========================================

@app.on_event("startup")
def startup_event():

    global rag

    print("Initializing Advanced RAG Pipeline...")

    rag = AdvancedRAGPipeline()

    print("RAG API Ready!")


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return FileResponse(
        BASE_DIR / "frontend" / "index.html"
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "online",
        "message": "Advanced RAG API is running"
    }


# ==========================================
# ASK AI WITH TOP-K CONTROL
# ==========================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    if rag is None:

        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized"
        )

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:

        result = rag.ask(
            question=question,
            top_k=request.top_k,
            max_distance=request.max_distance
        )

        return {

            "question": question,

            "answer": result["answer"],

            "sources": result["sources"],

            "retrieved_chunks": result["retrieved_chunks"],

            "retrieval_settings": {

                "top_k": request.top_k,

                "max_distance": request.max_distance

            }

        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==========================================
# UPLOAD DOCUMENT
# ==========================================

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if rag is None:

        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized"
        )

    original_filename = Path(file.filename or "").name

    extension = Path(original_filename).suffix.lower()

    if not original_filename or extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT and DOCX files are supported"
        )

    file_content = await file.read()

    if not file_content:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    if len(file_content) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Maximum file size allowed is 20 MB"
        )

    file_hash = calculate_file_hash(file_content)

    duplicate = find_duplicate_file(file_hash)

    if duplicate:

        raise HTTPException(
            status_code=409,
            detail={
                "message": "Duplicate document detected!",
                "uploaded_filename": original_filename,
                "existing_filename": duplicate
            }
        )

    document_id = str(uuid.uuid4())

    stored_filename = f"{document_id}_{original_filename}"

    file_path = DOCUMENT_FOLDER / stored_filename

    try:

        file_path.write_bytes(file_content)

        print(f"Document saved: {stored_filename}")

        text = load_document(str(file_path))

        if not text.strip():

            raise ValueError(
                "No readable text found in document"
            )

        chunks = split_documents([
            {
                "filename": original_filename,
                "content": text
            }
        ])

        if not chunks:

            raise ValueError(
                "No text chunks were generated"
            )

        texts = [chunk["text"] for chunk in chunks]

        metadatas = []

        ids = []

        for chunk in chunks:

            metadata = chunk["metadata"].copy()

            metadata["document_id"] = document_id

            metadata["file_hash"] = file_hash

            metadatas.append(metadata)

            ids.append(
                f"{document_id}_{metadata['chunk_id']}"
            )

        rag.retriever.vector_db.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )

        return {

            "status": "success",

            "message": "Document uploaded and indexed successfully!",

            "document_id": document_id,

            "filename": original_filename,

            "total_chunks": len(chunks)

        }

    except Exception as error:

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==========================================
# DOCUMENT LIBRARY
# ==========================================

@app.get("/documents")
def get_documents():

    documents = []

    for file_path in DOCUMENT_FOLDER.iterdir():

        if not file_path.is_file():
            continue

        try:

            stored_name = file_path.name

            if "_" not in stored_name:
                continue

            document_id, filename = stored_name.split("_", 1)

            file_stats = file_path.stat()

            documents.append({

                "document_id": document_id,

                "filename": filename,

                "file_size": file_stats.st_size,

                "uploaded_at": datetime.fromtimestamp(
                    file_stats.st_mtime
                ).isoformat()

            })

        except Exception:
            continue

    return {

        "total_documents": len(documents),

        "documents": documents

    }


# ==========================================
# DELETE DOCUMENT
# ==========================================

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):

    if rag is None:

        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized"
        )

    try:

        collection = rag.retriever.vector_db._collection

        result = collection.get(
            where={
                "document_id": document_id
            }
        )

        ids = result.get("ids", [])

        if not ids:

            raise HTTPException(
                status_code=404,
                detail="Document not found in vector database"
            )

        collection.delete(ids=ids)

        deleted_files = 0

        for file_path in DOCUMENT_FOLDER.iterdir():

            if (
                file_path.is_file()
                and file_path.name.startswith(document_id + "_")
            ):

                file_path.unlink()

                deleted_files += 1

        return {

            "status": "success",

            "message": "Document deleted successfully",

            "document_id": document_id,

            "deleted_chunks": len(ids),

            "deleted_files": deleted_files

        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==========================================
# API INFORMATION
# ==========================================

@app.get("/api-info")
def api_information():

    return {

        "project": "Advanced RAG Pipeline",

        "version": "3.0",

        "features": [

            "Document Upload",

            "Duplicate Detection",

            "SHA-256 File Hashing",

            "Document Chunking",

            "Vector Database Storage",

            "Semantic Search",

            "Similarity Threshold Filtering",

            "Top-K Retrieval Control",

            "Context Optimization",

            "AI Answer Generation",

            "Source Citations",

            "Document Management"

        ]

    }