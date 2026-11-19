import os
from pathlib import Path

from pypdf import PdfReader
from docx import Document


def load_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text() or ""

        text += f"\n[Page {page_number}]\n"
        text += page_text

    return text


def load_txt(file_path):

    with open(file_path, "r", encoding="utf-8") as file:

        return file.read()


def load_docx(file_path):

    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )

    return text


def load_document(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":

        return load_pdf(file_path)

    elif extension == ".txt":

        return load_txt(file_path)

    elif extension == ".docx":

        return load_docx(file_path)

    else:

        raise ValueError(
            f"Unsupported file format: {extension}"
        )


def load_all_documents(folder_path):

    documents = []

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):

            try:

                text = load_document(file_path)

                documents.append({
                    "filename": filename,
                    "content": text
                })

                print(f"Loaded: {filename}")

            except Exception as error:

                print(f"Error loading {filename}: {error}")

    return documents