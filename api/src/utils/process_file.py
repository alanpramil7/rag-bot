from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List


def process_file(file_path: str, file_extension: str) -> List[Document]:
    """
    Process different files and extract text content

    Args:
        file_path (str): PAth to file
        file_extension (str): Extension of file

    Return:
        str: Extracted text content
    """
    if file_extension == '.pdf':
        return _process_pdf(file_path)
    raise ValueError(f"Unsupported ectension: {file_extension}")


def _process_pdf(file_path: str) -> List[Document]:
    """Extract text from PDF"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
