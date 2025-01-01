from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from src.utils.logger import log_time
from typing import List


@log_time
def load_website(url: str) -> List[Document]:
    """
        Parse the website and return list of documents

        Args:
            url: URL to be scraped
        """
    loader = WebBaseLoader(url)
    return loader.load()
