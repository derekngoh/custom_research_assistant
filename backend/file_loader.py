from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredURLLoader
from pathlib import Path
from urllib.parse import urlparse

def load_file(file_path):
    extension = Path(file_path).suffix

    if extension == ".txt":
        loader = TextLoader(file_path)
        return loader.load()

    elif extension == ".csv":
        loader = CSVLoader(file_path)
        return loader.load()

    else:
        raise Exception(f"File extension {extension} is not supported.")

def load_url(url):
    if urlparse(url):
        return url
    else:
        raise Exception(f"URL {url} is not supported.")

def load_data_from_urls(urls: list):
    loader = UnstructuredURLLoader(urls)
    return loader.load()