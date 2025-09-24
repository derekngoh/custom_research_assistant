import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores import FAISS

import faiss
import pickle

def content_chunking(documents, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", ".", " "],
        chunk_size = chunk_size,
        chunk_overlap = overlap,
    )
    return splitter.split_documents(documents)

def get_embeddings(openai=True):
    if openai:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Did you load the .env file?")
        return OpenAIEmbeddings()
    else:
        return SentenceTransformer("all-mpnet-base-v2")

def create_faiss_vectorstore(docs, embedding):
    if not embedding:
        raise ValueError("Embeddings list is empty")
    return FAISS.from_documents(docs, embedding)

def save_faiss_vectorstore_components(vector_store, vec_store_dir):
    os.makedirs(vec_store_dir, exist_ok=True)

    faiss.write_index(vector_store.index, os.path.join(vec_store_dir, "index.faiss"))

    with open(os.path.join(vec_store_dir, "documents.pkl"), "wb") as f:
        pickle.dump(vector_store.docstore._dict, f)

    with open(os.path.join(vec_store_dir, "index_map.pkl"), "wb") as f:
        pickle.dump(vector_store.index_to_docstore_id, f)


def load_faiss_vectorstore_components(vec_store_dir, embedding_fn):
    index = faiss.read_index(os.path.join(vec_store_dir, "index.faiss"))

    with open(os.path.join(vec_store_dir, "documents.pkl"), "rb") as f:
        documents_dict = pickle.load(f)
    docstore = InMemoryDocstore(documents_dict)

    with open(os.path.join(vec_store_dir, "index_map.pkl"), "rb") as f:
        index_map = pickle.load(f)

    return FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_map,
        embedding_function=embedding_fn,
    )


