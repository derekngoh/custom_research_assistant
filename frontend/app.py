from dotenv import load_dotenv
import sys, os
from pathlib import Path
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
print("ROOT: ", ROOT_DIR)
sys.path.append(str(ROOT_DIR))

import streamlit as st
from backend import file_loader, content_processor, qn_answer_pipline
from langchain_community.llms import OpenAI

VEC_STORE_DIR = Path(__file__).resolve().parent.parent / "vectorstore"

load_dotenv(dotenv_path=ROOT_DIR / ".env")

st.title("AI Research Assistant")
st.sidebar.title("News Article URLs")

urls = []

value = st.sidebar.slider("Model creativity. 0 Factual; 1 Most creative.", 0.0, 1.0, step=0.1)

for i in range(5):
    url = st.sidebar.text_input(f'URL {i+1}')
    url = file_loader.load_url(url)
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

if "vec_store" not in st.session_state:
    st.session_state.vec_store = None

if process_url_clicked:
    main_placeholder.text("Processing URLs...")
    data = file_loader.load_data_from_urls(urls)
    main_placeholder.text("Loading data from URLs...")
    chunked_data = content_processor.content_chunking(data, 1000, 100)
    main_placeholder.text("Divide data into chunks...")
    embeddings = content_processor.get_embeddings()
    main_placeholder.text("Loading embeddings...")
    st.session_state.vec_store = content_processor.create_faiss_vectorstore(chunked_data, embeddings)
    main_placeholder.text("Setting up vector store...")
    if st.session_state.vec_store is None:
        raise RuntimeError("002 Vector store not loaded properly")
    content_processor.save_faiss_vectorstore_components(st.session_state.vec_store, VEC_STORE_DIR)
    main_placeholder.text("Vector store is ready.")
    time.sleep(2)

query = main_placeholder.text_input("Question: ")

if query:
    main_placeholder.text("Setting up language model...")
    llm = OpenAI(temperature=value, max_tokens=1000)
    main_placeholder.text("Preparing response...")
    response = qn_answer_pipline.get_answers_with_retrieval(query, llm, st.session_state.vec_store, debug=False)
    main_placeholder.text("Response is ready.")
    st.write("Answer:", response["answer"])
    st.write("Sources:", response["sources"])

