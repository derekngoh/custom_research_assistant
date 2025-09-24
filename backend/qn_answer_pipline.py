from langchain.chains import RetrievalQAWithSourcesChain
import langchain

def get_answers_with_retrieval(query, llm, vector_store, debug=False):
    if debug:
        langchain.debug = True

    if vector_store is None:
        raise RuntimeError("003 Vector store not loaded properly")

    retriever = vector_store.as_retriever()
    if not retriever:
        raise RuntimeError("⚠️ retriever is None or invalid")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
    return chain({'question': query})
