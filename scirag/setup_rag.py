import os.path

import numpy as np
import pandas as pd
import faiss
from langchain_community.vectorstores import FAISS

from pathlib import Path
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import VLLM
from tqdm import tqdm
from argparse import *
embedding_model_id = "BAAI/bge-small-en-v1.5"

def create_llm(path_model: Path):
    print("loading vllm")
    llm = VLLM(model=path_model,
               trust_remote_code=True,  # mandatory for hf models
               max_new_tokens=300,
               top_k=100,
               top_p=0.95,
               temperature=0.8,
               dtype="float32",

               # tensor_parallel_size=... # for distributed inference
               )

    return llm

def create_index(path_dataset: Path, path_index: Path, debug: bool):
    print("creating index")
    def generate_documents_stream():
        df = pd.read_parquet(path_dataset)
        if debug:
            df = df.sample(100)
        splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)

        for i, paper in tqdm(df.iterrows()):
            chunks = splitter.split_text(paper["full_text"])
            for chunk in chunks:
                yield Document(page_content=chunk, metadata={"source" : f"doc_{i}"})
    document_generator = generate_documents_stream()
    training_docs = []
    training_dataset_size = 10000
    m = 8
    batch_size= 1000
    bits = 8
    for _ in range(training_dataset_size):
        try:
            doc = next(document_generator)
            training_docs.append(doc.page_content)
        except StopIteration:
            break

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs = {'device': 'cuda'}
    )
    print("embedding for training")
    training_vectors = embeddings.embed_documents(training_docs)
    #embeddings_db = FAISS.from_documents(all_chunks, embeddings)
    training_vectors_np = np.array(training_vectors, dtype="float32")
    embedding_dimension = training_vectors_np.shape[1]
    assert embedding_dimension % m == 0
    quantizer = faiss.IndexFlatL2(embedding_dimension)

    # Create the IndexIVFPQ
    faiss.IndexPQ()

    faiss_index = faiss.IndexPQ(embedding_dimension, m, bits)
    print("training")
    faiss_index.train(training_vectors_np)
    vectorstore = FAISS(
        embedding_function=embeddings, # Still needed for query embedding
        index=faiss_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    document_generator = generate_documents_stream()
    current_batch = []

    print("indexing")
    for doc in document_generator:

        current_batch.append(doc.page_content)
        if len(current_batch) >= batch_size:
            batch_embeddings = embeddings.embed_documents(current_batch)
            embeddings_and_doc = zip(current_batch, batch_embeddings)
            print(embeddings_and_doc)
            vectorstore.add_embeddings(embeddings_and_doc)
            current_batch = []
    vectorstore.save_local(str(path_index))

def load_index(path_index: Path):
    print(f"loading index from {path_index}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id,)
    embeddings_db = FAISS.load_local(path_index, embeddings, allow_dangerous_deserialization=True)
    num_documents = len(embeddings_db.index_to_docstore_id)
    print(f"Total number of documents: {num_documents}")
    return embeddings_db

def return_prompt():
    prompt_template = """<|system|>As a helpful scientific assistants, please answer the question below, focusing on numerical data and using only the context below.
    Don't invent facts. If you can't provide a factual answer, say you don't know what the answer is.
    <|user|>
    question: {question}
    context: {context}
    <assistant>:
    """


    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt

def create_rag_pipeline(path_index: Path, path_model: Path, debug:bool=False):
    print("creating rag pipeline")
    llm = create_llm(path_model)
    embeddings_db = load_index(path_index)
    prompt = return_prompt()
    retriever = embeddings_db.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt":prompt})
    return chain

def create_args():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--path-model", type=str, required=True)
    parser.add_argument("--path-dataset", type=str)
    parser.add_argument("--path-index", type=str, required=True)
    parser.add_argument("--query", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = create_args()
    debug = args.debug
    path_model = args.path_model
    path_index = args.path_index
    path_dataset = args.path_dataset

    if not os.path.exists(path_index):
        create_index(path_dataset=path_dataset, path_index=path_index, debug=debug)

    chain = create_rag_pipeline(path_index, path_model, debug)
    if args.query:
        query = args.query
        answer = chain.run({"query": query})
        answer = answer.split("<｜Assistant｜>")[1]
        print(answer)
    else:
        while True:
            query = input("Enter query or exit to exit:")
            if query == "exit":
                break
            else:
                print(answer)
                answer = chain.run({"query": query})
                answer = answer.split("<｜Assistant｜>")[1]
                print(answer)




