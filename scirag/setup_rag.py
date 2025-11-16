import os.path

import numpy as np
import pandas as pd
import faiss
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import VLLM
from tqdm import tqdm
from argparse import *
load_dotenv()
from parse_pdf import *
from config import *


config = get_config()
#embedding_model_id = "gsarti/scibert-nli"
embedding_model_id = config["embedding_model_id"]
caption_embedding_model_id = config["caption_embedding_model_id"]

k = config["k"]
chunk_size = config["chunk_size"]

def create_gemini_llm():
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite", max_output_tokens=1024)
    return llm

def create_vllm(path_model: Path):
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

def create_image_index (path_dataset: Path, path_images_index: Path, path_artifacts: Path):

    def generate_image_stream():
        for file in os.listdir(path_dataset):
            if file.endswith(".pdf"):
                _, images = extract_content(path_dataset,  Path(file), path_artifacts)

                for image in images:
                    yield Document(page_content=image["caption"], metadata={"source" : file, "image_path": image["image_path"]})
    image_generator = generate_image_stream()
    embeddings = HuggingFaceEmbeddings(
        model_name=caption_embedding_model_id
    )

    vectorstore = FAISS.from_documents(list(image_generator), embeddings)
    vectorstore.save_local(str(path_images_index))

def load_image_index(path_image_index: Path):
    embeddings = HuggingFaceEmbeddings(model_name=caption_embedding_model_id)
    image_index = FAISS.load_local(path_image_index, embeddings, allow_dangerous_deserialization=True)
    return image_index


def create_index(path_dataset: Path, path_index: Path, path_artifacts: Path):
    print("creating index")
    def generate_documents_stream():

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        for file in os.listdir(path_dataset):
            if file.endswith(".pdf"):
                text, images = extract_content(path_dataset,  Path(file), path_artifacts)

                chunks = splitter.split_text(text)
                for chunk in chunks:
                    yield Document(page_content=chunk, metadata={"source" : file})

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
    #quantizer = faiss.IndexFlatL2(embedding_dimension)

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

    prompt = PromptTemplate(template="""Answer the following question based based on the following documents
     and use the figures to support your answer. Document :{context} Question {question}"""
                            , input_variables=["context", "question"])
    return prompt

def return_image_retrieval_prompt():
    prompt = PromptTemplate(template="Answer the following question based on the context. Context:{context} Question{question}", input_variables=["context", "question"])
    return prompt

def format_docs(docs):

    return "|".join(doc.metadata["image_path"]+"+"+doc.page_content for doc in docs)

def load_image_retriever(path_index: Path):

    embeddings_db = load_image_index(path_index)
    #retriever = embeddings_db.as_retriever(search_type="similarity_score_threshold",search_kwargs={'score_threshold': 0.000001})
    retriever = embeddings_db.as_retriever(search_kwargs={'k': 1})
    llm_compatible_chain=  retriever | RunnableLambda(lambda x :format_docs(x))         # Output: Single Context String

    return llm_compatible_chain

def load_text_retriever(path_index: Path):
    embeddings_db = load_index(path_index)
    retriever = embeddings_db.as_retriever(search_kwargs={"k": k})



    return retriever


def create_rag_pipeline(path_index: Path, path_image_index, path_model: Path = None):
    print("creating rag pipeline")
    image_retriever = load_image_retriever(path_image_index)
    text_retriever = load_text_retriever(path_index)
    if path_model:
        llm = create_vllm(path_model)
    else:
        llm = create_gemini_llm()

    prompt = return_prompt()

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=text_retriever, chain_type_kwargs={"prompt":prompt}, return_source_documents=True)
    concatenated_chain = RunnableParallel(
        output_a=image_retriever,
        output_b=chain
    )

    return concatenated_chain
def create_args():
    parser = ArgumentParser()
    parser.add_argument("--own-domain", action="store_true")
    parser.add_argument("--path-model", type=str)
    parser.add_argument("--path-dataset", type=str)
    parser.add_argument("--path-index", type=str, required=True)
    parser.add_argument("--path-image-index", type=str, required=True)
    parser.add_argument("--query", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = create_args()

    own_domain = args.own_domain
    path_model = args.path_model
    path_index = args.path_index
    path_image_index = args.path_image_index
    path_dataset = args.path_dataset
    path_artifacts = config["path_artifacts"]
    if not os.path.exists(path_index):
        create_index(path_dataset=path_dataset, path_index=path_index, path_artifacts=path_artifacts)

    if not os.path.exists(path_image_index):
        create_image_index(path_dataset=path_dataset, path_images_index=path_image_index, path_artifacts=path_artifacts)

    text_chain = create_rag_pipeline(path_index, path_image_index)

    if args.query:
            query = args.query
            answer = text_chain.stream( query)
            print(list(answer))
    else:
        while True:
            query = input("Enter query or exit to exit:")
            if query == "exit":
                break
            else:
                #print(answer)
                answers = text_chain.stream( query)

                #retrieved_docs = text_chain.get_relevant_documents({"query": query})
                # print(f"Retrieved {len(retrieved_docs)} documents:")
                # for i, doc in enumerate(retrieved_docs):
                #     print(f"\n--- Document {i+1} ---")
                #     print(f"Content: {doc.page_content}")
                #     print(f"Metadata: {doc.metadata}")
                    # print(answer["result"])

                for answer in list(answers):
                    for key, value in answer.items():
                        print(value)




