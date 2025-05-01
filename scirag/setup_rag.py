import os.path

import pandas as pd

from langchain.vectorstores import FAISS

from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import VLLM
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
               dtype="float16",

               # tensor_parallel_size=... # for distributed inference
               )

    return llm

def create_index(path_dataset: Path, path_index: Path, debug: bool):
    print("creating index")
    df = pd.read_parquet(path_dataset)
    if debug:
        df = df.sample(100)
    df.sample(10000)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1028, chunk_overlap=0)
    all_chunks = []
    for _, paper in tqdm(df.iterrows()):
        chunks = splitter.split_text(paper["full_text"])
        all_chunks += splitter.create_documents(chunks)



    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs = {'device': 'cuda'}
    )
    embeddings_db = FAISS.from_documents(all_chunks, embeddings)
    embeddings_db.save_local(path_index)

def load_index(path_index: Path):
    print(f"loading index from {path_index}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
    )
    embeddings_db = FAISS.load_local(path_index, embeddings, allow_dangerous_deserialization=True)
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
    retriever = embeddings_db.as_retriever(search_kwargs={"k": 10})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt":prompt})
    return chain

def create_args():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--path-model", type=str, required=True)
    parser.add_argument("--path-dataset", type=str)
    parser.add_argument("--path-index", type=str, required=True)

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

    while True:
        query = input("Enter query or exit to exit:")
        if query == "exit":
            break
        else:
            answer = chain.run({"query": query})
            answer = answer.split("<｜Assistant｜>")[1]
            print(answer)




