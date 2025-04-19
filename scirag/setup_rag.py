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

embedding_model_id = "BAAI/bge-small-en-v1.5"

def create_llm(path_model: Path):
    print("loading vllm")
    llm = VLLM(model=path_model,
               trust_remote_code=True,  # mandatory for hf models
               max_new_tokens=10,
               top_k=10,
               top_p=0.95,
               temperature=0.8,
               # tensor_parallel_size=... # for distributed inference
               )

    return llm

def create_index(path_dataset: Path, path_index: Path):
    print("creating index")
    df = pd.read_parquet(path_dataset)
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    all_chunks = []
    for _, paper in tqdm(df.iterrows()):
        chunks = splitter.split_text(paper["full_text"])
        all_chunks += splitter.create_documents(chunks)



    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
    )
    embeddings_db = FAISS.from_documents(all_chunks, embeddings)
    embeddings_db.save_local(path_index)

def load_index(path_index: Path):
    print("loading index")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
    )
    embeddings_db = FAISS.load_local(path_index, embeddings)
    return embeddings_db

def return_prompt():
    prompt_template = """<｜begin▁of▁sentence｜>
    As a helpful scientific assistants, please answer the question below, focusing on numerical data and using only the context below.
    Don't invent facts. If you can't provide a factual answer, say you don't know what the answer is.
    <｜User｜>
    question: {question}
    
    context: {context}
    <｜Assistant｜>
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt

def create_rag_pipeline(path_index: Path, path_model: Path):
    print("creating rag pipeline")
    llm = create_llm(path_model)
    embeddings_db = load_index(path_index)
    prompt = return_prompt()
    retriever = embeddings_db.as_retriever(search_kwargs={"k": 10})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt":prompt})
    return chain


if __name__ == "__main__":
    path_index = "data/index"
    path_dataset = "data/acl-publication-info.74k.parquet"
    path_model = "/bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen-1.5B"

    if not os.path.exists(path_index):
        create_index( path_dataset=path_dataset, path_index=path_index)

    chain = create_rag_pipeline(path_dataset, path_model)

    while True:
        query = input("Enter query or exit to exit:")
        if query == "exit":
            break
        else:
            answer = chain.run({"query": query})
            answer = answer.split("<｜Assistant｜>")[1]
            print(answer)




