import os.path
from typing import Union
from setup_rag import *
from fastapi import FastAPI
from config import *
import argparse



def init_ragchain(debug=False):
    global chain
    config = get_config()
    if config["own_domain"]:
        path_index = config["path_index_own_domain"]
        path_dataset = config["path_dataset_own_domain"]
    else:
        path_index = config["path_index"]
        path_dataset = config["path_dataset"]




    print(f"loading  {path_index}")
    if not os.path.exists(path_index):
        create_index( path_dataset=path_dataset, path_index=path_index, own_domain=config["own_domain"])

    chain = create_rag_pipeline(path_index)
chain = None
init_ragchain()

app = FastAPI()

@app.get("/chat")
def read_item(q: Union[str, None] = None):

    result = chain({"query": q})

    print(f"Final Answer: {result['result']}")

    print("\n--- Retrieved Source Documents ---")
    for i, doc in enumerate(result['source_documents']):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")



    return {"answer": result['result']}



