import os.path
from typing import Union
from setup_rag import *
from fastapi import FastAPI

print(f"Hello")



def init_ragchain():
    global chain
    path = os.path.dirname(os.path.realpath(__file__))

    path_index = f"{path}/../data/index"
    path_dataset = f"{path}/../data/acl-publication-info.74k.parquet"
    if os.path.exists("/bigwork/nhwpajjy/pre-trained-models"):
        path_model = "/bigwork/nhwpajjy/pre-trained-models/TinyLlama-1.1B-Chat-v1.0"
    elif os.path.exists("/mnt/home/yajjour/pre-trained-models"):
        path_model ="/mnt/home/yajjour/pre-trained-models/TinyLlama-1.1B-Chat-v1.0"
    else:
        path_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"loading {path_model} and {path_index}")
    if not os.path.exists(path_index):
        create_index( path_dataset=path_dataset, path_index=path_index, debug=False)

    chain = create_rag_pipeline(path_index, path_model, True)

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
