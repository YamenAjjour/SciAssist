from typing import Union
from setup_rag import *
from fastapi import FastAPI






def init_ragchain():
    global chain
    path_index = "data/index"
    path_dataset = "data/acl-publication-info.74k.parquet"
    path_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if not os.path.exists(path_index):
        create_index( path_dataset=path_dataset, path_index=path_index)

    chain = create_rag_pipeline(path_dataset, path_model)

chain = None
init_ragchain()

app = FastAPI()

@app.get("/chat")
def read_item(q: Union[str, None] = None):
    if not chain:
        return {"answer": "call /"}
    answer = chain.run({"query": q})
    answer = answer.split("<｜Assistant｜>")[1]
    return {"answer": answer}
