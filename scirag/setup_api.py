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
        path_image_index = config["path_index_images"]
        path_artifacts = config["path_artifacts"]
    else:
        path_index = config["path_index"]
        path_dataset = config["path_dataset"]
        path_image_index = config["path_images"]
        path_artifacts = config["path_artifacts"]




    print(f"loading  {path_index}")
    if not os.path.exists(path_index):
        create_index( path_dataset=path_dataset, path_index=path_index, path_artifacts=path_artifacts)
    print(f"loading  {path_image_index}")
    if not os.path.exists(path_image_index):
        create_image_index(path_dataset=path_dataset, path_images_index=path_image_index, path_artifacts=path_artifacts)

    chain = create_rag_pipeline(path_index, path_image_index)
chain = None
init_ragchain()

app = FastAPI()

@app.get("/chat")
def read_item(q: Union[str, None] = None):

    results = list(chain.stream(q))





    answer = {}
#    print(list(result))

    for result in results:

#        print(result)
        if "output_a" in result and result["output_a"]:
            answer["image_path"] = result["output_a"].split("+")[0]
            answer["image_caption"] = result["output_a"].split("+")[1]
        elif "output_b" in result:
            answer["result"] = result["output_b"]["result"]
    print(answer)
    return answer



