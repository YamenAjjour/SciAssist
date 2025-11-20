import os
import time

import pandas as pd
from pypdf import PdfReader
from typing import List
from pathlib import Path
from config import *
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image, FigureCaption, Table
from tqdm import tqdm
from urllib.request import urlretrieve
def preprocess_and_clean_files(documents_path: Path) -> pd.DataFrame:

    documents = []
    for path_document in os.listdir(documents_path):
        if ".pdf" in path_document:
            abs_path_document = os.path.join(documents_path, path_document)
            reader = PdfReader(abs_path_document)
            document = ""
            for page in reader.pages:
                document += page.extract_text()
            documents.append(document)

    return pd.DataFrame({"full_text":documents})


def download_files(path_dataset: Path, path_dataset_cache: Path):
    df = pd.read_parquet(path_dataset)
    if not os.path.exists(path_dataset_cache):
        os.mkdir(path_dataset_cache)
    for i, rec in tqdm(df.iterrows()):
        if not rec["url"].endswith(".pdf"):
            url =  rec["url"] +".pdf"
        else:
            url =  rec["url"]
            file = path_dataset_cache / rec["acl_id"] + ".pdf"
            print(url)
            urlretrieve(url, file)
            time.sleep(1)


def extract_content(path_folder: Path, file_name: Path, path_image_output_dir: Path):
    """

    :param path_folder: the folder where the papers exist
    :param file_name:  the name of the paper
    :param path_image_output_dir: the location for storing the images
    :return:
    """
    path_file = path_folder / file_name
    print(path_file)

    path_file_image_output = str(path_image_output_dir /  file_name).replace(".pdf", "")
    if not os.path.exists(path_file_image_output):
        os.mkdir(path_file_image_output)

    raw_pdf_elements = partition_pdf( filename=path_file, extract_images_in_pdf=True, extract_image_block_output_dir=path_file_image_output,
        process_attachments=True, strategy="hi_res")
    images = []
    text = ""
    for i, element in enumerate(raw_pdf_elements):

        image_path = getattr(element.metadata, 'image_path', None)
        if isinstance(element, Image) or image_path:
            image_path = element.metadata.image_path
            caption = "No Caption Found"

            if i < len(raw_pdf_elements) and isinstance(raw_pdf_elements[i+1], FigureCaption):
                caption = raw_pdf_elements[i+1].text

            images.append({
                "image_path": f"{image_path}",
                "caption": caption,
                "element_id": element.id
            })
        elif isinstance(element, Table):
            continue
        else:
            text+= element.text
    return text, images

if __file__ == "__main__":
    conf = get_config()
    pages = download_files(Path(conf["path_dataset"]), Path(conf["path_dataset_cache"]))
