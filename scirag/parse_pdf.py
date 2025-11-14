import os
import pandas as pd
from pypdf import PdfReader
from typing import List
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image, FigureCaption

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


def download_files(path_dataset, path_dataset_cache):
    df = pd.read_parquet(path_dataset)
    if not os.path.exists(path_dataset_cache):
        os.mkdir(path_dataset_cache)
    for i, rec in df.iterrows():
        if not rec["url"].endswith(".pdf"):
            url =  rec["url"] +".pdf"
        else:
            url =  rec["url"]
            file = path_dataset_cache + rec["acl_id"] + ".pdf"
            print(url)
            urlretrieve(url, file)


def extract_content(path_folder, file_name, path_image_output_dir):


    path_file = path_folder + file_name
    path_file_image_output = path_image_output_dir +  file_name
    raw_pdf_elements = partition_pdf(
        filename=path_file,
        extract_images_in_pdf=True,      # Crucial: enables image extraction
        image_output_dir_path=path_file_image_output,
        process_attachments=True,
        strategy="hi_res"
        # You can configure more options here (e.g., table extraction, chunk size)
    )





    extracted_data = []
    all_keys = []
    for i, element in enumerate(raw_pdf_elements):
        # Check if the element is an Image
        print(i)
        print(type(element))
        all_keys.extend(element.metadata.to_dict().keys())
        detection_class = getattr(element.metadata, 'detection_class', 'N/A')
        print(detection_class)
        #image_filename = getattr(element.metadata, 'filename', None)

        # Also check the image_path key, depending on your unstructured version/settings
        image_path = getattr(element.metadata, 'image_path', None)
        if isinstance(element, Image) or image_path:
            print("image")
            # The element.metadata.filename contains the local path to the extracted image
            image_path = element.metadata.image_path
            caption = "No Caption Found"

            # Look for the preceding text element as the potential caption
            # We look back one or two elements for a short piece of text (a caption)
            if i < len(raw_pdf_elements) and isinstance(raw_pdf_elements[i+1], FigureCaption):
                # Check if the preceding text is a reasonably short caption
                if len(raw_pdf_elements[i+1].text) < 500:
                    caption = raw_pdf_elements[i+1].text

            extracted_data.append({
                "image_path": f"{path_file_image_output}/{image_path}",
                "caption": caption,
                # 'id' is useful for the MultiVectorRetriever
                "element_id": element.id
            })
        if isinstance(element, FigureCaption):
            print(element.text)
    # 'extracted_data' now contains objects linking image paths and their potential captions.