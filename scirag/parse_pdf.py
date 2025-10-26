import os
import pandas as pd
from pypdf import PdfReader
from typing import List
from pathlib import Path

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

