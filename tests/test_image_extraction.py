import os.path
from unittest import TestCase
from scirag.parse_pdf import *
from scirag.config import *
from scirag.setup_rag import create_image_index, load_image_index


class TestImageExtraction(TestCase):

    def testFileDownload(self):
        conf = get_config()
        pages = download_files(Path(conf["path_dataset"]), Path(conf["path_dataset_cache"]))

    def testExtractContent(self):
        conf = get_config()
        path_data = conf["path_dataset_own_domain"]
        path_artifacts = conf["path_artifacts"]
        if not os.path.exists(path_artifacts):
            os.mkdir(path_artifacts)
        for file in os.listdir(path_data):
            if file.endswith(".pdf"):
                text, images = extract_content(path_data,  Path(file), path_artifacts)
                print(text)
                self.assertIsInstance(text, str)

    def test_image_indexing(self):
        conf = get_config()
        path_index_images = conf["path_index_images_own_domain"]
        path_artifacts = conf["path_artifacts"]
        path_dataset = conf["path_dataset_own_domain"]
        create_image_index(path_dataset, path_index_images, path_artifacts)
        self.assertFalse(False)

    def test_image_index_loading(self):
        conf = get_config()
        path_index_images = conf["path_index_images_own_domain"]

        return load_image_index(path_index_images)