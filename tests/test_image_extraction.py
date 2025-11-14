from unittest import TestCase
from scirag.parse_pdf import *
from scirag.config import *
class TestImageExtraction(TestCase):

    def testFileDownload(self):
        conf = get_config()
        pages = download_files(conf["path_dataset"], conf["path_dataset_cache"])