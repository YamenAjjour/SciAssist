from unittest import TestCase
from scirag.parse_pdf import preprocess_and_clean_files
from scirag.config import get_config
class TestDevDataset(TestCase):
    def test_dev_datas(self):
        config = get_config()
        df = preprocess_and_clean_files(config["path_dataset_debug"])
        print(len(df))


