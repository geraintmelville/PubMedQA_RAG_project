import json
from urllib.request import urlretrieve

from src.common import LABELED_DATA_PATH, RAW_DIR, UNLABELED_DATA_PATH

LABELED_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
)
UNLABELED_URL = (
    "https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/"
    "3b487bf64f698d2211163fb64cde6b102cd29bd2/data/ori_pqau.json"
)


def load_data():
    """Download the source files and load their records."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    labeled_path = LABELED_DATA_PATH
    unlabeled_path = UNLABELED_DATA_PATH

    urlretrieve(LABELED_URL, labeled_path)
    urlretrieve(UNLABELED_URL, unlabeled_path)

    with labeled_path.open(encoding="utf-8") as input_file:
        labeled_records = json.load(input_file)
    with unlabeled_path.open(encoding="utf-8") as input_file:
        unlabeled_records = json.load(input_file)

    print(f"Downloaded and loaded PubMedQA-L and PubMedQA-U from {RAW_DIR}")
    return labeled_records, unlabeled_records
