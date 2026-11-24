import json
from urllib.request import urlretrieve

from src.common import LABELED_DATA_PATH, RAW_DIR, UNLABELED_DATA_PATH

LABELED_URL = (
    "https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/"
    "3b487bf64f698d2211163fb64cde6b102cd29bd2/data/ori_pqal.json"
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

    if not labeled_path.exists():
        _download(LABELED_URL, labeled_path)
    if not unlabeled_path.exists():
        _download(UNLABELED_URL, unlabeled_path)

    with labeled_path.open(encoding="utf-8") as input_file:
        labeled_records = json.load(input_file)
    with unlabeled_path.open(encoding="utf-8") as input_file:
        unlabeled_records = json.load(input_file)

    print(f"Loaded PubMedQA-L and PubMedQA-U from {RAW_DIR}")
    return labeled_records, unlabeled_records


def _download(url, path):
    """Download to a temporary file and publish it only when complete."""
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        urlretrieve(url, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
