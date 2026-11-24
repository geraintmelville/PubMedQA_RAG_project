from src.common import CHUNK_PATHS
from src.processing.chunk import chunk
from src.processing.create_corpus import create_corpus
from src.processing.load import load_data


def main():
    labeled_records, unlabeled_records = load_data()
    create_corpus(labeled_records, unlabeled_records)
    for name in CHUNK_PATHS:
        chunk(name)


if __name__ == "__main__":
    main()
