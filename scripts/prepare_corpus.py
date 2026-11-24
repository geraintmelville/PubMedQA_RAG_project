from src.processing.chunk import chunk_all
from src.processing.create_corpus import create_corpus
from src.processing.load import load_data


def main():
    labeled_records, unlabeled_records = load_data()
    create_corpus(labeled_records, unlabeled_records)
    chunk_all()


if __name__ == "__main__":
    main()
