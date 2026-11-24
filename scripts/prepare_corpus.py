import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.processing.chunk_documents import chunk_documents
from src.processing.chunk_sections import chunk_sections
from src.processing.chunk_sentences import chunk_sentences
from src.processing.create_corpus import create_corpus
from src.processing.load import load_data


def main():
    labeled_records, unlabeled_records = load_data()
    create_corpus(labeled_records, unlabeled_records)
    chunk_documents()
    chunk_sections()
    chunk_sentences()


if __name__ == "__main__":
    main()
