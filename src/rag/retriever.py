# Implementation will be added step by step.
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


VECTOR_DB_DIR = Path("vector_db")


def load_vector_database():
    """Load the existing ChromaDB."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings
    )

    return vector_db


def retrieve_documents(query, k=3):
    """Retrieve the most relevant knowledge-base documents."""

    vector_db = load_vector_database()

    results = vector_db.similarity_search(
        query,
        k=k
    )

    return results


if __name__ == "__main__":

    query = """
    Customer is high risk, has a month-to-month contract,
    short tenure, high monthly charges, and does not have
    technical support or online security.
    What retention actions should be considered?
    """

    documents = retrieve_documents(query)

    print("=" * 60)
    print("RAG RETRIEVAL RESULTS")
    print("=" * 60)

    for i, document in enumerate(documents, start=1):

        print(f"\n--- Result {i} ---")

        print("Source:")
        print(
            document.metadata.get(
                "source",
                "Unknown"
            )
        )

        print("\nContent:")
        print(document.page_content)