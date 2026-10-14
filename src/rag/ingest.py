from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


KNOWLEDGE_BASE_DIR = Path("knowledge_base")
VECTOR_DB_DIR = Path("vector_db")


def load_documents():
    """Load all TXT documents from the knowledge base."""

    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8"
        }
    )

    documents = loader.load()

    print(f"Documents loaded: {len(documents)}")

    return documents


def split_documents(documents):
    """Split documents into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Document chunks created: {len(chunks)}")

    return chunks


def create_vector_database(chunks):
    """Create and persist the Chroma vector database."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR)
    )

    print("ChromaDB created successfully.")
    print(f"Vector database location: {VECTOR_DB_DIR}")

    return vector_db


def main():

    print("=" * 60)
    print("RAG KNOWLEDGE BASE INGESTION")
    print("=" * 60)

    documents = load_documents()

    chunks = split_documents(
        documents
    )

    create_vector_database(
        chunks
    )

    print("\nRAG ingestion completed successfully.")


if __name__ == "__main__":
    main()