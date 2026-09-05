import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Must match Retrieval_pipeline.py. Runs fully offline after the first model download.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Create the local embedding model (no OpenAI credits required)."""
    print(f"Loading local embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads the model; later runs use the local cache.)")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )

# 1st Step: Load documents from the docs directory
def load_documents(docs_path="docs"):
    """Load all text files from the docs directory."""
    print(f"Loading documents from {docs_path}...")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory '{docs_path}' does not exist. Please create it and add your company files .")

    # Load all .txt files from the docs direqctory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    ) 

    # Load the documents
    documents = loader.load()

    # Check if any documents were loaded
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in the directory '{docs_path}'. Please add your company files .")

    # Print the number of documents loaded and their metadata
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"   Source: {doc.metadata['source']}")
        print(f"   Content length: {len(doc.page_content)} characters")
        print(f"   Content preview: {doc.page_content[:100]}...")  # Print first 100 characters
        print(f"   Metadata: {doc.metadata}")
    return documents

# 2nd Step: Split documents into smaller chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Spliting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # Print the number of chunks created and their metadata
    if chunks:
        # Print the first 5 chunks for inspection
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        # Print the number of additional chunks if there are more than 5
        if len(chunks) > 5:
            print(f"\n...and {len(chunks) - 5} more chunks.")

    return chunks

# 3rd Step: Create and persist ChromaDB vector store
def create_vectorstore(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embeddings = get_embeddings()

    # Create ChromaDB vector store
    print("--- Creating ChromaDB vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("---Finishedq Creating ChromaDB vector store ---")

    print(f"Vector store created and saved in '{persist_directory}' directory.")
    return vectorstore

# Main function to run the ingestion pipeline
def main():
    """ Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline === \n")

    # Define paths
    docs_path = "docs"
    persist_directory = "db/chroma_db"

    # Skip only if the store already has vectors (a failed run can leave an empty folder)
    if os.path.exists(persist_directory):
        embedding_model = get_embeddings()
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        existing_count = vectorstore._collection.count()
        if existing_count > 0:
            print(f"Vector store already exists in '{persist_directory}'. Skipping ingestion.")
            print(f"Loaded existing vector store with {existing_count} vectors.")
            return vectorstore
        print("Existing vector store is empty. Re-running ingestion...\n")
    else:
        print("Persisting directory does not exist. Proceeding with ingestion pipeline...\n")

    # Load documents
    documents = load_documents(docs_path)

    # Split documents into chunks
    chunks = split_documents(documents)

    # Create and persist ChromaDB vector store
    vectorstore = create_vectorstore(chunks, persist_directory)

    print("\n Ingestion pipeline completed successfully.")
    return vectorstore

if __name__ == "__main__":
    main()




# documents = [
#    Document(
#        page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#        metadata={'source': 'docs/google.txt'}
#    ),
#    Document(
#        page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#        metadata={'source': 'docs/microsoft.txt'}
#    ),
#    Document(
#        page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#        metadata={'source': 'docs/nvidia.txt'}
#    ),
#    Document(
#        page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#        metadata={'source': 'docs/spacex.txt'}
#    ),
#    Document(
#        page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#        metadata={'source': 'docs/tesla.txt'}
#    )
# ]