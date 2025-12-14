import os
# Disable tokenizer parallelism warning (must be set before importing SentenceTransformer)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import chromadb


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "RAG document collection",
                "hnsw:space": "cosine",
                "hnsw:batch_size": 10000,
            },
        )

        print(
            f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # Handle empty or very short text
        if not text or len(text.strip()) == 0:
            return []
        
        # Ensure chunk_size is valid
        if chunk_size <= 0:
            chunk_size = 500

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """

        print(f"Processing {len(documents)} documents...")

        for doc_index, document in enumerate(documents):
            # Normalize input (support both dicts and raw strings)
            if isinstance(document, str):
                document = {"content": document,
                            "metadata": {"source": "raw_input"}}

            if not isinstance(document, dict):
                raise ValueError(
                    f"Invalid document format at index {doc_index}: {document}")

            content = document.get("content", "")
            
            # Skip empty documents
            if not content or len(content.strip()) == 0:
                print(f"Document {doc_index}: Skipping empty document")
                continue
            
            # Ensure metadata exists and is not empty
            metadata = document.get("metadata") or {
                "source": f"doc_{doc_index}"}
            
    
            chunks = self.chunk_text(content)

            print(
                f"Document {doc_index}: Split into {len(chunks)} chunks.")

            # Skip if no chunks were created
            if len(chunks) == 0:
                print(f"Document {doc_index}: No chunks created, skipping")
                continue

            chunk_ids = [
                f"doc_{doc_index}_chunk_{i}" for i in range(len(chunks))
            ]
            embeddings = self.embedding_model.encode(chunks).tolist()
            # Attach metadata to every chunk
            metadatas = [{**metadata, "chunk_index": i}
                         for i in range(len(chunks))]

            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """

        print(f"Searching for top {n_results} results for query: {query}")

        # Embedding the query using the same model used for documents
        print("Generating query embedding...")
        query_embedding = self.embedding_model.encode(
            [query]).tolist()[0]  # get the first (and only) embedding

        print("Querying collection...")
        print('\n')
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["distances", "metadatas", "documents"],
        )

        # Format results - handle nested lists from ChromaDB
        documents_list = results["documents"][0] if results["documents"] and isinstance(results["documents"][0], list) else results["documents"]
        metadatas_list = results["metadatas"][0] if results["metadatas"] and isinstance(results["metadatas"][0], list) else results["metadatas"]
        distances_list = results["distances"][0] if results["distances"] and isinstance(results["distances"][0], list) else results["distances"]
        ids_list = results["ids"][0] if results["ids"] and isinstance(results["ids"][0], list) else results["ids"]

        return {
            "documents": documents_list,
            "metadatas": metadatas_list,
            "distances": distances_list,
            "ids": ids_list
        }
