import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader, UnstructuredMarkdownLoader
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentTracker:
    def __init__(self, tracking_file="processed_docs.json"):
        self.tracking_file = tracking_file
        self.processed_docs = self._load_tracking_data()

    def _load_tracking_data(self) -> Dict:
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_tracking_data(self):
        with open(self.tracking_file, 'w') as f:
            json.dump(self.processed_docs, f, indent=2)

    def get_document_hash(self, identifier: str, content: str) -> str:
        """Create a hash from content and identifier"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{identifier}_{content_hash}"

    def is_document_processed(self, doc_hash: str) -> bool:
        return doc_hash in self.processed_docs

    def mark_document_processed(self, doc_hash: str, identifier: str, chunk_ids: List[str], content: str, metadata: Optional[Dict] = None):
        self.processed_docs[doc_hash] = {
            "identifier": identifier,
            "processed_date": datetime.now().isoformat(),
            "chunk_ids": chunk_ids,
            "content": content,  # Store original content
            "metadata": metadata or {}
        }
        self._save_tracking_data()

    def get_document_by_identifier(self, identifier: str) -> Optional[Dict]:
        """Retrieve document information by identifier"""
        for doc_hash, doc_info in self.processed_docs.items():
            if doc_info["identifier"] == identifier:
                return {
                    "doc_hash": doc_hash,
                    **doc_info
                }
        return None

class RAGManager:
    def __init__(self, data_dir="data", persist_dir="db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.doc_tracker = DocumentTracker()

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

    def load_documents(self) -> List:
        """Load documents from the data directory, skipping already processed ones"""
        loaders = {
            ".txt": (DirectoryLoader, {"glob": "**/*.txt", "loader_cls": TextLoader}),
            ".md": (DirectoryLoader, {"glob": "**/*.md", "loader_cls": UnstructuredMarkdownLoader}),
            ".pdf": (DirectoryLoader, {"glob": "**/*.pdf", "loader_cls": PyPDFLoader})
        }

        new_documents = []
        for ext, (loader_cls, loader_args) in loaders.items():
            matching_files = [f for f in os.listdir(self.data_dir) if f.endswith(ext)]
            if matching_files:
                loader = loader_cls(self.data_dir, **loader_args)
                documents = loader.load()

                for doc in documents:
                    # Use filename as identifier
                    filepath = doc.metadata.get('source', '')
                    identifier = os.path.basename(filepath)

                    # Process document using add_document method
                    was_added = self.add_document(
                        identifier=identifier,
                        content=doc.page_content,
                        metadata={
                            "filepath": filepath,
                            "file_type": ext,
                            **doc.metadata
                        }
                    )

                    if was_added:
                        new_documents.append(doc)

        print(f"Found {len(new_documents)} new documents to process")
        return new_documents

    def add_document(self,
                     identifier: str,
                     content: str,
                     metadata: Optional[Dict] = None,
                     chunk_size: int = 1000,
                     chunk_overlap: int = 200) -> bool:
        """
        Add a single document to the vector store.

        Args:
            identifier (str): Unique identifier for the document
            content (str): The text content of the document
            metadata (dict, optional): Additional metadata for the document
            chunk_size (int): Size of text chunks for splitting
            chunk_overlap (int): Overlap between chunks

        Returns:
            bool: True if document was added, False if it was already processed
        """
        # Create document hash
        doc_hash = self.doc_tracker.get_document_hash(identifier, content)

        # Check if already processed
        if self.doc_tracker.is_document_processed(doc_hash):
            print(f"Document {identifier} already processed, skipping...")
            return False

        # Create metadata
        full_metadata = {
            "source": identifier,
            "doc_hash": doc_hash,
            **(metadata or {})
        }

        # Create langchain document
        doc = Document(page_content=content, metadata=full_metadata)

        # Split document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents([doc])

        # Generate chunk IDs
        chunk_ids = [hashlib.md5(f"{doc_hash}_{i}".encode()).hexdigest()
                     for i in range(len(texts))]

        # Add chunk IDs to metadata
        for text, chunk_id in zip(texts, chunk_ids):
            text.metadata["chunk_id"] = chunk_id

        # Add to vector store
        existing_store = self.load_existing_vector_store()

        if existing_store:
            existing_store.add_documents(texts)
            self.vector_store = existing_store
        else:
            self.vector_store = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )

        # Mark as processed with chunk IDs
        self.doc_tracker.mark_document_processed(
            doc_hash,
            identifier,
            chunk_ids,
            content,
            metadata=full_metadata
        )

        print(f"Successfully added document: {identifier}")
        return True

    def get_document(self, identifier: str) -> Optional[Dict]:
        """
        Retrieve a document by its identifier.

        Args:
            identifier (str): The unique identifier of the document

        Returns:
            Optional[Dict]: Document information including content, metadata, and chunks
        """
        # Get document info from tracker
        doc_info = self.doc_tracker.get_document_by_identifier(identifier)
        if not doc_info:
            return None

        return {
            "identifier": identifier,
            "content": doc_info["content"],
            "metadata": doc_info["metadata"],
            "processed_date": doc_info["processed_date"],
            "doc_hash": doc_info["doc_hash"]
        }

    def get_document_chunks(self, identifier: str) -> List[Dict]:
        """
        Retrieve all chunks for a specific document.

        Args:
            identifier (str): The unique identifier of the document

        Returns:
            List[Dict]: List of chunks with their content and metadata
        """
        doc_info = self.doc_tracker.get_document_by_identifier(identifier)
        if not doc_info or not self.vector_store:
            return []

        # Use Chroma's get() method to retrieve chunks by their IDs
        chunk_docs = self.vector_store.get(
            ids=doc_info["chunk_ids"],
            include=["documents", "metadatas"]
        )

        chunks = []
        for i, (doc, metadata) in enumerate(zip(chunk_docs["documents"], chunk_docs["metadatas"])):
            chunks.append({
                "chunk_id": doc_info["chunk_ids"][i],
                "content": doc,
                "metadata": metadata
            })

        return chunks

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search the vector store"""
        if not self.vector_store:
            self.load_existing_vector_store()

        if not self.vector_store:
            raise ValueError("No vector store available. Please create one first.")

        return self.vector_store.similarity_search(query, k=k)

    def load_existing_vector_store(self):
        """Load existing vector store"""
        if os.path.exists(self.persist_dir):
            self.vector_store = Chroma(
                collection_name="documents",
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            return self.vector_store
        return None
