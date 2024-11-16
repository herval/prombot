import json
from typing import Dict, List, Optional
import hashlib
import os
from datetime import datetime

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGManager:
    def __init__(self, data_dir="data", persist_dir="db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings()

        # Initialize two Chroma collections - one for documents, one for metadata
        self.docs_collection = self._init_collection("documents")
        self.metadata_collection = self._init_collection("metadata")

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

    def _init_collection(self, collection_name: str) -> Chroma:
        """Initialize a Chroma collection"""
        return Chroma(
            collection_name=collection_name,
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def _get_document_hash(self, identifier: str, content: str) -> str:
        """Create a hash from content and identifier"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{identifier}_{content_hash}"

    def is_document_processed(self, doc_hash: str) -> bool:
        """Check if document is already processed"""
        results = self.metadata_collection.get(
            where={"doc_hash": doc_hash},
            include=["metadatas"]
        )
        return len(results['ids']) > 0

    def _store_document_metadata(self,
                                 doc_hash: str,
                                 identifier: str,
                                 chunk_ids: List[str],
                                 content: str,
                                 metadata: Optional[Dict] = None):
        """Store document metadata in the metadata collection"""
        metadata_doc = Document(
            page_content=content,  # We store the full content here
            metadata={
                "doc_hash": doc_hash,
                "identifier": identifier,
                "chunk_ids": ",".join(chunk_ids),  # Convert list to string
                "processed_date": datetime.now().isoformat(),
                "original_metadata": json.dumps(metadata or {}),  # Serialize metadata dict
                "is_metadata_record": True
            }
        )

        self.metadata_collection.add_documents([metadata_doc])

    def get_document_by_identifier(self, identifier: str) -> Optional[Dict]:
        """Retrieve document information by identifier"""
        results = self.metadata_collection.get(
            where={
                "identifier": identifier,
                "is_metadata_record": True
            },
            include=["documents", "metadatas"]
        )

        if not results['ids']:
            return None

        metadata = results['metadatas'][0]
        return {
            "identifier": identifier,
            "content": results['documents'][0],
            "metadata": json.loads(metadata.get("original_metadata", "{}")),  # Deserialize metadata
            "processed_date": metadata.get("processed_date"),
            "doc_hash": metadata.get("doc_hash")
        }

    def add_document(self,
                     identifier: str,
                     content: str,
                     metadata: Optional[Dict] = None,
                     chunk_size: int = 1000,
                     chunk_overlap: int = 200) -> bool:
        """Add a single document to the vector store."""
        doc_hash = self._get_document_hash(identifier, content)

        if self.is_document_processed(doc_hash):
            print(f"Document {identifier} already processed, skipping...")
            return False

        full_metadata = {
            "source": identifier,
            "doc_hash": doc_hash,
            **(metadata or {})
        }

        doc = Document(page_content=content, metadata=full_metadata)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents([doc])

        chunk_ids = [hashlib.md5(f"{doc_hash}_{i}".encode()).hexdigest()
                     for i in range(len(texts))]

        for text, chunk_id in zip(texts, chunk_ids):
            text.metadata["chunk_id"] = chunk_id

        # Add document chunks to the docs collection
        self.docs_collection.add_documents(texts)

        # Store metadata
        self._store_document_metadata(
            doc_hash,
            identifier,
            chunk_ids,
            content,
            metadata=full_metadata
        )

        print(f"Successfully added document: {identifier}")
        return True

    def get_document_chunks(self, identifier: str) -> List[Dict]:
        """Retrieve all chunks for a specific document."""
        doc_info = self.get_document_by_identifier(identifier)
        if not doc_info:
            return []

        metadata = self.metadata_collection.get(
            where={"identifier": identifier, "is_metadata_record": True},
            include=["metadatas"]
        )

        if not metadata['ids']:
            return []

        # Split the chunk_ids string back into a list
        chunk_ids = metadata['metadatas'][0].get('chunk_ids', '').split(',')
        if not chunk_ids[0]:  # Handle empty string case
            return []

        # Get all chunks by their IDs from the docs collection
        chunks = []
        for chunk_id in chunk_ids:
            results = self.docs_collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )

            if results['ids']:
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": results['documents'][0],
                    "metadata": results['metadatas'][0]
                })

        return chunks

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search the vector store"""
        # Search only in the docs collection, not the metadata
        return self.docs_collection.similarity_search(
            query=query,
            k=k,
        )

    def load_documents(self) -> List:
        """Load documents from the data directory"""
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
                    filepath = doc.metadata.get('source', '')
                    identifier = os.path.basename(filepath)

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
