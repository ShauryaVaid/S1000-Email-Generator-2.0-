"""ChromaDB integration for email context storage and retrieval."""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from datetime import datetime


class ChromaManager:
    """Manage ChromaDB for email context storage and retrieval."""

    def __init__(self, persist_directory: str, collection_name: str):
        """
        Initialize ChromaDB connection.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_email(
        self,
        email_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an email to the collection.

        Args:
            email_id: Unique identifier for the email
            content: Email content to embed and store
            metadata: Optional metadata (sender, subject, date, etc.)
        """
        if metadata is None:
            metadata = {}

        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()

        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[email_id]
        )

    def search_similar(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar emails.

        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of matching emails with metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata
        )

        formatted_results = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'id': results['ids'][0][i] if results['ids'] else '',
                    'distance': results['distances'][0][i] if results['distances'] else None
                })

        return formatted_results

    def get_email_by_id(self, email_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an email by its ID.

        Args:
            email_id: Unique identifier for the email

        Returns:
            Email data or None if not found
        """
        results = self.collection.get(ids=[email_id])

        if results and results['documents']:
            return {
                'content': results['documents'][0],
                'metadata': results['metadatas'][0] if results['metadatas'] else {},
                'id': results['ids'][0]
            }
        return None

    def delete_email(self, email_id: str) -> bool:
        """
        Delete an email from the collection.

        Args:
            email_id: Unique identifier for the email

        Returns:
            True if deleted, False otherwise
        """
        try:
            self.collection.delete(ids=[email_id])
            return True
        except Exception:
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            'total_emails': count,
            'collection_name': self.collection.name
        }
