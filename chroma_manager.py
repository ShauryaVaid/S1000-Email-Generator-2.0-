"""ChromaDB integration for email context storage and retrieval with fail-safe initialization."""

import os
import shutil
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ChromaManager:
    """Manage ChromaDB for email context storage and retrieval."""

    def __init__(self, persist_directory: str, collection_name: str):
        """
        Initialize ChromaDB connection with self-healing fallback for schema issues.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        import chromadb

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        try:
            self._init_db()
        except Exception as e:
            logger.warning(f"ChromaDB init error ({e}), attempting self-healing re-initialization...")
            # If the database file is corrupted or from an older version, reset directory safely
            if os.path.exists(persist_directory):
                try:
                    shutil.rmtree(persist_directory)
                except Exception as del_err:
                    logger.error(f"Could not clear persist directory: {del_err}")
            self._init_db()

    def _init_db(self):
        """Internal initialization of Chroma client and collection."""
        import chromadb

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
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
        if not self.collection:
            return

        if metadata is None:
            metadata = {}

        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()

        try:
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[email_id]
            )
        except Exception as e:
            logger.warning(f"Failed to add document to ChromaDB: {e}")

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
        if not self.collection:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )

            formatted_results = []
            if results and results.get('documents'):
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                        'id': results['ids'][0][i] if results.get('ids') else '',
                        'distance': results['distances'][0][i] if results.get('distances') else None
                    })

            return formatted_results
        except Exception as e:
            logger.warning(f"ChromaDB search error: {e}")
            return []

    def get_email_by_id(self, email_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an email by its ID."""
        if not self.collection:
            return None

        try:
            results = self.collection.get(ids=[email_id])
            if results and results.get('documents'):
                return {
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0] if results.get('metadatas') else {},
                    'id': results['ids'][0]
                }
        except Exception:
            pass
        return None

    def delete_email(self, email_id: str) -> bool:
        """Delete an email from the collection."""
        if not self.collection:
            return False

        try:
            self.collection.delete(ids=[email_id])
            return True
        except Exception:
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if not self.collection:
            return {'total_emails': 0, 'collection_name': self.collection_name}

        try:
            count = self.collection.count()
            return {
                'total_emails': count,
                'collection_name': self.collection.name
            }
        except Exception:
            return {'total_emails': 0, 'collection_name': self.collection_name}
