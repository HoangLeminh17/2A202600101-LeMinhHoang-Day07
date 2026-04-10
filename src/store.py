from __future__ import annotations

from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            from chromadb.config import Settings
            self._client = chromadb.Client(Settings(anonymized_telemetry=False))
            # Clear if already exists to ensure fresh tests
            try:
                self._client.delete_collection(name=collection_name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True

        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: Build a normalized stored record for one document.
        embedding = self._embedding_fn(doc.content)
        record = {
            "id": f"{self._collection_name}_{self._next_index}",
            "content": doc.content,
            "embedding": embedding,
            "metadata": doc.metadata or {},
        }
        self._next_index += 1
        return record
    
    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        query_emb = self._embedding_fn(query)
        scored = []
        for r in records:
            score = compute_similarity(query_emb, r["embedding"])
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**r, "score": score} for score, r in scored[:top_k]]

    def add_documents(self, docs: list[Document]) -> None:
        """Embed each document's content and store it."""
        if self._use_chroma:
            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for doc in docs:
                emb = self._embedding_fn(doc.content)
                doc_id = f"{self._collection_name}_{self._next_index}"
                self._next_index += 1

                ids.append(doc_id)
                documents.append(doc.content)
                embeddings.append(emb)
                # Ensure metadata includes doc_id for later deletion
                meta = (doc.metadata or {}).copy()
                if "doc_id" not in meta:
                    meta["doc_id"] = doc.id
                metadatas.append(meta)

            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

        else:
            for doc in docs:
                record = self._make_record(doc)
                # Ensure metadata includes doc_id for later deletion
                if "doc_id" not in record["metadata"]:
                    record["metadata"]["doc_id"] = doc.id
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents to query."""
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k
            )
            output = []
            if results["documents"]:
                for i in range(len(results["documents"][0])):
                    output.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 / (1.0 + (results["distances"][0][i] if results.get("distances") else 0.0))
                    })
            return output

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """Search with optional metadata pre-filtering."""
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=metadata_filter
            )
            output = []
            if results["documents"]:
                for i in range(len(results["documents"][0])):
                    output.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 / (1.0 + (results["distances"][0][i] if results.get("distances") else 0.0))
                    })
            return output

        # In-memory filtering
        filtered_records = self._store
        if metadata_filter:
            filtered_records = []
            for r in self._store:
                match = True
                for k, v in metadata_filter.items():
                    if r["metadata"].get(k) != v:
                        match = False
                        break
                if match:
                    filtered_records.append(r)
        
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.
        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            before = self.get_collection_size()
            self._collection.delete(where={"doc_id": doc_id})
            after = self.get_collection_size()
            return before > after

        initial_len = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        return len(self._store) < initial_len
