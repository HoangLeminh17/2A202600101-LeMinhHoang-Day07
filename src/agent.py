from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        """Initialize with a store and an LLM generator function."""
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Answer a question by retrieving relevant context and generating a response.
        """
        # 1. Retrieve top-k relevant chunks
        results = self.store.search(question, top_k=top_k)
        
        # 2. Build context string
        context_parts = []
        for i, r in enumerate(results):
            content = r.get("content", "").strip()
            context_parts.append(f"Source {i+1}:\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # 3. Build prompt
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question below. "
            "If the answer is not in the context, say that you don't know based on the provided information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        # 4. Generate answer
        return self.llm_fn(prompt)
