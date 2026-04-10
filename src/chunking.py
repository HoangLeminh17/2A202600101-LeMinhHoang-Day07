from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
            
        # Split on .!? followed by whitespace, maintaining the punctuation
        # We use a non-capturing lookbehind to avoid consuming the punctuation if we split
        # but a simple way is re.split with capturing groups and then merging
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i : i + self.max_sentences_per_chunk]
            chunk_text = " ".join(s.strip() for s in chunk_sentences if s.strip())
            if chunk_text:
                chunks.append(chunk_text)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        """Recursively split text using separators in priority order."""
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        """Recursive helper used by RecursiveChunker.chunk."""
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            # Fallback to fixed size if no separators left
            return FixedSizeChunker(self.chunk_size).chunk(current_text)

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]
        
        # Split by current separator
        if sep == "":
            splits = list(current_text)
        else:
            splits = current_text.split(sep)

        final_chunks = []
        current_chunk_parts = []
        current_chunk_len = 0

        for s in splits:
            # Add back the separator if it wasn't the last empty string case
            # Note: actual implementations often handle joining differently, 
            # but we'll try to keep it simple while staying under limit.
            part = s + (sep if sep != "" else "")
            
            if len(part) > self.chunk_size:
                # If we have stuff in the current buffer, flush it
                if current_chunk_parts:
                    final_chunks.append("".join(current_chunk_parts).rstrip(sep))
                    current_chunk_parts = []
                    current_chunk_len = 0
                
                # Recurse on the oversized part
                final_chunks.extend(self._split(s, next_seps))
            else:
                if current_chunk_len + len(part) <= self.chunk_size:
                    current_chunk_parts.append(part)
                    current_chunk_len += len(part)
                else:
                    if current_chunk_parts:
                        final_chunks.append("".join(current_chunk_parts).rstrip(sep))
                    current_chunk_parts = [part]
                    current_chunk_len = len(part)

        if current_chunk_parts:
            final_chunks.append("".join(current_chunk_parts).rstrip(sep))

        return [c for c in final_chunks if c.strip()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        """Run all built-in chunking strategies and compare their results."""
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        
        comparison = {}
        for name, strategy in strategies.items():
            chunks = strategy.chunk(text)
            if not chunks:
                comparison[name] = {"count": 0, "avg_length": 0, "max_len": 0, "min_len": 0}
                continue
                
            lengths = [len(c) for c in chunks]
            comparison[name] = {
                "count": len(chunks),
                "avg_length": sum(lengths) / len(chunks),
                "max_len": max(lengths),
                "min_len": min(lengths),
                "chunks": chunks[:3] # Sample of first 3 chunks
            }
        return comparison
