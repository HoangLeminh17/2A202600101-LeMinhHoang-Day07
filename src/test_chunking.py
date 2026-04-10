import os
from src.chunking import SentenceChunker, RecursiveChunker, ChunkingStrategyComparator

class SoftwareEngineeringChunker(RecursiveChunker):
    """
    Custom strategy for Software Engineering (SE) domain.
    Optimized to keep technical hierarchies (headers) and code blocks together.
    """
    SE_SEPARATORS = [
        "\n# ",   # H1
        "\n## ",  # H2
        "\n### ", # H3
        "\n\n",   # Double newline (Paragraphs)
        "\n",     # Single newline (Line breaks)
        ". ",     # Sentence end
        " ",      # Word
        ""        # Character (fallback)
    ]

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        super().__init__(separators=self.SE_SEPARATORS, chunk_size=chunk_size)

def test_chunking_on_book():
    file_path = "data/book.md"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    full_text = text
    chunk_size = 500 # Default for baseline
    
    print(f"--- Full Chunking Analysis on {file_path} ---")
    
    comparator = ChunkingStrategyComparator()
    report = comparator.compare(full_text, chunk_size=chunk_size)
    
    # Run Custom Chunker (SE optimized)
    se_chunker = SoftwareEngineeringChunker(chunk_size=800)
    se_chunks = se_chunker.chunk(full_text)
    se_lengths = [len(c) for c in se_chunks]
    report["custom_se"] = {
        "count": len(se_chunks),
        "avg_length": sum(se_lengths) / len(se_chunks) if se_chunks else 0,
        "max_len": max(se_lengths) if se_lengths else 0,
        "min_len": min(se_lengths) if se_lengths else 0,
        "chunks": se_chunks[:3] # Top-3 chunks
    }
    
    for strategy, stats in report.items():
        print(f"\nStrategy: {strategy}")
        print(f"  Count: {stats['count']}")
        print(f"  Avg Length: {stats['avg_length']:.2f}")
        print(f"  Max/Min: {stats['max_len']}/{stats['min_len']}")
        
        print("  Sample Chunks:")
        for i, c in enumerate(stats['chunks'][:2]):
            display_text = c.replace('\n', '\\n')
            preview = display_text if len(display_text) < 120 else display_text[:117] + "..."
            print(f"    [{i}] {preview}")

if __name__ == "__main__":
    test_chunking_on_book()
