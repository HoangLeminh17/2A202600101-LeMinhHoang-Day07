from src.chunking import RecursiveChunker, compute_similarity
from src.models import Document
from src.store import EmbeddingStore
from src.embeddings import _mock_embed

# Define custom strategy here as requested not to modify src/chunking.py
class SoftwareEngineeringChunker(RecursiveChunker):
    SE_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        super().__init__(separators=self.SE_SEPARATORS, chunk_size=chunk_size)

def run_benchmark():
    file_path = "data/book.md"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 1. Chunking
    chunker = SoftwareEngineeringChunker(chunk_size=800)
    chunks = chunker.chunk(text)
    documents = [Document(id=f"chunk_{i}", content=c, metadata={"source": "book.md"}) for i, c in enumerate(chunks)]
    
    # 2. Store
    store = EmbeddingStore()
    store.add_documents(documents)
    
    # 3. Queries
    queries = [
        "What are the core benefits of using Information Systems?",
        "Governance in urbanized IT environments",
        "Explain business and IT alignment",
        "Role of a Series Editor",
        "Who is Jean-Charles Pomerol?"
    ]
    
    print("--- Benchmark Results ---")
    for i, q in enumerate(queries):
        results = store.search(q, top_k=1)
        if results:
            res = results[0]
            content_preview = res["content"].replace("\n", " ")[:100].strip() + "..."
            print(f"Q{i+1}: {q}")
            print(f"  Chunk: {content_preview}")
            print(f"  Score: {res['score']:.4f}")
        else:
            print(f"Q{i+1}: No results found")

def run_similarity_check():
    print("\n--- Similarity Prediction Check ---")
    pairs = [
        ("Phát triển phần mềm sạch.", "Phát triển phần mềm sạch."),
        ("SOLID principles are essential.", "Clean code is important."),
        ("Recursive functions call themselves.", "Functions that invoke itself."),
        ("Python is a language.", "I love eating chocolate cake."),
        ("The server error is 404.", "The cat is on the mat.")
    ]
    
    for i, (a, b) in enumerate(pairs):
        score = compute_similarity(_mock_embed(a), _mock_embed(b))
        print(f"Pair {i+1}: {score:.4f}")

if __name__ == "__main__":
    import os
    run_benchmark()
    run_similarity_check()
