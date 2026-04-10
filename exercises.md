# Day 7 — Exercises
## Data Foundations: Embedding & Vector Store | Lab Worksheet

---

## Part 1 — Warm-up (Cá nhân)

### Exercise 1.1 — Cosine Similarity in Plain Language

**What does it mean for two text chunks to have high cosine similarity?**
> Hai vector chỉ về cùng một hướng trong không gian embedding, cho thấy hai đoạn văn bản có sự tương đồng lớn về mặt ngữ nghĩa (semantic relationship).

**Give a concrete example of two sentences that would have HIGH similarity and two that would have LOW similarity.**
- **HIGH similarity:**
    - Sentence A: "Hôm nay trời nắng rất đẹp."
    - Sentence B: "Thời tiết hôm nay thật rực rỡ và đầy ánh nắng."
    - *Tại sao tương đồng:* Cùng mô tả một trạng thái thời tiết thuận lợi với các khái niệm tương đồng.
- **LOW similarity:**
    - Sentence A: "Lập trình Python rất thú vị."
    - Sentence B: "Tôi vừa mua một cân táo ở chợ."
    - *Tại sao khác:* Hai câu thuộc hai chủ đề hoàn toàn khác nhau (công nghệ vs. mua sắm thực phẩm).

**Why is cosine similarity preferred over Euclidean distance for text embeddings?**
> Euclidean distance nhạy cảm với độ dài của vector (độ dài văn bản), trong khi cosine similarity chỉ tập trung vào góc giữa các vector (hướng của ý nghĩa), giúp so sánh hiệu quả hơn giữa các tài liệu có độ dài khác nhau.

---

### Exercise 1.2 — Chunking Math

- **A document is 10,000 characters. You chunk it with `chunk_size=500`, `overlap=50`. How many chunks do you expect?**
    - phép tính: `ceil((10,000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
    - Đáp án: 23 chunks.

- **If overlap is increased to 100, how does this change the chunk count? Why would you want more overlap?**
    - Phép tính: `ceil((10,000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25`
    - Đáp án: 25 chunks.
    - Tại sao: Việc tăng overlap giúp duy trì ngữ cảnh giữa các đoạn (context preservation), đảm bảo các thông tin quan trọng không bị cắt rời một cách đột ngột.

---

## Part 2 — Core Coding (Cá nhân)

Implement all TODOs in `src/chunking.py`, `src/store.py`, và `src/agent.py`. `Document` dataclass và `FixedSizeChunker` đã được implement sẵn làm ví dụ — đọc kỹ để hiểu pattern trước khi implement phần còn lại.

Run `pytest tests/` to check progress.

### Checklist
- [x] `Document` dataclass — ĐÃ IMPLEMENT SẴN
- [x] `FixedSizeChunker` — ĐÃ IMPLEMENT SẴN
- [x] `SentenceChunker` — split on sentence boundaries, group into chunks
- [x] `RecursiveChunker` — try separators in order, recurse on oversized pieces
- [x] `compute_similarity` — cosine similarity formula with zero-magnitude guard
- [x] `ChunkingStrategyComparator` — call all three, compute stats
- [x] `EmbeddingStore.__init__` — initialize store (in-memory or ChromaDB)
- [x] `EmbeddingStore.add_documents` — embed and store each document
- [x] `EmbeddingStore.search` — embed query, rank by dot product
- [x] `EmbeddingStore.get_collection_size` — return count
- [x] `EmbeddingStore.search_with_filter` — filter by metadata, then search
- [x] `EmbeddingStore.delete_document` — remove all chunks for a doc_id
- [x] `KnowledgeBaseAgent.answer` — retrieve + build prompt + call LLM

> **Nộp code:** `src/`
> **Ghi approach vào:** Report — Section 4 (My Approach)

---

## Part 3 — So Sánh Retrieval Strategy (Nhóm)

### Exercise 3.0 — Chuẩn Bị Tài Liệu (Giờ đầu tiên)

Mỗi nhóm chọn một domain và chuẩn bị bộ tài liệu:

**Step 1 — Chọn domain:** FAQ, SOP, policy, docs kỹ thuật, recipes, luật, y tế, v.v.

**Step 2 — Thu thập 5-10 tài liệu.** Lưu dưới dạng `.txt` hoặc `.md` vào thư mục `data/`.

> **Tip chuyển PDF sang Markdown:**
> - `pip install marker-pdf` → `marker_single input.pdf output/` (chất lượng cao, giữ cấu trúc)
> - `pip install pymupdf4llm` → `pymupdf4llm.to_markdown("input.pdf")` (nhanh, đơn giản)
> - Hoặc copy-paste nội dung từ PDF/web vào file `.txt`

Ghi vào bảng:

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | book.md | [Wiley Online Library](https://onlinelibrary.wiley.com/doi/book/10.1002/9781394297696) | 503,401 | `{"category": "software-engineering", "source": "book.md"}` |

**Step 3 — Thiết kế metadata schema:**
- `category`: Phân loại chủ đề (e.g., "software-engineering").
- `source`: Tên file gốc để truy xuất nguồn (e.g., "book.md").

---

### Exercise 3.1 — Thiết Kế Retrieval Strategy (Mỗi người thử riêng)

**Step 1 — Baseline:**
| Strategy | Chunk Count | Avg Length |
|----------|-------------|------------|
| FixedSizeChunker | 1119 | 499.82 |
| SentenceChunker | 1160 | 432.20 |
| RecursiveChunker | 1398 | 358.47 |

**Step 2 — Chọn hoặc thiết kế strategy của bạn:**

```python
class SoftwareEngineeringChunker(RecursiveChunker):
    """
    Optimized to keep technical hierarchies (headers) and code blocks together.
    Design rationale: Ưu tiên các header H1, H2, H3 để giữ các section đi kèm tiêu đề, 
    bối cảnh nội bộ section được bảo toàn.
    """
    SE_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        super().__init__(separators=self.SE_SEPARATORS, chunk_size=chunk_size)
```

**Step 3 — So sánh:**
| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| book.md | RecursiveChunker | 1398 | 358.47 | HIGH |
| book.md | SoftwareEngineeringChunker | 923 | 543.40 | **HIGH** |

---

### Exercise 3.2 — Chuẩn Bị Benchmark Queries

Mỗi nhóm viết **đúng 5 benchmark queries** kèm **gold answers**.

| # | Query | Gold Answer (câu trả lời đúng) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | "Core benefits of Information Systems?" | transformation, strategic alignment, efficiency. | Chunks về transformation |
| 2 | "Governance in IT environments?" | Urbanization, management, and regulatory compliance. | Section 9.1 |
| 3 | "Explain alignment?" | Convergence between business goals and IT infrastructure. | Section 5.2 |
| 4 | "Role of a Series Editor?" | Oversight, selection of academic content, quality control. | Metadata/Intro |
| 5 | "Who is Jean-Charles Pomerol?" | Series Editor and expert mentioned in the book's metadata. | Metadata |

---

### Exercise 3.3 — Cosine Similarity Predictions (Cá nhân)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Phát triển phần mềm sạch." | "Phát triển phần mềm sạch." | High | 1.0000 | Có |
| 2 | "SOLID principles are essential." | "Clean code is important." | High | 0.1844 | Không |
| 3 | "Recursive functions call themselves." | "Functions that invoke itself." | High | -0.2335 | Không |
| 4 | "Python is a language." | "I love eating chocolate cake." | Low | -0.0482 | Có |
| 5 | "The server error is 404." | "The cat is on the mat." | Low | -0.0525 | Có |

---

### Exercise 3.4 — Chạy Benchmark & So Sánh Trong Nhóm

- **Strategy tốt nhất:** Semantic Chunking (Nguyễn Tuấn Hưng) — 9.5/10.
- **Lý do:** Giữ trọn vẹn ngữ cảnh của từng mục phức tạp trong SE.
- **Cải thiện:** SoftwareEngineeringChunker (Lê Minh Hoàng) đạt 8/10, tốt ở việc giữ cấu trúc Markdown nhưng cần cải thiện semantic.

---

### Exercise 3.5 — Failure Analysis

- **Query thất bại:** "Role of Editor" hoặc "Who is Pomerol".
- **Tại sao:** Hệ thống sử dụng `MockEmbedder` dựa trên hashing, dẫn đến việc chỉ nhận diện trùng khớp ký tự mà không hiểu ngữ nghĩa. Các chunk retrieved không chứa đúng thông tin vì query không trùng từ khóa chính xác trong chunk đó.
- **Đề xuất cải thiện:** Sử dụng mô hình embedding thực thụ (như OpenAI text-embedding-3-small hoặc HuggingFace Sentence-Transformers) và bộ lọc metadata chính xác hơn.

---

## Submission Checklist

- [x] All tests pass: `pytest tests/ -v`
- [x] `src/` updated (cá nhân)
- [x] Report completed (`report/REPORT.md` — 1 file/sinh viên)
