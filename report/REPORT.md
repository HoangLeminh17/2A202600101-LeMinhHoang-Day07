# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Minh Hoàng
**Nhóm:** D1
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector chỉ về cùng một hướng trong không gian embedding, cho thấy hai đoạn văn bản có sự tương đồng lớn về mặt ngữ nghĩa (semantic relationship).

- Sentence A: "Hôm nay trời nắng rất đẹp."
- Sentence B: "Thời tiết hôm nay thật rực rỡ và đầy ánh nắng."
- Tại sao tương đồng: Cùng mô tả một trạng thái thời tiết thuận lợi với các khái niệm tương đồng.

**Ví dụ LOW similarity:**
- Sentence A: "Lập trình Python rất thú vị."
- Sentence B: "Tôi vừa mua một cân táo ở chợ."
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau (công nghệ vs. mua sắm thực phẩm).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Euclidean distance nhạy cảm với độ dài của vector (độ dài văn bản), trong khi cosine similarity chỉ tập trung vào góc giữa các vector (hướng của ý nghĩa), giúp so sánh hiệu quả hơn giữa các tài liệu có độ dài khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* `ceil((10,000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`

> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số lượng chunk sẽ tăng lên (25 chunks). Việc tăng overlap giúp duy trì ngữ cảnh giữa các đoạn (context preservation), đảm bảo các thông tin quan trọng không bị cắt rời một cách đột ngột.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Software Engineering

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*Chọn domain này vì nó chứa các khái niệm cốt lõi, có cấu trúc rõ ràng và phân cấp, rất phù hợp để thử nghiệm các chiến lược chunking khác nhau. Việc hiểu và truy xuất chính xác các nguyên lý như SOLID, DRY, KISS là một bài toán thực tế và hữu ích cho các kỹ sư phần mềm.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | book.md | https://onlinelibrary.wiley.com/doi/book/10.1002/9781394297696?msockid=342527e00a4661fb18ff345a0bdc6080 | 503401 | `{"category": "software-engineering", "source": "book.md"}` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | string | "software-engineering" | Giúp lọc các tài liệu theo chủ đề lớn, hữu ích khi hệ thống có nhiều domain khác nhau. |
| source | string | "book.md" | Cho phép truy xuất nguồn gốc của chunk, giúp xác minh thông tin và cung cấp thêm ngữ cảnh cho người dùng. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
|book.md | FixedSizeChunker (`fixed_size`) | 1119 | 499.82 | LOW |
|book.md | SentenceChunker (`by_sentences`) | 1160 | 432.20 | MEDIUM |
|book.md | RecursiveChunker (`recursive`) | 1398 | 358.47 | HIGH |

### Strategy Của Tôi

**Loại:** `SoftwareEngineeringChunker` (Custom Recursive Strategy)

**Mô tả cách hoạt động:**
> Chiến lược này kế thừa từ `RecursiveChunker` nhưng được cấu hình lại với danh sách `separators` ưu tiên các ký tự phân tách cấu trúc Markdown như Header H1 (`\n# `), H2 (`\n## `) và H3 (`\n### `). Nó cố gắng giữ toàn bộ một mục tài liệu hoặc một đoạn code block trong cùng một chunk bằng cách chỉ chia nhỏ hơn khi kích thước vượt quá 800 ký tự.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu Software Engineering (như file `book.md`) thường có các định nghĩa và ví dụ code đi kèm. Việc sử dụng Header làm điểm ngắt ưu tiên giúp các kiến thức liên quan không bị tách rời, giúp hệ thống RAG truy xuất được đầy đủ ngữ cảnh của một khái niệm thay vì chỉ một phần vụn vặt.

**Code snippet (nếu custom):**
```python
class SoftwareEngineeringChunker(RecursiveChunker):
    """
    Optimized to keep technical hierarchies (headers) and code blocks together.
    """
    SE_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        super().__init__(separators=self.SE_SEPARATORS, chunk_size=chunk_size)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
|book.md | RecursiveChunker (Best Baseline) | 1398 | 358.47 | HIGH |
|book.md | **SoftwareEngineeringChunker** | 923 | 543.40 | **HIGH** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Lê Minh Hoàng | SoftwareEngineeringChunker | 8 | Tận dụng cấu trúc Markdown để giữ các section đi kèm tiêu đề, bối cảnh nội bộ section được bảo toàn. | Quá phụ thuộc vào định dạng file; dễ thất bại nếu câu hỏi yêu cầu dữ liệu nằm rải rác ở nhiều mục khác nhau. |
| Nguyễn Tuấn Hưng | Semantic Chunking | 9.5 | Giữ trọn vẹn ngữ cảnh của từng mục, truy xuất chính xác. | Các chunk có thể rất lớn, không phù hợp với các mô hình có giới hạn context nhỏ. |
| Nguyễn Xuân Hải | Parent-Child Chunking | 8 | Child nhỏ giúp tìm kiếm vector đúng mục tiêu, ít nhiễu | Gửi cả khối Parent lớn vào Prompt làm tăng chi phí API. |
| Nguyễn Đăng Hải | DocumentStructureChunker | 6.3 | Giữ ngữ cảnh theo heading/list/table; grounding tốt cho tài liệu dài | Phức tạp hơn và tốn xử lý hơn; lợi thế giảm khi dữ liệu ít cấu trúc |
| Thái Minh Kiên | Agentic Chunking | 8 | chunk giữ được ý nghĩa trọn vẹn, retrieval chính xác hơn, ít trả về nửa vời | Với dataset lớn cost sẽ tăng mạnh, chậm hơn pipeline thường |
| Trần Trung Hậu | Token-Based Chunking | 8 | Kiểm soát chính xác tuyệt đối giới hạn đầu vào và chi phí API. | Cắt rất máy móc, dễ làm đứt gãy ngữ nghĩa giữa chừng. |
| Tạ Bảo Ngọc | Sliding Window + Overlap | 7 | Giữ vẹn câu/khối logic, tối ưu length | Bị trùng dữ liệu -> tăng số chunk |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Chiến lược **Semantic Chunking** của bạn Nguyễn Tuấn Hưng là tốt nhất cho domain này. Bởi vì tài liệu Software Engineering chứa nhiều khái niệm phức tạp và trừu tượng, việc phân đoạn dựa trên ý nghĩa (semantic) giúp đảm bảo các thực thể liên quan luôn nằm cùng nhau, từ đó mang lại độ chính xác truy xuất cao nhất (9.5/10) và giúp Agent hiểu sâu bối cảnh hơn các phương pháp cắt dựa trên cấu trúc thông thường.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng thư viện `re` với regex `(?<=[.!?])\s+` để tách câu dựa trên các dấu kết thúc (. ! ?) mà không làm mất dấu câu đó. Sau đó, gom các câu này lại thành nhóm dựa trên tham số `max_sentences_per_chunk` để tạo thành các chunk có ý nghĩa hoàn chỉnh hơn so với việc cắt theo số lượng ký tự.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Sử dụng thuật toán đệ quy để chia nhỏ văn bản dựa trên danh sách các ký tự phân tách (separators) theo thứ tự ưu tiên (đoạn văn -> dòng -> câu -> từ). Nếu một đoạn văn bản vẫn lớn hơn `chunk_size`, nó sẽ được gửi xuống cấp độ separator tiếp theo cho đến khi đạt kích thước mong muốn. Base case là khi không còn separator nào hoặc văn bản đã đủ nhỏ.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Các tài liệu được chia nhỏ thành các chunk, sau đó được embedding hóa và lưu trữ vào ChromaDB (hoặc danh sách in-memory). Khi tìm kiếm, query cũng được chuyển thành vector và so sánh với kho dữ liệu. Điểm tương đồng được chuẩn hóa về thang đo similarity (0 đến 1) thay vì để nguyên khoảng cách (distance).

**`search_with_filter` + `delete_document`** — approach:
> Áp dụng lọc theo metadata trước khi thực hiện tìm kiếm vector để đảm bảo tốc độ và độ chính xác (pre-filtering). `delete_document` thực hiện xóa tất cả các chunk có `doc_id` tương ứng để đảm bảo tính nhất quán của dữ liệu khi tài liệu gốc bị gỡ bỏ.

### KnowledgeBaseAgent

**`answer`** — approach:
> Cấu trúc prompt được thiết kế theo dạng: System Instruction -> Context (Retrieved Chunks) -> User Question. Tôi sử dụng kỹ thuật "grounding" bằng cách ra lệnh cho Agent chỉ được trả lời dựa trên context được cung cấp, nếu không có thông tin thì phải thành thật trả lời là không biết để tránh hiện tượng ảo giác (hallucination).

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.13.2, pytest-9.0.2, pluggy-1.5.0
plugins: anyio-4.13.0, langsmith-0.7.26
collected 42 items

tests/test_solution.py PASSED                                           [100%]
============================= 42 passed in 1.48s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Phát triển phần mềm sạch." | "Phát triển phần mềm sạch." | High | 1.0000 | Có |
| 2 | "SOLID principles are essential." | "Clean code is important." | High | 0.1844 | Không |
| 3 | "Recursive functions call themselves." | "Functions that invoke itself." | High | -0.2335 | Không |
| 4 | "Python is a language." | "I love eating chocolate cake." | Low | -0.0482 | Có |
| 5 | "The server error is 404." | "The cat is on the mat." | Low | -0.0525 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là các câu có ý nghĩa tương đương (như Pair 2, 3) lại có điểm tương đồng rất thấp (gần 0). Điều này cho thấy hệ thống đang sử dụng `MockEmbedder` dựa trên thuật toán băm (hashing) thay vì mô hình ngôn ngữ thực thụ, dẫn đến việc nó chỉ nhận diện được sự trùng khớp chính xác về ký tự chứ không hiểu được ngữ nghĩa sâu xa.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | "Core benefits of Information Systems?" | transformation, strategic alignment, efficiency. |
| 2 | "Governance in IT environments?" | Urbanization, management, and regulatory compliance. |
| 3 | "Explain alignment?" | Convergence between business goals and IT infrastructure. |
| 4 | "Role of a Series Editor?" | Oversight, selection of academic content, quality control. |
| 5 | "Who is Jean-Charles Pomerol?" | Series Editor and expert mentioned in the book's metadata. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | "Benefits of IS" | In the face of this transformation, flaws soon appeared... | 0.4718 | Yes |IS mang lại sự chuyển đổi nhưng cũng nảy sinh nhiều lỗi sau khi chuyển đổi. |
| 2 | "IT Governance" | Box 9.1. Money laundering, cryptocurrencies and virtual gold coins ... | 0.4439 | Partial |Đề cập đến quản trị trong các khía cạnh tài chính/crypto. |
| 3 | "Explain alignment" | Territorial Urbanization 113 Figure 5.2. Example of urbanization... | 0.4615 | No |Nói về đô thị hóa lãnh thổ, không trực tiếp giải thích về alignment. |
| 4 | "Role of Editor" | 4.8.6.3. _Modeling information content_ The class diagram can also be used... | 0.4400 | No |Tập trung vào mô hình hóa nội dung thông tin, không nói về Editor. |
| 5 | "Who is Pomerol" | Modeling can be focused on a description of the informational content... | 0.4675 | No |Nói về các kiểu modeling, không cung cấp thông tin về Pomerol. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được cách thiết kế Metadata Schema đa dạng từ bạn Ngọc (category, difficulty), giúp việc lọc tài liệu trở nên trực quan hơn nhiều so với chỉ lọc theo tên file.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm 4 đã visualize được kết quả của các phương pháp khác nhau, giải thích và phân tích các kết quả đó, giúp tôi hiểu rõ hơn về cách các phương pháp chunking khác nhau hoạt động và ảnh hưởng đến kết quả tìm kiếm.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ tập trung hơn vào việc làm sạch dữ liệu (data cleaning) trước khi chunking, loại bỏ các ký tự rác từ quá trình convert PDF để cải thiện chất lượng embedding, và sẽ thử nghiệm với `SemanticChunker` nếu có tài nguyên GPU.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
