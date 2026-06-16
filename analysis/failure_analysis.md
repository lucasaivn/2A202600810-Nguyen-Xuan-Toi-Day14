# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark

- **Tổng số cases:** 50
- **Tỉ lệ Pass/Fail:** 7/43 (Pass rate: 14%)
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.70
    - Relevancy: 0.22
- **Điểm LLM-Judge trung bình:** 2.3 / 5.0
- **Hit Rate:** 0.0%
- **MRR:** 0.0
- **Agreement Rate (Multi-Judge):** 28.0%
- **Phiên bản Agent:** Agent_V2_Optimized vs Agent_V1_Base (Delta: -0.10)

---

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng ước tính | Nguyên nhân dự kiến |
|----------|------------------|---------------------|
| Lạc đề (Off-topic) | ~20 | Agent trả lời template cứng, không đọc câu hỏi thật |
| Hallucination | ~12 | Agent bịa thêm thông tin không có trong context |
| Incomplete Answer | ~8 | Câu trả lời quá ngắn, thiếu chi tiết |
| Wrong on Adversarial | ~3 | Agent không nhận ra câu hỏi bẫy, vẫn cố trả lời |

---

## 3. Phân tích 5 Whys (3 case tệ nhất)

### Case #1: Agent trả lời lạc đề với câu hỏi về RAG pipeline
1. **Symptom:** Agent trả lời câu hỏi về quy trình RAG bằng câu template cứng không liên quan.
2. **Why 1:** Câu trả lời không khớp nội dung câu hỏi → Relevancy score = 0.1
3. **Why 2:** Agent không thật sự tìm kiếm context liên quan trước khi trả lời.
4. **Why 3:** MainAgent đang dùng câu trả lời hardcode thay vì gọi LLM thật.
5. **Why 4:** Chưa tích hợp Vector DB thật vào pipeline Retrieval.
6. **Root Cause:** Agent thiếu bước Retrieval thật — không có dữ liệu đầu vào thì LLM không thể sinh câu trả lời đúng.

---

### Case #2: Agent bịa thêm thông tin khi trả lời câu hỏi adversarial
1. **Symptom:** Câu hỏi hỏi thứ không có trong tài liệu nhưng Agent vẫn trả lời tự tin → Faithfulness = 0.3
2. **Why 1:** Agent không có cơ chế phát hiện "câu hỏi ngoài phạm vi tài liệu".
3. **Why 2:** System prompt không có hướng dẫn "Nếu không biết, hãy nói không biết".
4. **Why 3:** Không có bước kiểm tra độ phủ context trước khi sinh câu trả lời.
5. **Why 4:** Pipeline thiếu bước confidence check sau Retrieval.
6. **Root Cause:** System prompt yếu — không ràng buộc Agent phải trung thực khi thiếu thông tin, dẫn đến Hallucination.

---

### Case #3: 2 model Judge bất đồng nghiêm trọng (Agreement Rate thấp)
1. **Symptom:** gpt-4o-mini chấm 4 điểm, gpt-3.5-turbo chấm 1 điểm cho cùng một câu trả lời.
2. **Why 1:** Câu trả lời mơ hồ — đúng một phần nhưng thiếu chi tiết quan trọng.
3. **Why 2:** Prompt chấm điểm chưa có rubric rõ ràng, mỗi model tự diễn giải tiêu chí.
4. **Why 3:** Không có ví dụ mẫu (few-shot) để calibrate thang điểm giữa 2 model.
5. **Why 4:** gpt-3.5-turbo ít nhạy cảm hơn với câu trả lời mơ hồ so với gpt-4o-mini.
6. **Root Cause:** Rubric chấm điểm không đủ cụ thể → mỗi model judge diễn giải khác nhau → Agreement Rate chỉ đạt 28%.

---

## 4. Kế hoạch cải tiến (Action Plan)

- [ ] **Tích hợp Vector DB thật** (ChromaDB hoặc FAISS) vào MainAgent để Retrieval hoạt động thực sự — giải quyết Hit Rate 0%.
- [ ] **Cập nhật System Prompt** thêm ràng buộc: *"Chỉ trả lời dựa trên context được cung cấp. Nếu không có thông tin, trả lời: Tôi không tìm thấy thông tin này trong tài liệu."*
- [ ] **Cải thiện Judge Prompt** — thêm rubric chi tiết và ví dụ few-shot để 2 model calibrate đồng đều, mục tiêu Agreement Rate > 70%.
- [ ] **Thêm bước Reranking** sau Retrieval để đảm bảo context liên quan nhất được đưa lên đầu.
- [ ] **Thêm Confidence Check** — nếu similarity score của context < ngưỡng 0.7, không sinh câu trả lời mà trả về "Không đủ thông tin".