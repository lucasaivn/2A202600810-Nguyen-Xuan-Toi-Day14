# Reflection — Lab 14: AI Evaluation Factory
**Họ và tên:** Nguyễn Xuân Tới  
**Mã sinh viên:** 2A202600810  
**Ngày nộp:** 2026-06-16

---

## 1. Em đã làm gì trong buổi lab này?

Trong buổi lab, em tham gia xây dựng hệ thống đánh giá tự động (Evaluation Factory) cho AI Agent. Cụ thể, em implement phần Synthetic Data Generation — dùng OpenAI API để tự động tạo 50 test cases từ 5 tài liệu khác nhau, bao gồm cả câu hỏi thông thường và câu hỏi "bẫy" (adversarial) để kiểm tra xem agent có bịa thông tin không có trong tài liệu không.

Em cũng tham gia viết phần ExpertEvaluator để tính điểm Faithfulness và Relevancy thật bằng cách gọi LLM, thay vì dùng số hardcode như ban đầu.

---

## 2. Điều em học được từ kết quả benchmark

Kết quả chạy thật khá bất ngờ: điểm trung bình chỉ đạt 2.3/5, Hit Rate 0%, và Agreement Rate giữa 2 model judge chỉ 28%. Lúc đầu em tưởng agent sẽ đạt điểm cao hơn.

Sau khi phân tích, em hiểu ra nguyên nhân chính: agent đang dùng câu trả lời cứng (hardcode), không thật sự tìm kiếm tài liệu liên quan. Điều này khiến Relevancy rất thấp (0.22) vì câu trả lời không bám vào câu hỏi thật.

Bài học lớn nhất: **đo lường trước khi tối ưu**. Nếu không có hệ thống eval này, em không thể biết chính xác agent đang tệ ở đâu — bịa thông tin, lạc đề, hay tìm sai tài liệu.

---

## 3. Điều em thấy khó nhất

Phần khó nhất là hiểu sự khác biệt giữa Hit Rate và MRR. Ban đầu em nghĩ 2 chỉ số này giống nhau vì đều liên quan đến Retrieval. Sau khi làm xong mới hiểu: Hit Rate chỉ đo "có tìm đúng không", còn MRR đo "tìm đúng nhưng xếp ở vị trí thứ mấy" — vị trí càng cao thì điểm càng cao.

---

## 4. Phân tích kỹ thuật nâng cao (Technical Depth)

### Cohen's Kappa — Đo mức độ đồng thuận thật sự giữa 2 Judge

Trong bài, em dùng Agreement Rate đơn giản: nếu 2 model chênh nhau ≤ 1 điểm thì coi là "đồng ý". Nhưng có vấn đề: nếu 2 model cùng chấm ngẫu nhiên, xác suất chúng vô tình đồng ý vẫn khá cao.

**Cohen's Kappa** giải quyết vấn đề này bằng cách loại trừ phần đồng thuận ngẫu nhiên ra khỏi phép tính: κ = (Po - Pe) / (1 - Pe)
Trong đó:
- `Po` = tỉ lệ đồng thuận thực tế (observed agreement)
- `Pe` = tỉ lệ đồng thuận kỳ vọng nếu chấm ngẫu nhiên (expected by chance)

Thang đánh giá: κ < 0.4 = kém, 0.4–0.6 = trung bình, 0.6–0.8 = tốt, > 0.8 = xuất sắc.

Nếu bài em đạt Agreement Rate 28% nhưng κ ≈ 0.1, điều đó chứng tỏ 2 model không thật sự đồng thuận — phần lớn là may mắn. Đây là lý do rubric chấm điểm cần được chuẩn hóa để κ đạt ít nhất 0.6.

---

### Position Bias — LLM Judge thiên vị về vị trí

Khi yêu cầu LLM so sánh 2 câu trả lời (A vs B), các nghiên cứu cho thấy LLM có xu hướng chọn câu trả lời **ở vị trí đầu tiên** nhiều hơn, bất kể chất lượng thật sự. Đây gọi là **Position Bias**.

Ví dụ: Nếu ta hỏi "Câu trả lời nào tốt hơn: [A] hay [B]?", LLM thường chọn [A]. Nếu đổi thứ tự thành "[B] hay [A]?", nó lại chọn [B].

**Cách phát hiện:** Chạy cùng một cặp 2 lần với thứ tự đảo ngược. Nếu kết quả thay đổi → Position Bias đang ảnh hưởng.

**Cách giảm thiểu:**
- Dùng nhiều lần chấm với thứ tự ngẫu nhiên, lấy kết quả đa số.
- Thêm rubric cụ thể để LLM judge bám theo tiêu chí, không bị ảnh hưởng bởi vị trí.
- Dùng Chain-of-Thought: yêu cầu Judge giải thích lý do trước khi chấm điểm.

---

## 5. Nếu có thêm thời gian, em sẽ cải thiện gì?

- Tích hợp Vector DB thật (ChromaDB) vào MainAgent để Hit Rate không còn là 0%.
- Cải thiện Judge Prompt bằng cách thêm rubric chi tiết và ví dụ few-shot — mục tiêu đưa Agreement Rate lên trên 70%, κ ≥ 0.6.
- Implement Position Bias check: chạy mỗi cặp 2 lần với thứ tự đảo ngược để phát hiện và loại bỏ bias.
- Cost tracking đã được tích hợp — chi phí mỗi lần eval ~$0.002–0.005 USD, có thể giảm thêm bằng cách cache câu hỏi lặp lại.

---

## 6. Một câu tóm tắt buổi học

> "Xây AI thì dễ, nhưng biết AI đang sai ở đâu mới là thứ khó — và đó chính xác là thứ hệ thống Evaluation Factory này giải quyết."