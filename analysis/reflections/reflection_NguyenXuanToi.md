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

## 4. Nếu có thêm thời gian, em sẽ cải thiện gì?

- Tích hợp Vector DB thật (ChromaDB) vào MainAgent để Hit Rate không còn là 0%.
- Cải thiện Judge Prompt bằng cách thêm rubric chi tiết và ví dụ few-shot — mục tiêu đưa Agreement Rate lên trên 70%.
- Thêm cost tracking để biết chính xác mỗi lần eval tốn bao nhiêu tiền, từ đó tối ưu chi phí.

---

## 5. Một câu tóm tắt buổi học

> "Xây AI thì dễ, nhưng biết AI đang sai ở đâu mới là thứ khó — và đó chính xác là thứ hệ thống Evaluation Factory này giải quyết."