import json
import asyncio
import os
from typing import List, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

async def generate_qa_from_text(text: str, num_pairs: int = 5) -> List[Dict]:
    """
    Sử dụng OpenAI API để tạo các cặp (Question, Expected Answer, Context)
    từ đoạn văn bản cho trước.
    Bao gồm ít nhất 1 câu hỏi 'bẫy' (adversarial) hoặc cực khó.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"Generating {num_pairs} QA pairs from text...")

    prompt = f"""Từ đoạn văn bản sau, hãy tạo chính xác {num_pairs} cặp câu hỏi-trả lời.
Yêu cầu:
- Ít nhất 1 câu hỏi phải là loại "adversarial" (hỏi thứ KHÔNG có trong tài liệu để kiểm tra xem AI có bịa không)
- Các câu còn lại là factual (dựa trên nội dung văn bản)
- Trả về JSON object với key "pairs" chứa array

Format JSON:
{{
  "pairs": [
    {{
      "question": "câu hỏi ở đây",
      "expected_answer": "câu trả lời kỳ vọng",
      "context": "đoạn văn bản liên quan trực tiếp (trích từ văn bản gốc)",
      "ground_truth_doc_id": "doc_001",
      "metadata": {{
        "difficulty": "easy hoặc hard",
        "type": "factual hoặc adversarial"
      }}
    }}
  ]
}}

Văn bản:
{text}
"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("pairs", [])


async def main():
    # Tài liệu mẫu — thay bằng nội dung thật của nhóm nếu có
    documents = [
        {
            "id": "doc_001",
            "text": """Trí tuệ nhân tạo (AI) là lĩnh vực khoa học máy tính tập trung vào việc tạo ra các hệ thống
có khả năng thực hiện các nhiệm vụ thường đòi hỏi trí thông minh của con người.
Các ứng dụng AI bao gồm nhận dạng giọng nói, ra quyết định, dịch thuật ngôn ngữ và nhận dạng hình ảnh.
Machine Learning (ML) là một nhánh của AI, cho phép máy tính học từ dữ liệu mà không cần lập trình tường minh.
Deep Learning sử dụng mạng nơ-ron nhiều lớp để học các biểu diễn phức tạp từ dữ liệu lớn.
Large Language Models (LLM) như GPT-4 được huấn luyện trên hàng tỷ văn bản để hiểu và tạo ra ngôn ngữ tự nhiên."""
        },
        {
            "id": "doc_002",
            "text": """RAG (Retrieval-Augmented Generation) là kỹ thuật kết hợp tìm kiếm tài liệu với sinh văn bản.
Quy trình RAG gồm 3 bước chính: (1) Người dùng đặt câu hỏi, (2) Hệ thống tìm kiếm các đoạn văn bản liên quan
từ cơ sở dữ liệu vector, (3) LLM tổng hợp câu trả lời dựa trên context tìm được.
Vector Database lưu trữ các embedding (biểu diễn số học) của văn bản để tìm kiếm ngữ nghĩa nhanh chóng.
Chunking là quá trình chia nhỏ tài liệu thành các đoạn nhỏ hơn trước khi lưu vào vector database.
Hallucination xảy ra khi LLM tạo ra thông tin sai lệch không có trong context được cung cấp."""
        },
        {
            "id": "doc_003",
            "text": """Evaluation (đánh giá) AI là quá trình đo lường chất lượng và hiệu suất của các hệ thống AI.
RAGAS là framework phổ biến để đánh giá hệ thống RAG, bao gồm các metrics: Faithfulness (độ trung thực),
Answer Relevancy (độ liên quan), Context Precision và Context Recall.
Faithfulness đo lường xem câu trả lời có được hỗ trợ bởi context hay không (tránh hallucination).
Answer Relevancy đo lường xem câu trả lời có phù hợp với câu hỏi không.
Hit Rate là tỷ lệ câu hỏi mà retriever tìm được ít nhất 1 tài liệu đúng trong kết quả.
MRR (Mean Reciprocal Rank) đo thứ hạng trung bình của tài liệu đúng đầu tiên trong kết quả tìm kiếm."""
        },
        {
            "id": "doc_004",
            "text": """Benchmark là quá trình đo lường và so sánh hiệu suất của các hệ thống AI.
Golden Dataset là tập dữ liệu kiểm tra chất lượng cao, được tạo thủ công hoặc bán tự động,
dùng để đánh giá nhất quán qua các phiên bản agent khác nhau.
Regression Testing trong AI kiểm tra xem phiên bản mới có tệ hơn phiên bản cũ không.
Synthetic Data Generation (SDG) là kỹ thuật dùng LLM để tự động tạo dữ liệu kiểm tra đa dạng.
Async programming cho phép chạy nhiều tác vụ song song, giúp tăng tốc độ benchmark đáng kể.
Batch processing chia dataset thành nhóm nhỏ để tránh bị rate limit từ API."""
        },
        {
            "id": "doc_005",
            "text": """Failure Analysis (Phân tích thất bại) là bước quan trọng sau khi chạy benchmark.
Failure Clustering nhóm các lỗi tương tự lại để tìm pattern, ví dụ: Hallucination, Incomplete Answer, Tone Mismatch.
Phân tích 5 Whys là kỹ thuật tìm nguyên nhân gốc rễ bằng cách hỏi "Tại sao?" 5 lần liên tiếp.
Root Cause Analysis giúp xác định lỗi nằm ở tầng nào: Ingestion, Chunking, Retrieval hay Prompting.
Action Plan là kế hoạch cải tiến cụ thể dựa trên kết quả phân tích, ví dụ: thay đổi chunking strategy,
cập nhật system prompt, hoặc thêm bước reranking vào pipeline."""
        },
    ]

    all_pairs = []
    pairs_per_doc = 10  # 5 docs × 10 pairs = 50 cases

    tasks = [
        generate_qa_from_text(doc["text"], pairs_per_doc)
        for doc in documents
    ]

    results = await asyncio.gather(*tasks)

    for doc, pairs in zip(documents, results):
        for pair in pairs:
            pair["ground_truth_doc_id"] = doc["id"]
            all_pairs.append(pair)

    os.makedirs("data", exist_ok=True)
    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"✅ Done! Tạo được {len(all_pairs)} test cases → data/golden_set.jsonl")

    # Thống kê nhanh
    adversarial = sum(1 for p in all_pairs if p.get("metadata", {}).get("type") == "adversarial")
    hard = sum(1 for p in all_pairs if p.get("metadata", {}).get("difficulty") == "hard")
    print(f"   Adversarial cases: {adversarial}")
    print(f"   Hard cases: {hard}")


if __name__ == "__main__":
    asyncio.run(main())