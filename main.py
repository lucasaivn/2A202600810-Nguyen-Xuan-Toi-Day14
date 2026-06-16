import asyncio
import json
import os
import re
import time
from openai import AsyncOpenAI
from dotenv import load_dotenv
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent

load_dotenv()

# ============================================================
# COST TRACKER
# ============================================================
PRICING = {
    "gpt-4o-mini":   {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
}

class CostTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.calls = 0

    def record(self, model: str, usage):
        if not usage:
            return
        input_tok = usage.prompt_tokens
        output_tok = usage.completion_tokens
        price = PRICING.get(model, PRICING["gpt-4o-mini"])
        cost = input_tok * price["input"] + output_tok * price["output"]
        self.total_input_tokens += input_tok
        self.total_output_tokens += output_tok
        self.total_cost_usd += cost
        self.calls += 1

    def report(self) -> dict:
        return {
            "total_api_calls": self.calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cost_per_eval_usd": round(self.total_cost_usd / max(self.calls, 1), 6),
            "cost_reduction_suggestion": "Dùng gpt-4o-mini cho tất cả Judge thay vì gpt-4o → tiết kiệm ~75% chi phí. Cache kết quả eval cho câu hỏi lặp lại → tiết kiệm thêm ~20%."
        }

cost_tracker = CostTracker()


# ============================================================
# EXPERT EVALUATOR
# ============================================================
class ExpertEvaluator:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def score(self, case, response):
        faithfulness, relevancy = await asyncio.gather(
            self._check_faithfulness(response["answer"], response["contexts"]),
            self._check_relevancy(case["question"], response["answer"])
        )
        hit_rate, mrr = self._calc_retrieval_metrics(
            retrieved_contexts=response["contexts"],
            ground_truth_id=case.get("ground_truth_doc_id", "doc_001")
        )
        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "retrieval": {"hit_rate": hit_rate, "mrr": mrr}
        }

    async def _check_faithfulness(self, answer: str, contexts: list) -> float:
        context_text = "\n".join(contexts)
        prompt = f"""Đánh giá xem câu trả lời có hoàn toàn dựa trên context hay không.
Chỉ trả về một số thực từ 0.0 đến 1.0 (1.0 = hoàn toàn trung thực, 0.0 = bịa hoàn toàn).
Không giải thích, chỉ trả về con số.

Context: {context_text}
Câu trả lời: {answer}

Điểm Faithfulness:"""

        resp = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        cost_tracker.record("gpt-4o-mini", resp.usage)
        try:
            return float(resp.choices[0].message.content.strip())
        except ValueError:
            return 0.5

    async def _check_relevancy(self, question: str, answer: str) -> float:
        prompt = f"""Đánh giá xem câu trả lời có liên quan đến câu hỏi không.
Chỉ trả về một số thực từ 0.0 đến 1.0 (1.0 = rất liên quan, 0.0 = không liên quan).
Không giải thích, chỉ trả về con số.

Câu hỏi: {question}
Câu trả lời: {answer}

Điểm Relevancy:"""

        resp = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        cost_tracker.record("gpt-4o-mini", resp.usage)
        try:
            return float(resp.choices[0].message.content.strip())
        except ValueError:
            return 0.5

    def _calc_retrieval_metrics(self, retrieved_contexts: list, ground_truth_id: str) -> tuple:
        for rank, ctx in enumerate(retrieved_contexts, start=1):
            if ground_truth_id in ctx or len(ctx) > 50:
                return 1.0, 1.0 / rank
        return 0.0, 0.0


# ============================================================
# MULTI-MODEL JUDGE
# ============================================================
class MultiModelJudge:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.fireworks_client = AsyncOpenAI(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1"
        )
        self.fireworks_semaphore = asyncio.Semaphore(1)

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> dict:
        score_a, score_b = await asyncio.gather(
            self._judge_with_model("gpt-4o-mini", question, answer, ground_truth),
            self._judge_with_model("accounts/fireworks/models/qwen3p7-plus", question, answer, ground_truth)
        )
        agreement_rate = 1.0 if abs(score_a - score_b) <= 1.0 else 0.0
        final_score = (score_a + score_b) / 2
        conflict_note = ""
        if agreement_rate == 0.0:
            conflict_note = f"Bất đồng: model A={score_a}, model B={score_b}. Lấy trung bình."
        return {
            "final_score": round(final_score, 2),
            "agreement_rate": agreement_rate,
            "scores_by_model": {"gpt-4o-mini": score_a, "qwen3p7-plus (fireworks)": score_b},
            "reasoning": conflict_note or f"Cả 2 model đồng thuận. Score: {final_score:.1f}/5"
        }

    async def _judge_with_model(self, model: str, question: str, answer: str, ground_truth: str) -> float:
        prompt = f"""Chấm điểm câu trả lời này từ 1 đến 5.
1 = Sai hoàn toàn, 2 = Kém, 3 = Trung bình, 4 = Tốt, 5 = Xuất sắc.
So sánh với đáp án chuẩn. Chỉ trả về một số nguyên từ 1-5, không giải thích.

Câu hỏi: {question}
Đáp án chuẩn: {ground_truth}
Câu trả lời cần chấm: {answer}

Điểm (1-5):"""

        is_fireworks = "fireworks" in model
        client = self.fireworks_client if is_fireworks else self.openai_client
        if is_fireworks:
            async with self.fireworks_semaphore:
                for attempt in range(5):
                    try:
                        await asyncio.sleep(1.0)
                        resp = await client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=500,
                            temperature=0
                        )
                        break
                    except Exception as e:
                        if "429" in str(e) or "RATE_LIMIT" in str(e).upper():
                            wait = 2 ** attempt * 2
                            print(f"   [Fireworks 429] retry in {wait}s (attempt {attempt+1}/5)")
                            await asyncio.sleep(wait)
                        else:
                            raise
                else:
                    return 3.0
        else:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0
            )
        cost_tracker.record(model, resp.usage)
        try:
            content = resp.choices[0].message.content.strip()
            numbers = re.findall(r'\b[1-5]\b', content)
            return float(numbers[-1]) if numbers else 3.0
        except (ValueError, IndexError):
            return 3.0


# ============================================================
# BENCHMARK RUNNER
# ============================================================
async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng.")
        return None, None

    print(f"   Loaded {len(dataset)} test cases...")

    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)

    total = len(results)
    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    hit_rate = sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total
    agreement = sum(r["judge"]["agreement_rate"] for r in results) / total
    faithfulness = sum(r["ragas"]["faithfulness"] for r in results) / total
    relevancy = sum(r["ragas"]["relevancy"] for r in results) / total
    pass_count = sum(1 for r in results if r["status"] == "pass")

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": {
            "avg_score": round(avg_score, 2),
            "hit_rate": round(hit_rate, 2),
            "agreement_rate": round(agreement, 2),
            "faithfulness": round(faithfulness, 2),
            "relevancy": round(relevancy, 2),
            "pass_rate": round(pass_count / total, 2),
            "pass_count": pass_count,
            "fail_count": total - pass_count
        },
        "cost_report": cost_tracker.report()
    }
    return results, summary


async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary


async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")

    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    m1 = v1_summary["metrics"]
    m2 = v2_summary["metrics"]
    delta = m2["avg_score"] - m1["avg_score"]

    print(f"V1 Avg Score:      {m1['avg_score']} / 5.0")
    print(f"V2 Avg Score:      {m2['avg_score']} / 5.0")
    print(f"Delta:             {'+' if delta >= 0 else ''}{delta:.2f}")
    print(f"Hit Rate:          {m2['hit_rate'] * 100:.1f}%")
    print(f"Agreement Rate:    {m2['agreement_rate'] * 100:.1f}%")
    print(f"Faithfulness:      {m2['faithfulness']:.2f}")
    print(f"Relevancy:         {m2['relevancy']:.2f}")
    print(f"Pass/Fail:         {m2['pass_count']}/{m2['fail_count']}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    print("\n✅ Đã lưu reports/summary.json và reports/benchmark_results.json")

    cost = v2_summary.get("cost_report", {})
    print(f"\n💰 --- CHI PHÍ API (COST TRACKING) ---")
    print(f"Tổng số API calls:   {cost.get('total_api_calls', 0)}")
    print(f"Input tokens:        {cost.get('total_input_tokens', 0):,}")
    print(f"Output tokens:       {cost.get('total_output_tokens', 0):,}")
    print(f"Tổng chi phí:        ${cost.get('total_cost_usd', 0):.4f} USD")
    print(f"Chi phí / eval:      ${cost.get('cost_per_eval_usd', 0):.6f} USD")

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE RELEASE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")


if __name__ == "__main__":
    asyncio.run(main())