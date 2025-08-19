from openai import OpenAI
import datasets
import re
import tqdm
import json
import argparse
import os
import random
import time

# ===== 
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='7777')
parser.add_argument('--model_name', type=str, default='Qwen3-1.7B')
parser.add_argument('--shots', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out_dir', type=str, default='mmlu_outputs')
parser.add_argument('--rate_limit', type=float, default=0.0)
args = parser.parse_args()

port = args.port
model_name = args.model_name

# =====
client = OpenAI(
    api_key="sk-2ebed1ea66a2488580174fe62768a811",
    base_url="https://api.deepseek.com/v1"
)

def llm_output(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": text}],
            max_tokens=512,
            temperature=0.0,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"API 调用失败：{str(e)}")

# 官方评测
LETTER_RE = re.compile(r"\b([A-D])\b", flags=re.IGNORECASE)
random.seed(args.seed)

def extract_letter(s):
    if not s:
        return None
    m = LETTER_RE.findall(s.strip())
    return m[-1].upper() if m else None

def format_example(ex, include_answer=True):
    q = ex['question']
    c = ex['choices']
    a = ex['answer']
    body = f"Question: {q}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer: "
    return body + (a if include_answer else "")

def build_fewshot_prompt(dev_examples, k):
    k = min(k, len(dev_examples))
    return "\n\n".join(format_example(ex, True) for ex in dev_examples[:k]) + "\n\n"

def normalize_record(rec):
    q = rec.get("question", rec.get("prompt", ""))
    choices = rec.get("choices", [
        rec.get("A", ""), rec.get("B", ""), rec.get("C", ""), rec.get("D", "")
    ])
    ans = rec.get("answer", rec.get("label", ""))
    if isinstance(ans, int):
        ans = "ABCD"[ans]
    return {"question": q, "choices": choices, "answer": str(ans).strip().upper()}

def load_split(subject: str, split: str):
    if split == "dev":
        return datasets.load_dataset("cais/mmlu", subject, split="validation")
    return datasets.load_dataset("cais/mmlu", subject, split=split)


def evaluate_subject(subject, shots):
    dev = [normalize_record(x) for x in load_split(subject, "dev")]
    test = [normalize_record(x) for x in load_split(subject, "test")]

    random.shuffle(dev)
    fewshot = build_fewshot_prompt(dev, shots)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{subject}.jsonl")

    correct, total = 0, 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in tqdm.tqdm(test, desc=subject):
            prompt = fewshot + format_example(ex, include_answer=False)
            pred_text = llm_output(prompt)
            pred_letter = extract_letter(pred_text)
            gold = ex["answer"]
            is_correct = int(pred_letter == gold)
            correct += is_correct
            total += 1

            fout.write(json.dumps({
                "subject": subject,
                "question": ex["question"],
                "choices": ex["choices"],
                "gold": gold,
                "model_raw": pred_text,
                "pred": pred_letter,
                "correct": is_correct
            }, ensure_ascii=False) + "\n")

            if args.rate_limit > 0:
                time.sleep(args.rate_limit)

    acc = correct / total if total else 0.0
    return acc, correct, total

if __name__ == "__main__":
    subjects = datasets.get_dataset_config_names("cais/mmlu")
    subjects.sort()

    per_subject = {}
    micro_c, micro_t = 0, 0

    for s in subjects:
        try:
            acc, c, n = evaluate_subject(s, args.shots)
            per_subject[s] = {"acc": acc, "correct": c, "total": n}
            micro_c += c
            micro_t += n
            print(f"[{s}] acc={acc:.4f} ({c}/{n})")
        except Exception as e:
            per_subject[s] = {"error": str(e)}
            print(f"[{s}] ERROR: {e}")

    valid = [v["acc"] for v in per_subject.values() if "acc" in v]
    macro = (sum(valid) / len(valid)) if valid else 0.0
    micro = (micro_c / micro_t) if micro_t else 0.0

    summary = {
        "model": model_name,
        "shots": args.shots,
        "micro_acc": micro,
        "macro_acc": macro,
        "per_subject": per_subject
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nMicro accuracy: {micro:.4%} ({micro_c}/{micro_t})")
    print(f"Macro accuracy: {macro:.4%}")
    print(f"Results saved to {args.out_dir}")


