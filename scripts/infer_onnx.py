import onnxruntime as ort
import argparse
import json
import time
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="model.quant.onnx")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("out")

    sess = ort.InferenceSession(args.onnx)

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    times = []
    for i in range(args.runs):
        txt = texts[i % len(texts)]
        enc = tokenizer(
            txt,
            return_tensors="np",
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        start = time.perf_counter()
        sess.run(
            None,
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
        )
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times_sorted = sorted(times)
    p50 = times_sorted[len(times) // 2]
    p95 = times_sorted[int(len(times) * 0.95)]

    print(f"ONNX p50 = {p50:.2f} ms")
    print(f"ONNX p95 = {p95:.2f} ms")


if __name__ == "__main__":
    main()
