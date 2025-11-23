import json
import time
import argparse
import onnxruntime as ort
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--max_length", type=int, default=48)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("out_finetuned")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    ort.set_default_logger_severity(3)  # disable ORT logging

    session = ort.InferenceSession(
        args.onnx, sess_options, providers=["CPUExecutionProvider"]
    )

    # Load sample texts
    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            if len(texts) >= 300:
                break

    encoded = [
        tokenizer(
            t,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        for t in texts
    ]

    print("Warming up...")
    for _ in range(20):
        session.run(
            ["logits"],
            {
                "input_ids": encoded[0]["input_ids"],
                "attention_mask": encoded[0]["attention_mask"],
            },
        )

    print(f"Running inference on {len(encoded)} examples for {args.runs} runs...")

    p50_list = []
    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(
            ["logits"],
            {
                "input_ids": encoded[0]["input_ids"],
                "attention_mask": encoded[0]["attention_mask"],
            },
        )
        p50_list.append((time.perf_counter() - start) * 1000)

    p50 = sorted(p50_list)[len(p50_list) // 2]
    p95 = sorted(p50_list)[int(len(p50_list) * 0.95)]

    print(f"ONNX p50 = {p50:.2f} ms")
    print(f"ONNX p95 = {p95:.2f} ms")


if __name__ == "__main__":
    main()
