import time
import argparse
import onnxruntime as ort
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=48)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    ort.set_default_logger_severity(3)

    session = ort.InferenceSession(
        f"{args.model_dir}/model.quant.onnx",
        sess_options,
        providers=["CPUExecutionProvider"],
    )

    with open(args.input, "r", encoding="utf-8") as f:
        sample = f.readline()

    encoded = tokenizer(
        sample,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    )

    print("Warming up...")
    for _ in range(20):
        session.run(
            ["logits"],
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
        )

    print(f"Benchmarking for {args.runs} runs...")

    times = []
    for _ in range(args.runs):
        s = time.perf_counter()
        session.run(
            ["logits"],
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
        )
        times.append((time.perf_counter() - s) * 1000)

    p50 = sorted(times)[len(times) // 2]
    p95 = sorted(times)[int(len(times) * 0.95)]

    print(f"Latency over {args.runs} runs:")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")


if __name__ == "__main__":
    main()
