import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--onnx_path", default="model.onnx")
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()

    dummy = tokenizer(
        "hello world",
        return_tensors="pt",
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
    )

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        args.onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model with dynamic axes to {args.onnx_path}")


if __name__ == "__main__":
    main()
