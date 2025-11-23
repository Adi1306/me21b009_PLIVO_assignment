import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--onnx_path", required=True)
    args = ap.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()
    model.to("cpu")
    torch.set_num_threads(1)

    # Dummy input
    sample = "hello how are you"
    dummy = tokenizer(
        sample,
        return_tensors="pt",
        max_length=64,  
        padding="max_length",
        truncation=True,
    )

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        args.onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print("Saved ONNX model:", args.onnx_path)


if __name__ == "__main__":
    main()

