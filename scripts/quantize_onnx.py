import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_model", required=True)
    ap.add_argument("--out_model", required=True)
    args = ap.parse_args()

    print("Quantizing model...")
    quantize_dynamic(
        args.in_model,
        args.out_model,
        weight_type=QuantType.QInt8,
        optimize_model=True,  # IMPORTANT for speed
    )

    print("Saved INT8 model:", args.out_model)


if __name__ == "__main__":
    main()
