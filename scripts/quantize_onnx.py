from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_model", default="model.onnx")
    ap.add_argument("--out_model", default="model.quant.onnx")
    args = ap.parse_args()

    quantize_dynamic(args.in_model, args.out_model, weight_type=QuantType.QInt8)
    print(f"Quantized model saved to {args.out_model}")


if __name__ == "__main__":
    main()
