# PII Named Entity Recognition — DistilBERT (Fine-Tuned + ONNX INT8)

This repository contains my submission for the **PLIVO AI/ML Engineer Assignment**, where the goal was to build a **PII Named Entity Recognition (NER)** system optimized for:

- High precision/recall  
- Fast inference (<20ms latency)  
- ONNX Runtime INT8 optimization  
- Low-resource CPU environment  

## 1. Model & Tokenizer Used

**Base Model:** distilbert-base-uncased  
**Tokenizer:** AutoTokenizer.from_pretrained("distilbert-base-uncased")

## 2. Training Setup

### Training Command:
```
python src/train.py --model_name distilbert-base-uncased --train data/train.jsonl --dev data/dev.jsonl --out_dir out_tiny --batch_size 16 --epochs 5 --max_length 32 --lr 3e-5
```

### Hyperparameters:
- Batch Size: 16  
- Epochs: 5  
- LR: 3e-5  
- Max Length: 32  

## 3. Final Metrics
```
Macro-F1: 0.998
PII F1: 1.000
```

## 4. ONNX Runtime + INT8 Optimization

### Export:
```
python scripts/export_onnx.py --model_dir out_tiny --onnx_path out_tiny/model.onnx
```

### Quantize:
```
python scripts/quantize_onnx.py --in_model out_tiny/model.onnx --out_model out_tiny/model.quant.onnx
```

## 5. Latency Results
```
p50 = 10.78 ms
p95 = 11.88 ms
```

## 6. How to Run

### Predict:
```
python src/predict.py --model_dir out_tiny --input data/dev.jsonl --output out_tiny/dev_pred.json
```

### Evaluate:
```
python src/eval_span_f1.py --gold data/dev.jsonl --pred out_tiny/dev_pred.json
```

### ONNX Latency:
```
python scripts/infer_onnx.py --onnx out_tiny/model.quant.onnx --input data/dev.jsonl --runs 50 --max_length 32
```

 

## 7. Loom Video Checklist
- Final results  
- Code explanation  
- Model + tokenizer  
- Hyperparameters  
- Metrics  
- Latency tradeoffs  
