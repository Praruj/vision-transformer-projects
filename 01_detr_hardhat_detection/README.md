# 🦺 DETR Hardhat Object Detection

> **Part of the [Vision Transformer Projects](../README.md) portfolio**

Fine-tuned **DETR (DEtection TRansformer)** with a ResNet-50 backbone to detect hardhats in construction/workplace images.

## 📓 Notebook

[**Open 03_07_DETR.ipynb →**](03_07_DETR.ipynb)

The notebook covers:
- DETR architecture walkthrough (CNN backbone → Encoder → Decoder → Prediction Heads)
- Why transformers for object detection (global context, no anchors, no NMS)
- Positional embeddings: row + column embeddings as spatial GPS
- 100 object queries as learned "detectives"
- Bipartite matching (Hungarian algorithm) for permutation-invariant loss
- Fine-tuning on hardhat dataset using HuggingFace Transformers

## 🏗️ Architecture Summary

```
IMAGE → ResNet-50 → 1×1 Conv → Positional Encoding
      → Transformer Encoder (global context)
      → Transformer Decoder (100 object queries)
      → Class Head + BBox Head
      → Hungarian Matching → Final detections
```

## 🔑 Key Concepts

| Concept | What It Solves |
|---------|----------------|
| **Set prediction** | Eliminates duplicate/overlapping box problem (no NMS needed) |
| **Bipartite matching** | Order-invariant loss: model doesn't care *which* query finds the object |
| **Global self-attention** | "This ear + that body = same cat" — long-range dependencies |
| **Object queries** | Learned slots that specialize during training to find specific object types |

## 📦 Model Weights

Model hosted on HuggingFace Hub: [`PRaruj/detr-resnet50-hardhat-finetuned`](https://huggingface.co/PRaruj/detr-resnet50-hardhat-finetuned)

## 🔗 References

- [DETR Paper: End-to-End Object Detection with Transformers (Carion et al., 2020)](https://arxiv.org/abs/2005.12872)
- [HuggingFace DETR Documentation](https://huggingface.co/docs/transformers/model_doc/detr)
