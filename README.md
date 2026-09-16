# 🔭 Vision Transformer Projects

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-FFD21E)
![License](https://img.shields.io/badge/License-MIT-green)
![Demo](https://img.shields.io/badge/🚀_Live_Demo-HF_Spaces-blue)

**End-to-end computer vision projects using transformer architectures — from object detection to model compression.**

[**🖥️ Live Interactive Demo →**](https://huggingface.co/spaces/PRaruj/vision-transformer-projects) &nbsp;|&nbsp;
[**📄 My Portfolio →**](https://www.prarujthapa.com.np) &nbsp;|&nbsp;
[**🔗 GitHub →**](https://github.com/PRaruj)

</div>

---

## 📌 Projects at a Glance

| # | Project | Task | Key Technique | Key Result |
|---|---------|------|---------------|------------|
| 1 | [🦺 DETR Hardhat Detection](#-project-1--detr-hardhat-object-detection) | Object Detection | Bipartite Matching | **Loss: `1.35`** |
| 2 | [🐾 ViT + Knowledge Distillation](#-project-2--vit--knowledge-distillation-oxford-pets) | Model Compression | Teacher-Student KD | **Accuracy: `11.6%` → `37.6%`** |

> Both projects share a **unified live demo** — upload your own image and run inference in the browser, no installation required.

---

## 🦺 Project 1 — DETR Hardhat Object Detection

### Overview
Fine-tuned [DETR (DEtection TRansformer)](https://arxiv.org/abs/2005.12872) with a ResNet-50 backbone to detect **hardhats in workplace images**, solving a real-world safety compliance use case.

DETR reframes object detection as a **set prediction problem**, eliminating the need for hand-crafted anchors and Non-Maximum Suppression (NMS).

### 📈 Training Performance
The model was fine-tuned over 1,000 steps, demonstrating stable convergence using the Hungarian algorithm for loss calculation.
*   **Final Training Loss:** `1.35`
*   **Throughput:** `3.96` samples/second
*   **Total FLOPs:** `3.81e+18`

### Architecture

```text
INPUT IMAGE
    │
    ▼
┌────────────┐
│  ResNet-50 │  ← CNN Backbone: visual feature extractor
│  Backbone  │    (edges → textures → objects)
└─────┬──────┘
      │  [B, 2048, H, W]  →  1×1 Conv  →  [B, 256, H/32, W/32]
      ▼
┌────────────┐
│ Positional │  ← Row + Column embeddings (spatial GPS)
│ Encodings  │
└─────┬──────┘
      ▼
┌────────────┐
│ Transformer│  ← Global self-attention across all image patches
│   Encoder  │    "This region + that region = same object"
└─────┬──────┘
      │
      │  + 100 learnable Object Queries
      ▼
┌────────────┐
│ Transformer│  ← Each query finds one unique object
│   Decoder  │
└─────┬──────┘
      ▼
┌──────────────────────────────────┐
│  Prediction Heads                │
│  ├── Class Head (softmax)        │  → "hardhat" / "no object"
│  └── BBox Head (sigmoid) × 4     │  → [cx, cy, w, h] normalized
└─────┬────────────────────────────┘
      ▼
┌────────────┐
│  Bipartite │  ← Hungarian Algorithm: match predictions ↔ ground truth
│  Matching  │    (order-invariant, no NMS needed!)
└────────────┘
```

### Why DETR Over Traditional Detectors?

| Feature | Anchor-Based (YOLO, Faster R-CNN) | DETR |
|---|---|---|
| Anchors | ✅ Required | ❌ Anchor-free |
| NMS Post-processing | ✅ Required | ❌ Not needed |
| Global Context | ❌ Local receptive field | ✅ Full self-attention |
| Predictions | Ordered, overlapping | **Set** — unique, permutation-invariant |

### 📓 [View Full Notebook →](01_detr_hardhat_detection/Fine_tuning_vit_object_detection_detr_hardhat.ipynb)

---

## 🐾 Project 2 — ViT + Knowledge Distillation (Oxford Pets)

### Overview
Two-phase experiment on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (7,390 images, 37 breeds) to demonstrate **model compression without increasing parameter count**.

### 📈 The Results (The Power of KD)
Training a fresh ViT model from scratch on 37 complex classes is difficult. By using a large, pre-trained teacher model to provide "soft labels," we dramatically accelerated learning.

*   **Baseline Accuracy (10 epochs):** `11.6%`
*   **Distilled Student Accuracy (10 epochs):** **`37.6%`**
*   *Note: Random guessing yields ~2.7%. The distilled model achieved more than triple the accuracy in the exact same timeframe.*

### What is Knowledge Distillation?

```text
TEACHER MODEL (large, pretrained, frozen)
    │
    │  produces soft labels (probability distributions)
    │  e.g., Persian: 0.70 | Siamese: 0.20 | Maine Coon: 0.08 | ...
    │  (richer signal than one-hot: [1, 0, 0, ...])
    ▼
DISTILLATION LOSS
    = α × CrossEntropy(student_logits, hard_labels)
    + (1-α) × KLDiv(student_logits/T, teacher_logits/T)
    │
    │  T = temperature (softens distributions to reveal inter-class structure)
    │  α = balancing coefficient
    ▼
STUDENT MODEL (same ViT architecture, trained with teacher guidance)
    │
    ▼
Result: Student learns richer inter-class relationships from soft labels
        → Better generalization than training on hard labels alone
```

### 📓 [View Full Notebook →](02_vit_knowledge_distillation/03_10_knowledge_distillation_exercise.ipynb)

---

## 🖥️ Live Demo

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/PRaruj/vision-transformer-projects)

**Tab 1 — 🦺 DETR Hardhat Detector**
- Upload any workplace/construction image
- Model draws bounding boxes around hardhats with confidence scores

**Tab 2 — 🐾 ViT KD Comparison (Oxford Pets)**
- Upload any pet image
- See Top-5 predictions from **both** models side-by-side
- Observe how KD model produces more calibrated probability distributions

### Run Locally

```bash
git clone https://github.com/PRaruj/vision-transformer-projects
cd vision-transformer-projects
pip install -r requirements.txt
python demo/app.py
# Open http://localhost:7860
```

---

## 🛠️ Tech Stack

```python
stack = {
    "Models":      ["DETR", "ResNet-50", "ViT (from scratch)", "ViT (KD-compressed)"],
    "Frameworks":  ["PyTorch", "HuggingFace Transformers", "HuggingFace Datasets"],
    "Concepts":    ["Object Detection", "Set Prediction", "Bipartite Matching",
                    "Knowledge Distillation", "Vision Transformers", "Model Compression"],
    "Demo":        ["Gradio", "HuggingFace Spaces"]
}
```

---

## 📁 Repository Structure

```text
vision-transformer-projects/
├── README.md                                        ← You are here
├── 01_detr_hardhat_detection/
│   ├── README.md                                    ← Deep-dive: DETR architecture & training
│   └── Fine_tuning_vit_object_detection_detr_hardhat.ipynb ← Full training notebook
├── 02_vit_knowledge_distillation/
│   ├── README.md                                    ← Deep-dive: KD theory & results
│   └── 03_10_knowledge_distillation_exercise.ipynb
├── demo/
│   ├── app.py                                       ← Combined Gradio demo (2 tabs)
│   └── requirements.txt
├── scripts/
│   └── upload_models_to_hf.py                       ← Upload model weights to HF Hub
├── requirements.txt
└── .gitignore
```

---

## 🧠 Skills Demonstrated

| Skill | Evidence in This Repo |
|-------|----------------------|
| **Transformer Architecture** | DETR encoder-decoder + ViT patch embeddings from scratch |
| **Object Detection** | Bipartite matching, Hungarian algorithm, permutation-invariant set prediction |
| **Vision Transformers** | Manual patch splitting, CLS token, positional encoding, MHA blocks |
| **Knowledge Distillation** | Temperature scaling, KL divergence loss, teacher-student curriculum |
| **Model Compression** | Same model size, better generalization via soft label supervision |
| **HuggingFace Ecosystem** | Transformers, Datasets, Hub model upload, Spaces deployment |
| **Production Demo** | Gradio UI with multi-tab layout, deployable to HF Spaces |

---

<details>
<summary>🇯🇵 日本語でサクッと読む（クリックして展開 / Click to view Japanese summary）</summary>

## 日本語概要

このリポジトリは、2つのコンピュータビジョン × トランスフォーマーアーキテクチャのプロジェクトをまとめたポートフォリオショーケースです。

### プロジェクト1：DETR ヘルメット検出
- DETR（Detection Transformer）をResNet-50バックボーンと組み合わせ、作業現場画像からヘルメットを検出するモデルをファインチューニング。
- **成果**: ハンガリーアルゴリズムを用いた二部マッチングにより、最終学習Loss `1.35` を達成。
- スループット: `3.96` samples/second

### プロジェクト2：ViT × 知識蒸留（Oxford Pets）
- 大型の事前学習済みViTを教師モデルとし、知識蒸留でスチューデントモデルを改善。
- **成果**: スクラッチ学習の精度（`11.6%`）を、同一エポック数で **`37.6%`** まで大幅に向上。

### デモ
Hugging Face Spaces上でブラウザからインタラクティブに推論が可能です。

</details>
