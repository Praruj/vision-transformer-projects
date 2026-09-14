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

| # | Project | Task | Key Technique | Model |
|---|---------|------|---------------|-------|
| 1 | [🦺 DETR Hardhat Detection](#-project-1--detr-hardhat-object-detection) | Object Detection | Bipartite Matching, Hungarian Algorithm | DETR + ResNet-50 |
| 2 | [🐾 ViT + Knowledge Distillation](#-project-2--vit--knowledge-distillation-oxford-pets) | Image Classification + Model Compression | Teacher-Student KD, ViT from Scratch | ViT (Base & KD-compressed) |

> Both projects share a **unified live demo** — upload your own image and run inference in the browser, no installation required.

---

## 🦺 Project 1 — DETR Hardhat Object Detection

### Overview
Fine-tuned [DETR (DEtection TRansformer)](https://arxiv.org/abs/2005.12872) with a ResNet-50 backbone to detect **hardhats in workplace images**, solving a real-world safety compliance use case.

DETR reframes object detection as a **set prediction problem**, eliminating the need for hand-crafted anchors and Non-Maximum Suppression (NMS).

### Architecture

```
INPUT IMAGE
    │
    ▼
┌────────────┐
│  ResNet-50  │  ← CNN Backbone: visual feature extractor
│  Backbone   │    (edges → textures → objects)
└─────┬──────┘
      │  [B, 2048, H, W]  →  1×1 Conv  →  [B, 256, H/32, W/32]
      ▼
┌────────────┐
│ Positional  │  ← Row + Column embeddings (spatial GPS)
│ Encodings  │
└─────┬──────┘
      ▼
┌────────────┐
│ Transformer │  ← Global self-attention across all image patches
│   Encoder   │    "This region + that region = same object"
└─────┬──────┘
      │
      │  + 100 learnable Object Queries
      ▼
┌────────────┐
│ Transformer │  ← Each query finds one unique object
│   Decoder   │
└─────┬──────┘
      ▼
┌──────────────────────────────────┐
│  Prediction Heads                │
│  ├── Class Head (softmax)        │  → "hardhat" / "no object"
│  └── BBox Head (sigmoid) × 4    │  → [cx, cy, w, h] normalized
└─────┬────────────────────────────┘
      ▼
┌────────────┐
│  Bipartite  │  ← Hungarian Algorithm: match predictions ↔ ground truth
│  Matching   │    (order-invariant, no NMS needed!)
└────────────┘
```

### Why DETR Over Traditional Detectors?

| | Anchor-Based (YOLO, Faster R-CNN) | DETR |
|--|------|------|
| Anchors | ✅ Required | ❌ Anchor-free |
| NMS Post-processing | ✅ Required | ❌ Not needed |
| Global Context | ❌ Local receptive field | ✅ Full self-attention |
| Architecture | CNN only | CNN + Transformer |
| Predictions | Ordered, overlapping | **Set** — unique, permutation-invariant |

### Key Concepts
- **Backbone**: ResNet-50 (pretrained) → 2048 channels → 1×1 Conv → 256 channels (hidden_dim)
- **Positional Embeddings**: Learned row + column embeddings concatenated per spatial location
- **Object Queries**: 100 learned vectors; queries specialize in finding different objects during training
- **Bipartite Matching Loss**: Hungarian algorithm aligns predictions to ground truth without ordering

### 📓 [View Full Notebook →](01_detr_hardhat_detection/03_07_DETR.ipynb)

---

## 🐾 Project 2 — ViT + Knowledge Distillation (Oxford Pets)

### Overview
Two-phase experiment on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (7,390 images, 37 breeds):

1. **Phase 1**: Train a Vision Transformer (ViT) from scratch — establish baseline
2. **Phase 2**: Apply **Knowledge Distillation** using a large pretrained ViT as teacher — improve student with same architecture

This demonstrates **model compression without increasing parameter count**: same size, better performance.

### What is Knowledge Distillation?

```
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

### ViT Architecture (from Scratch)

```
INPUT IMAGE (224×224×3)
    │
    ▼  Divide into 16×16 patches → 196 patches
┌─────────────────┐
│ Linear Projection│  [196, 768]  (patch embeddings)
└────────┬────────┘
         │  Prepend [CLS] token  →  [197, 768]
         │  Add Positional Embeddings (learned)
         ▼
┌─────────────────┐
│  Transformer    │  × N encoder blocks:
│  Encoder Blocks │    LayerNorm → Multi-Head Self-Attention → LayerNorm → FFN
└────────┬────────┘
         │  Extract [CLS] token (global representation)
         ▼
┌─────────────────┐
│   MLP Head      │  → 37 output classes (pet breeds)
└─────────────────┘
```

### Experiment: Base vs KD Model

| Model | Architecture | Training Signal |
|-------|-------------|-----------------|
| ViT from Scratch | ViT (custom) | Hard one-hot labels only |
| ViT with KD | ViT (same) | Hard labels + Teacher soft labels |

> **KD advantage**: The student learns *which breeds look similar* from the teacher's soft distributions, not just *which class is correct*. This improves calibration and generalization.

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
    "Demo":        ["Gradio", "HuggingFace Spaces"],
    "Data":        ["Custom hardhat dataset", "Oxford-IIIT Pets (7,390 images, 37 breeds)"]
}
```

---

## 📁 Repository Structure

```
vision-transformer-projects/
├── README.md                                        ← You are here
├── 01_detr_hardhat_detection/
│   ├── README.md                                    ← Deep-dive: DETR architecture & training
│   └── 03_07_DETR.ipynb                             ← Full training notebook
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

## 👤 Author

**Praruj Thapa** · タパ プラルズ · AI/ML Engineer | Data Scientist
📍 Osaka, Japan · 🎓 Kyoto University of Advanced Science (B.E. 2025)

> *"Data is just the world, speaking in numbers. I build systems that listen."*

[![GitHub](https://img.shields.io/badge/GitHub-PRaruj-181717?logo=github)](https://github.com/PRaruj)
[![Portfolio](https://img.shields.io/badge/Portfolio-prarujthapa.com.np-4A90D9)](https://www.prarujthapa.com.np/articles/)

---

<details>
<summary>🇯🇵 日本語でサクッと読む（クリックして展開 / Click to view Japanese summary）</summary>

## 日本語概要

このリポジトリは、2つのコンピュータビジョン × トランスフォーマーアーキテクチャのプロジェクトをまとめたポートフォリオショーケースです。

### プロジェクト1：DETR ヘルメット検出
- DETR（Detection Transformer）をResNet-50バックボーンと組み合わせ、作業現場画像からヘルメットを検出するモデルをファインチューニング
- アンカーフリー・NMS不要のセット予測アプローチを採用
- ハンガリーアルゴリズムによる二部マッチングで予測と正解を対応付け

### プロジェクト2：ViT × 知識蒸留（Oxford Pets）
- Vision Transformerをスクラッチから構築しベースラインを確立
- 大型の事前学習済みViTを教師モデルとし、知識蒸留でスチューデントモデルを改善
- モデルサイズを変えずに汎化性能を向上させるモデル圧縮技術を実証

### デモ
Hugging Face Spaces上でブラウザからインタラクティブに推論が可能です。

**著者**：タパ プラルズ（AI/MLエンジニア、大阪在住、京都先端科学大学 工学部 2025年卒）

</details>
