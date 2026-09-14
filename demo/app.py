"""
Vision Transformer Projects — Interactive Gradio Demo
Author: Praruj Thapa (https://github.com/PRaruj)

Tab 1: DETR Hardhat Object Detection
Tab 2: ViT Knowledge Distillation — Oxford Pets (Base vs KD model comparison)

To run locally:
    pip install -r requirements.txt
    python demo/app.py

Model weights are loaded from Hugging Face Hub.
Set HF_MODEL_DETR and HF_MODEL_VIT env vars to override model IDs.
"""

import os
import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Model IDs (override via environment variables) ────────────────────────────
DETR_MODEL_ID  = os.getenv("HF_MODEL_DETR", "PRaruj/detr-resnet50-hardhat-finetuned")
VIT_BASE_ID    = os.getenv("HF_MODEL_VIT_BASE", "PRaruj/oxford-pets-vit-from-scratch")
VIT_KD_ID      = os.getenv("HF_MODEL_VIT_KD",   "PRaruj/oxford-pets-vit-with-kd")

# ─── Oxford Pets class labels (37 breeds) ──────────────────────────────────────
OXFORD_PETS_LABELS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx", "American Bulldog", "American Pit Bull Terrier",
    "Basset Hound", "Beagle", "Boxer", "Chihuahua", "English Cocker Spaniel",
    "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese",
    "Japanese Chin", "Keeshond", "Leonberger", "Miniature Pinscher",
    "Newfoundland", "Pomeranian", "Pug", "Saint Bernard", "Samoyed",
    "Scottish Terrier", "Shiba Inu", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier"
]

COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
]


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETR HARDHAT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

detr_model = None
detr_processor = None

def load_detr_model():
    global detr_model, detr_processor
    if detr_model is None:
        try:
            from transformers import DetrForObjectDetection, DetrImageProcessor
            print(f"Loading DETR model from: {DETR_MODEL_ID}")
            detr_processor = DetrImageProcessor.from_pretrained(DETR_MODEL_ID)
            detr_model = DetrForObjectDetection.from_pretrained(DETR_MODEL_ID)
            detr_model.eval()
            print("✅ DETR model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load fine-tuned DETR model: {e}")
            print("Falling back to base DETR (COCO, no hardhat classes)...")
            from transformers import DetrForObjectDetection, DetrImageProcessor
            detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            detr_model.eval()
    return detr_model, detr_processor


def draw_boxes(image: Image.Image, results, threshold: float = 0.5):
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    detections = []

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        score_val = score.item()
        if score_val < threshold:
            continue

        label_id = label.item()
        label_name = detr_model.config.id2label.get(label_id, f"class_{label_id}")
        cx, cy, bw, bh = box.tolist()

        # Convert from centre-relative to pixel coordinates
        x0 = int((cx - bw / 2) * w)
        y0 = int((cy - bh / 2) * h)
        x1 = int((cx + bw / 2) * w)
        y1 = int((cy + bh / 2) * h)

        color = COLORS[label_id % len(COLORS)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        caption = f"{label_name}: {score_val:.2f}"
        draw.rectangle([x0, y0 - 18, x0 + len(caption) * 7, y0], fill=color)
        draw.text((x0 + 2, y0 - 16), caption, fill="white")
        detections.append(f"**{label_name}** — confidence: {score_val:.1%}  (box: [{x0},{y0},{x1},{y1}])")

    return image, detections


def detect_hardhats(image: Image.Image, confidence_threshold: float):
    """Run DETR inference and return annotated image + detection summary."""
    if image is None:
        return None, "⚠️ Please upload an image."

    model, processor = load_detr_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[image.height, image.width]])
    results = processor.post_process_object_detection(
        outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]

    annotated = image.copy()
    annotated, detections = draw_boxes(annotated, results, threshold=confidence_threshold)

    if detections:
        summary = f"### 🟢 Found {len(detections)} detection(s):\n\n" + "\n\n".join(f"- {d}" for d in detections)
    else:
        summary = f"### 🔴 No objects detected above {confidence_threshold:.0%} confidence threshold."

    return annotated, summary


def build_detr_tab():
    with gr.Tab("🦺 DETR Hardhat Detector"):
        gr.Markdown("""
## 🦺 DETR Hardhat Object Detection
Fine-tuned **DETR + ResNet-50** for workplace hardhat detection.
- **No anchors, no NMS** — pure set prediction via bipartite matching
- Upload any construction/workplace image to detect hardhats
        """)

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="Upload Image")
                threshold = gr.Slider(
                    minimum=0.1, maximum=0.99, value=0.5, step=0.05,
                    label="Confidence Threshold"
                )
                detect_btn = gr.Button("🔍 Detect Hardhats", variant="primary")

            with gr.Column(scale=1):
                img_output = gr.Image(type="pil", label="Detection Result")
                summary_output = gr.Markdown(label="Detections")

        gr.Examples(
            examples=[],
            inputs=img_input,
            label="Example Images (add sample images to demo/examples/detr/)"
        )

        detect_btn.click(
            fn=detect_hardhats,
            inputs=[img_input, threshold],
            outputs=[img_output, summary_output]
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ViT KNOWLEDGE DISTILLATION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

vit_base_model = None
vit_kd_model = None
vit_processor = None


def load_vit_models():
    global vit_base_model, vit_kd_model, vit_processor

    if vit_base_model is None or vit_kd_model is None:
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor

            print(f"Loading ViT base model from: {VIT_BASE_ID}")
            vit_processor = ViTImageProcessor.from_pretrained(VIT_BASE_ID)
            vit_base_model = ViTForImageClassification.from_pretrained(VIT_BASE_ID)
            vit_base_model.eval()
            print("✅ ViT base model loaded")

            print(f"Loading ViT KD model from: {VIT_KD_ID}")
            vit_kd_model = ViTForImageClassification.from_pretrained(VIT_KD_ID)
            vit_kd_model.eval()
            print("✅ ViT KD model loaded")

        except Exception as e:
            print(f"⚠️ Could not load custom ViT models: {e}")
            print("Falling back to base google/vit-base-patch16-224 (ImageNet, not Oxford Pets)...")
            from transformers import ViTForImageClassification, ViTImageProcessor
            vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            vit_base_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
            vit_base_model.eval()
            vit_kd_model = vit_base_model  # Same in fallback mode

    return vit_base_model, vit_kd_model, vit_processor


def get_top5(logits, labels, k=5):
    """Return top-k (label, probability) pairs."""
    probs = torch.softmax(logits, dim=-1)[0]
    top_vals, top_idxs = torch.topk(probs, k)
    return [(labels[i.item()] if i.item() < len(labels) else f"class_{i.item()}", v.item())
            for i, v in zip(top_idxs, top_vals)]


def classify_pet(image: Image.Image):
    """Run both models and return comparison results."""
    if image is None:
        return None, None, "⚠️ Please upload an image."

    base_m, kd_m, processor = load_vit_models()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        base_logits = base_m(**inputs).logits
        kd_logits   = kd_m(**inputs).logits

    # Determine label list
    if hasattr(base_m.config, "id2label") and base_m.config.id2label:
        labels = [base_m.config.id2label[i] for i in range(len(base_m.config.id2label))]
    else:
        labels = OXFORD_PETS_LABELS

    base_top5 = get_top5(base_logits, labels)
    kd_top5   = get_top5(kd_logits,   labels)

    def format_top5(top5, title):
        lines = [f"## {title}\n"]
        for rank, (label, prob) in enumerate(top5, 1):
            bar = "█" * int(prob * 20)
            lines.append(f"**{rank}. {label}** — {prob:.1%}  `{bar}`")
        return "\n\n".join(lines)

    base_md = format_top5(base_top5, "🔵 ViT from Scratch (Baseline)")
    kd_md   = format_top5(kd_top5,   "🟠 ViT with Knowledge Distillation")

    note = """
---
> **💡 What to look for**: The KD model's probability distribution is often more *calibrated* —
> similar-looking breeds receive higher secondary probabilities because the student learned
> inter-class similarity from the teacher's soft labels.
    """

    return base_md, kd_md, note


def build_kd_tab():
    with gr.Tab("🐾 ViT KD Comparison"):
        gr.Markdown("""
## 🐾 Vision Transformer + Knowledge Distillation
Two models, same architecture — different training:
- **Baseline**: ViT trained from scratch on Oxford Pets
- **KD model**: Same ViT, trained with a large pretrained teacher providing soft labels

Upload a pet image to see how their predictions differ!
        """)

        with gr.Row():
            with gr.Column(scale=1):
                pet_input = gr.Image(type="pil", label="Upload Pet Image")
                classify_btn = gr.Button("🧠 Compare Models", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    base_output = gr.Markdown(label="Baseline ViT")
                    kd_output   = gr.Markdown(label="ViT + KD")
                note_output = gr.Markdown()

        gr.Markdown("""
### 🧪 Knowledge Distillation — Quick Reference

| Concept | Explanation |
|---------|------------|
| **Soft labels** | Teacher outputs probability distributions (richer than one-hot) |
| **Temperature T** | Softens distributions to reveal inter-class structure |
| **Loss** | α × CE(student, hard) + (1-α) × KL(student/T, teacher/T) |
| **Result** | Student learns *which classes are similar*, not just *which is correct* |
        """)

        classify_btn.click(
            fn=classify_pet,
            inputs=[pet_input],
            outputs=[base_output, kd_output, note_output]
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(
        title="Vision Transformer Projects | Praruj Thapa",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1100px; margin: auto; }"
    ) as demo:
        gr.Markdown("""
# 🔭 Vision Transformer Projects
### by [Praruj Thapa](https://github.com/PRaruj) · AI/ML Engineer · Osaka, Japan

Two computer vision projects using transformer architectures:
- **Tab 1**: DETR fine-tuned for hardhat object detection
- **Tab 2**: ViT + Knowledge Distillation on Oxford Pets (base vs KD comparison)

> 📂 [GitHub Repo](https://github.com/PRaruj/vision-transformer-projects) &nbsp;|&nbsp;
> 📄 [Portfolio](https://www.prarujthapa.com.np/articles/)
        """)

        build_detr_tab()
        build_kd_tab()

        gr.Markdown("""
---
*Built with PyTorch · HuggingFace Transformers · Gradio*
        """)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set share=True to get a public ngrok URL temporarily
    )
