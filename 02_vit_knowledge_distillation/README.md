# 🐾 ViT + Knowledge Distillation — Oxford Pets

> **Part of the [Vision Transformer Projects](../README.md) portfolio**

Two-phase experiment training a Vision Transformer on the Oxford-IIIT Pet Dataset (37 breeds, 7,390 images):
1. ViT from scratch (baseline)
2. ViT with Knowledge Distillation (teacher-student, same student architecture)

## 📓 Notebook

[**Open 03_10_knowledge_distillation_exercise.ipynb →**](03_10_knowledge_distillation_exercise.ipynb)

The notebook covers:
- Building ViT from scratch: patch splitting, linear projection, [CLS] token, positional encodings, MHA blocks
- Knowledge Distillation theory: why soft labels carry more information than hard labels
- Temperature scaling: how higher T reveals inter-class similarity structure
- Combined distillation loss: CE (hard) + KL (soft, teacher-guided)
- Comparison: Base ViT vs KD ViT on same dataset

## 🧠 Architecture: ViT from Scratch

```
224×224 image → split into 196 patches (16×16)
             → Linear projection → [196, 768]
             → + [CLS] token + Positional Embeddings
             → N × Transformer Encoder Blocks (LayerNorm + MHA + FFN)
             → [CLS] token output → MLP Head → 37 classes
```

## 🎓 Knowledge Distillation

```
Teacher (pretrained, large) → soft labels (probability distributions)
                                     ↓
Loss = α × CrossEntropy(student, hard_labels)
     + (1-α) × KLDiv(student_logits/T, teacher_logits/T)
                                     ↓
Student (same ViT, trains smarter)
```

| Parameter | Role |
|-----------|------|
| Temperature T | Higher T → softer distributions → more inter-class info transferred |
| α (alpha) | Balances hard-label CE vs. soft-label KL loss |

## 📦 Model Weights

- Baseline: [`PRaruj/oxford-pets-vit-from-scratch`](https://huggingface.co/PRaruj/oxford-pets-vit-from-scratch)
- KD Model: [`PRaruj/oxford-pets-vit-with-kd`](https://huggingface.co/PRaruj/oxford-pets-vit-with-kd)

## 🔗 References

- [An Image is Worth 16x16 Words: ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
