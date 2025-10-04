# AIRL-Assigment
GitHub repository for the assignment.

---

##  Q1 — Vision Transformer (ViT) on CIFAR-10


###  Goal  
Implement a Vision Transformer (ViT) from scratch in PyTorch and train it on the CIFAR-10 dataset (10 classes).  


---

### ⚙️ How to Run (Google Colab)

1. Open **[q1.ipynb](./q1.ipynb)** in Google Colab.  
2. Go to **Runtime → Change runtime type → GPU**.  
3. Run all cells sequentially.  
4. The notebook automatically downloads CIFAR-10, trains ViT, and reports the final test accuracy.

---

###  Model Overview

| Component | Description |
|------------|-------------|
| **Patch Embedding** | Splits each 32×32 image into 4×4 or 8×8 patches, flattens, and projects to embedding dimension |
| **CLS Token** | A learnable classification token prepended to patch embeddings |
| **Positional Embeddings** | Learnable positional encodings added to the patch embeddings |
| **Transformer Encoder Blocks** | Stacked Multi-Head Self-Attention + MLP layers with residual connections and LayerNorm |
| **Classifier Head** | Uses CLS token embedding → Linear layer → Softmax |

---

###  Best Configuration

| Hyperparameter | Value |
|-----------------|--------|
| Patch Size | 4×4 |
| Embedding Dim | 256 |
| Depth | 6 Transformer blocks |
| Heads | 8 |
| MLP Ratio | 4 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| LR Scheduler | Cosine Annealing |
| Learning Rate | 3e-4 |
| Batch Size | 128 |
| Epochs | 50 |
| Data Augmentation | RandomCrop, RandomHorizontalFlip, CutMix, MixUp |

---

###  Results

| Metric | Value |
|---------|--------|
| **Test Accuracy** | **76 %** |
| Training Time (GPU: T4) | ~45 minutes |
| Parameters | ~22 M |

---



##  Q2 — Text-Driven Image Segmentation (SAM 2 + GroundingDINO)

###  Goal  
Perform text-prompted segmentation on an image using **Segment Anything 2 (SAM 2)** guided by **GroundingDINO** for region proposals.

---


###  Pipeline Overview

1. **Input Image**  
2. **Text Prompt → Region Proposal** via **GroundingDINO**  
3. **Region Proposal → Mask Generation** via **SAM 2**  
4. **Overlay Mask on Image**


###  Limitations

- Segmentation quality depends on **GroundingDINO’s region accuracy**.   
- SAM 2 models are large — require **GPU memory ≥ 12 GB** for smooth inference.

---



## 🧾 Results Summary

| Task | Model | Dataset | Metric | Best Result |
|-------|--------|----------|----------|--------------|
| **Q1** | Vision Transformer | CIFAR-10 | Test Accuracy | **76 %** |
| **Q2** | SAM 2 + GroundingDINO | Custom Image | Qualitative | Successful mask overlay |

---

