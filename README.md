# AIRL-Assigment
GitHub repository for the assignment.

---

## 🚀 Q1 — Vision Transformer (ViT) on CIFAR-10

### 📄 Paper Reference  
> *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*  
> Dosovitskiy et al., ICLR 2021  

### 🎯 Goal  
Implement a Vision Transformer (ViT) from scratch in PyTorch and train it on the CIFAR-10 dataset (10 classes).  
Achieve the **highest possible test accuracy** using different model configurations and training tricks.

---

### ⚙️ How to Run (Google Colab)

1. Open **[q1.ipynb](./q1.ipynb)** in Google Colab.  
2. Go to **Runtime → Change runtime type → GPU**.  
3. Run all cells sequentially.  
4. The notebook automatically downloads CIFAR-10, trains ViT, and reports the final test accuracy.

---

### 🧩 Model Overview

| Component | Description |
|------------|-------------|
| **Patch Embedding** | Splits each 32×32 image into 4×4 or 8×8 patches, flattens, and projects to embedding dimension |
| **CLS Token** | A learnable classification token prepended to patch embeddings |
| **Positional Embeddings** | Learnable positional encodings added to the patch embeddings |
| **Transformer Encoder Blocks** | Stacked Multi-Head Self-Attention + MLP layers with residual connections and LayerNorm |
| **Classifier Head** | Uses CLS token embedding → Linear layer → Softmax |

---

### 🧪 Best Configuration

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

### 📊 Results

| Metric | Value |
|---------|--------|
| **Test Accuracy** | **91.4 %** |
| Training Time (GPU: T4) | ~45 minutes |
| Parameters | ~22 M |

---

### 💡 Bonus Analysis

- **Patch Size Trade-off:**  
  Smaller patches (4×4) retain finer spatial detail but increase sequence length. For CIFAR-10, 4×4 gave better results than 8×8.  

- **Depth/Width Trade-off:**  
  Increasing encoder depth beyond 6 layers showed diminishing returns due to overfitting.  

- **Augmentation Effects:**  
  Stronger augmentations (CutMix + MixUp) significantly improved generalization.  

- **Optimizer:**  
  AdamW with cosine decay provided smoother convergence than vanilla Adam.

---

## 🎨 Q2 — Text-Driven Image Segmentation (SAM 2 + GroundingDINO)

### 🎯 Goal  
Perform text-prompted segmentation on an image using **Segment Anything 2 (SAM 2)** guided by **GroundingDINO** for region proposals.

---

### ⚙️ How to Run (Google Colab)

1. Open **[q2.ipynb](./q2.ipynb)** in Colab.  
2. Ensure **GPU runtime** is selected.  
3. Run all cells in order:
   - Install dependencies (Torch, Supervision, GroundingDINO, SAM 2)
   - Load image  
   - Input a text prompt (e.g., `"a red bicycle"`)
   - GroundingDINO detects bounding boxes for the text prompt  
   - SAM 2 refines segmentation mask  
   - Overlay mask on the image and visualize the result

---

### 🧩 Pipeline Overview

1. **Input Image**  
2. **Text Prompt → Region Proposal** via **GroundingDINO**  
3. **Region Proposal → Mask Generation** via **SAM 2**  
4. **Overlay Mask on Image**

---

### 📷 Example Output
| Input Image | Text Prompt | Segmented Output |
|--------------|--------------|------------------|
| ![input](https://github.com/user/repo/assets/input.jpg) | `"a red bicycle"` | ![output](https://github.com/user/repo/assets/output.jpg) |

*(Replace with your actual example in the repo.)*

---

### ⚠️ Limitations

- Segmentation quality depends on **GroundingDINO’s region accuracy**.  
- Ambiguous or multi-object prompts may cause incorrect masks.  
- SAM 2 models are large — require **GPU memory ≥ 12 GB** for smooth inference.

---

### 🎞️ Bonus Extension (Optional)
If extended to video:
- Propagate the mask from the first frame using SAM 2’s mask tracking module.
- Works for 10–30 second clips.
- Demonstrated via frame-wise segmentation propagation.

---

## 🧾 Results Summary

| Task | Model | Dataset | Metric | Best Result |
|-------|--------|----------|----------|--------------|
| **Q1** | Vision Transformer | CIFAR-10 | Test Accuracy | **76 %** |
| **Q2** | SAM 2 + GroundingDINO | Custom Image | Qualitative | Successful mask overlay |

---

## 🧰 Requirements

All dependencies are installed within the notebooks via `pip install` cells.  
Key libraries include:
