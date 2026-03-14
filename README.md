# ML4SCI-DeepLense-Gravitational Lens Finding-Krrish-Kumar

## Tasks & Results

**Evaluation Metric:** **ROC-AUC** and **Recall**. These metrics were selected to prioritize the detection of rare lensing substructures, ensuring the architectures effectively differentiate between subtle physical signals (Vortices/Subhalos) and standard lensing profiles.

---

### Task 1: Data Preprocessing & Pipeline

**Objective:** Transform raw physics-based `.npy` files and image data into a structured format sensitive to subtle lensing distortions.
**What I Did:**
* **Dataset Structuring:** Processed 3-class data (No Substructure, Subhalo, Vortex) into a unified pipeline handling both NumPy arrays and standard image formats.
* **Physics-Based Normalization:** Overrode standard min-max scaling with **Z-score Standardization** ($Z = \frac{x - \mu}{\sigma}$). This was critical for amplifying the signal of faint substructures that are often lost in raw intensity ranges.
* **Outcome:** Established a robust pipeline that boosted early-stage model convergence and prevented "signal washout" in the physics data.

---

### Task 2: Common Task - Multi-Class Substructure Classification

**Objective:** Classify strong lensing images into three distinct categories (No Substructure, Subhalo, Vortex) to identify the nature of the lens.
**What I Did:**
* **Architecture:** Developed **LensingNet**, utilizing a ResNet-18 backbone integrated with a custom **Spatial-Channel Attention** mechanism to focus on the lensing ring.
* **Optimization:** Used **AdamW** with a **Learning Rate Warmup** phase ($1\times10^{-5}$ to $5\times10^{-5}$) and **Gradient Clipping** at 0.5 to stabilize training.
* **Common Task Results:**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **No Substructure** | 0.54 | 0.99 | 0.70 | 1250 |
| **Subhalo** | 0.73 | 0.29 | 0.41 | 1250 |
| **Vortex** | 0.74 | 0.57 | 0.64 | 1250 |
| **Macro AUC** | **0.8325** | **Accuracy** | **0.62** | **3750** |

**Key Takeaways:** The attention mechanism in **LensingNet** proved vital for capturing the complex distortions of the Vortex class. While "No Substructure" was easily identified, the "Subhalo" class remains the most challenging due to the microscopic scale of mass perturbations.

---

### Task 3: Specific Task - Rare Signal Detection & Benchmarking

**Objective:** Compare advanced architectures for identifying minority "Lens" signals in highly imbalanced datasets.
**What I Did:**
* **Architectural Implementation:** Deployed **Vision Transformer (ViT)** and **ResAttentionNet** to evaluate global vs. local attention.
* **Sensitivity Enhancement:** Integrated **Test-Time Augmentation (TTA)** to maintain high sensitivity under extreme class imbalance.
* **Specific Task Results:**

| Model Approach | Test AUC | Recall | Key Strength |
| :--- | :--- | :--- | :--- |
| **Vision Transformer (ViT + TTA)** | **0.9761** | **0.8615** | Highest balance; superior global attention. |
| **ResAttentionNet + TTA** | 0.9720 | 0.8600 | Effective localized feature extraction. |
| **Standard CNN** | 0.9698 | 0.3744 | High accuracy, but failed on Recall. |

**Key Takeaways:** The **ViT + TTA** configuration emerged as the most balanced architecture. While the standard CNN achieved high accuracy (0.9928), its lack of an attention mechanism resulted in poor sensitivity for rare signal detection.

---

### Model Performance Summary

| Model | Task | Best AUC | Primary Metric |
| :--- | :--- | :--- | :--- |
| **LensingNet (ResNet+Attn)** | **Common** | **0.8325** | Macro AUC (3-Class) |
| **Vision Transformer (ViT)** | **Specific** | **0.9761** | Test AUC (Binary) |
| **ResAttentionNet** | **Specific** | 0.9720 | Test AUC (Binary) |
| **Standard CNN** | **Baseline** | 0.9698 | Test AUC (Binary) |

**Final Conclusions:** For the **Common Task**, spatial attention was necessary to distinguish specific substructure types. For the **Specific Task**, the combination of **Global Attention (ViT)** and **TTA** was the only strategy that effectively identified the minority class without sacrificing precision.
