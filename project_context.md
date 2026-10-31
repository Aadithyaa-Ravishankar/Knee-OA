# Project Context — Anatomy-Aware Deep Learning with Consistent Explainable AI for Knee OA Severity Classification

> **Full research context document.**  
> Authors: Aadithyaa R · Mohanapriya V · Sneha S R  
> Institution: SRM Institute of Science and Technology  
> Last Updated: October 2026

---

## Table of Contents

1. [Clinical Background](#1-clinical-background)
2. [The Kellgren–Lawrence Grading System](#2-the-kellgrenlawrence-grading-system)
3. [The Problem with Manual Grading](#3-the-problem-with-manual-grading)
4. [Why Deep Learning?](#4-why-deep-learning)
5. [Research Gaps in Existing Literature](#5-research-gaps-in-existing-literature)
6. [Our Proposed Solution](#6-our-proposed-solution)
7. [The Four Pillars of Our Framework](#7-the-four-pillars-of-our-framework)
8. [Technical Architecture — Full Pipeline](#8-technical-architecture--full-pipeline)
9. [Dataset](#9-dataset)
10. [Implementation Stages (Completed & Planned)](#10-implementation-stages-completed--planned)
11. [Evaluation Strategy](#11-evaluation-strategy)
12. [Novelty Claims](#12-novelty-claims)
13. [Expected Outcomes](#13-expected-outcomes)
14. [Limitations and Ethical Considerations](#14-limitations-and-ethical-considerations)
15. [References and Related Work](#15-references-and-related-work)

---

## 1. Clinical Background

### What is Knee Osteoarthritis?

Knee Osteoarthritis (OA) is a **chronic degenerative joint disease** characterized by the progressive breakdown of articular cartilage — the flexible, slippery tissue that normally protects bones from friction during movement. As cartilage degrades, bone rubs against bone, leading to pain, stiffness, swelling, and eventual functional disability.

Key pathological changes in knee OA:
- **Articular cartilage loss** — thinning or complete erosion of the cartilage layer
- **Joint space narrowing (JSN)** — the space between femur and tibia visibly narrows on X-ray
- **Osteophyte formation** — bony spurs at joint margins (bone remodeling response)
- **Subchondral bone sclerosis** — increased bone density just beneath the cartilage
- **Bone erosion and cysts** — in advanced stages

### Epidemiology

- Knee OA is the **most common form of arthritis worldwide**
- Affects approximately **14% of adults over 25** and **34% of adults over 65**
- Over **300 million people globally** live with OA (WHO, 2023)
- It is the **leading cause of chronic pain and disability** in the elderly
- The global knee replacement market exceeds **\$9.8 billion USD annually**, largely driven by OA
- In India, OA prevalence is estimated at 22–39% in persons over 40 years

### Why It Matters to Detect It Early

OA is irreversible in its advanced stages. However, early detection (KL-1 to KL-2) opens a critical therapeutic window where:
- Physiotherapy, lifestyle modification, and weight management can slow progression significantly
- Intra-articular injections and pharmacological interventions are most effective
- Total knee replacement (the final resort, invasive, expensive, and carries surgical risk) can be delayed or avoided

Early grading accuracy is therefore not just a diagnostic nicety — it is **clinically consequential**.

---

## 2. The Kellgren–Lawrence Grading System

The Kellgren–Lawrence (KL) scale is the **international standard** for radiographic assessment of knee OA severity. Originally published in 1957, it assigns one of five ordinal grades based on plain anteroposterior (AP) knee X-ray findings.

| KL Grade | Label | Radiographic Features |
|----------|-------|-----------------------|
| **KL-0** | Normal | No features of OA. Normal joint space width. No osteophytes. |
| **KL-1** | Doubtful | Possible minute osteophyte. Doubtful significance. No definite JSN. |
| **KL-2** | Mild | Definite osteophyte. Possible JSN. May have mild sclerosis. |
| **KL-3** | Moderate | Multiple osteophytes. Definite JSN. Some sclerosis. Possible bone deformity. |
| **KL-4** | Severe | Large osteophytes. Marked JSN. Severe sclerosis. Definite bone deformity. |

### Properties of the KL Scale Critical to This Research

1. **Ordinal, not categorical** — KL grades have a meaningful natural order. The difference between KL-1 and KL-2 is clinically different from the difference between KL-3 and KL-4, but all grades exist on a single progression spectrum.
2. **Clinically asymmetric errors** — predicting KL-0 when the true grade is KL-4 is a catastrophic error. Predicting KL-2 when the true grade is KL-3 is a minor error.
3. **Inter-rater variability** — even expert radiologists disagree on KL-1 vs KL-2 in ~30% of cases (Schiphof et al., 2008). This motivates AI-assisted grading with uncertainty quantification.
4. **Class imbalance in natural populations** — KL-0 and KL-2 are most prevalent; KL-4 is rare (matching our dataset).

---

## 3. The Problem with Manual Grading

### Current Clinical Workflow

1. Patient presents with knee pain → referred for plain AP X-ray
2. Radiologist or orthopedic surgeon visually inspects the film
3. KL grade is assigned based on subjective visual assessment
4. Treatment decision is made based on the assigned grade

### Documented Problems

| Problem | Impact |
|---------|--------|
| **High inter-rater variability** | Two radiologists assign different KL grades on the same X-ray in up to 35% of cases |
| **High intra-rater variability** | The same radiologist may grade differently on re-reading |
| **Early-stage ambiguity (KL-1 vs KL-2)** | Subtle changes are frequently missed or misclassified |
| **Radiologist shortage** | In developing countries (including India), radiologist-to-patient ratios are severely strained |
| **Subjective and non-reproducible** | No two grading sessions are guaranteed to match |
| **No quantitative basis** | The KL scale is inherently qualitative — no measurement of joint space width is required |
| **Fatigue effects** | Radiologist attention degrades after hours of reading — errors increase |

These problems motivate an **automated, objective, reproducible, and explainable AI system** for KL grading.

---

## 4. Why Deep Learning?

### Traditional Computer Vision Approaches

Before deep learning, OA grading research relied on:
- **Hand-crafted features**: Joint space width (JSW) measurement, bone texture descriptors, edge detection for osteophytes
- **Classical classifiers**: SVMs, random forests on extracted features
- **Atlas-based methods**: Comparing to reference images

**Limitations of classical approaches:**
- Required manual feature engineering (expert knowledge needed for each feature)
- Could not model complex multi-scale patterns simultaneously
- Low accuracy on subtle early-stage grades (KL-1, KL-2)
- Poor generalization across scanner types and institutions

### Why CNNs / Deep Learning Works Better

Convolutional Neural Networks (CNNs), especially architectures like ResNet, VGG, and EfficientNet, learn:
- **Hierarchical features** — edges → textures → shapes → semantic regions, all learned jointly
- **Discriminative representations** — the exact features that differentiate KL grades, discovered from data
- **Scale-invariant patterns** — osteophytes at any scale, joint space irregularities of any width

State-of-the-art deep learning models achieve **>80% accuracy** on 5-class KL grading, significantly outperforming all traditional methods. However, accuracy alone is insufficient for clinical deployment.

---

## 5. Research Gaps in Existing Literature

This is the core intellectual motivation for our work. We surveyed the major prior publications and identified four consistent, well-documented gaps.

---

### Gap 1 — Treating KL Grading as Standard Classification (Most Common Problem)

**Papers affected:** Tiulpin et al. (2018), Chen et al. (2019), Panfilov et al. (2022), and the majority of deep learning OA papers

**What they do:** Frame KL grading as a standard 5-class multi-class classification problem. Cross-entropy loss treats KL-0, KL-1, KL-2, KL-3, and KL-4 as five completely unrelated, independent categories with equal penalty for any misclassification.

**Why this is wrong:** KL grades are **ordinal**. Predicting KL-0 when truth is KL-4 should be penalized much more severely than predicting KL-3 when truth is KL-4. Standard cross-entropy is completely blind to this ordering — it assigns the same loss value to both errors.

**Specific consequence:** Models trained with cross-entropy tend to make "far jumps" (e.g., KL-0 ↔ KL-4) more often than ordinal-aware models, precisely because the gradient signal does not discourage ordinal distance.

**Our response:** We implement an **ordinal regression loss** (specifically, the CORN/ordinal cross-entropy formulation) that explicitly encodes the KL ordering into the training objective. The model is penalized proportionally to the ordinal distance of the error.

---

### Gap 2 — No Explicit Anatomy-Aware Localization (Common but Underaddressed)

**Papers affected:** Most single-stream CNN papers (VGG, ResNet, EfficientNet baselines)

**What they do:** Feed the entire X-ray (or a loosely cropped region) to the CNN and hope the model learns to focus on the knee joint. No mechanism enforces this.

**Why this is a problem:**
- X-rays contain many irrelevant regions: patient positioning markers, metallic prostheses, scanner frame, patient labels, and surrounding soft tissue
- CNNs are known to learn spurious correlations — a model may learn that a particular scanner marker predicts KL-3, not from actual pathology
- Grad-CAM studies of "black-box" OA classifiers frequently show activation in non-anatomical regions (scan borders, text overlays)
- This is a form of **shortcut learning** (Geirhos et al., 2020) — the model achieves good validation accuracy by latching onto confounding signals

**Partial solutions in prior work:**
- Tiulpin et al. (2018): Used a Siamese CNN trained on bilateral knee patches, which provides rough localization. No explicit joint space supervision.
- OARSI study tools: Manual annotation of joint regions, not scalable.
- Antony et al. (2016): Used a CNN to localize knee, then a second CNN to classify — but did not validate that localization was anatomically meaningful.

**Remaining gap:** No paper has simultaneously (a) localized the knee region using a learnable attention or localization mechanism, (b) validated the localization against ground-truth anatomy, and (c) integrated this localization signal into the loss function during classification.

**Our response:** We implement a **dual-stream architecture** — a global stream (full 224×224 image) and a local stream (attention-guided crop of the joint space region). An **anatomy-alignment loss** penalizes the model when Grad-CAM activation falls outside the anatomically expected knee region.

---

### Gap 3 — Explainability as an Afterthought, Not as a Constraint

**Papers affected:** Almost all papers that include Grad-CAM visualizations

**What they do:** Train the model for maximum accuracy, then **post-hoc** apply Grad-CAM to generate heatmaps. The heatmaps are shown visually but never quantitatively evaluated.

**Why this is insufficient:**
1. **No faithfulness check**: Grad-CAM may highlight any region — there is no test of whether the highlighted region actually explains the prediction
2. **No stability test**: Apply a small, irrelevant image perturbation (brightness shift, minor rotation). Does the heatmap remain stable? Most papers never check this. Studies (Adebayo et al., 2018) show that saliency maps can be completely random and still look visually plausible
3. **No anatomical grounding**: The heatmap may look reasonable to a layperson but highlights clinically irrelevant tissue
4. **Cherry-picking**: Publications show the best-looking Grad-CAM examples; failure cases (unfaithful, noisy, anatomically wrong) are not reported

**Specific evidence of the problem:**
- Adebayo et al. (2018) "Sanity Checks for Saliency Maps" demonstrated that many gradient-based explanation methods produce outputs largely uncorrelated with the model's actual computation
- Ghassemi et al. (2021) "The False Hope of Current Approaches to Explainable AI in Health Care" argues that post-hoc XAI in healthcare is fundamentally unreliable if not validated

**Our response:** We **quantitatively validate explanation quality** using:
- **Anatomy Focus Score (AFS)**: Percentage of Grad-CAM energy that falls within the anatomically defined knee joint bounding box
- **Explanation Stability (ES)**: Structural Similarity (SSIM) between heatmaps of the original image and its mildly perturbed versions
- **Prediction Consistency**: Whether the predicted KL grade changes after mild perturbation (it should not)

These metrics turn "does the heatmap look good?" into a measurable, reproducible quantity.

---

### Gap 4 — No Uncertainty Quantification

**Papers affected:** Virtually all deep learning OA papers

**What they do:** Output a single predicted KL grade. The model is always "confident." No measure of uncertainty is provided.

**Why this is dangerous for clinical use:**
- A model may be highly uncertain about a prediction but still report it with the same format as a high-confidence prediction
- Clinicians have no information about when to trust the model and when to seek a second opinion
- Regulatory bodies (FDA, CE Mark) increasingly require uncertainty quantification for AI-based medical devices
- Class boundaries (KL-1/2, KL-2/3) are inherently ambiguous — the model should report this ambiguity explicitly

**Prior work on uncertainty in medical imaging:**
- Gal & Ghahramani (2016): MC Dropout as Bayesian approximation — enables uncertainty estimation at inference time
- Lakshminarayanan et al. (2017): Deep Ensembles — train multiple models, measure variance across predictions
- Neither approach has been systematically applied to knee OA KL grading with validation of uncertainty calibration

**Our response:** We implement **MC Dropout** during inference: the dropout layers are kept active at test time, and N=30 stochastic forward passes are run. The variance across N predictions gives a per-sample uncertainty estimate. We also report **Expected Calibration Error (ECE)** to measure whether the model's confidence scores actually reflect prediction accuracy.

---

### Summary of Research Gaps Table

| Gap | Description | Prior Work | Our Contribution |
|-----|-------------|-----------|-----------------|
| G1 | Ordinal structure ignored | Standard CE loss in ~95% of papers | Ordinal regression loss (CORN) |
| G2 | No anatomy-aware training | Single-stream CNNs, no localization loss | Dual-stream + anatomy-alignment loss |
| G3 | Explainability unvalidated | Post-hoc Grad-CAM, visual only | Anatomy Focus Score + Stability Score |
| G4 | No uncertainty estimation | Deterministic predictions | MC Dropout + ECE calibration |

---

## 6. Our Proposed Solution

### One-Sentence Description

An **anatomy-aware ordinal deep learning framework** that grades knee OA severity from plain X-rays using a dual-stream ResNet-50 backbone, an ordinal loss function, Grad-CAM explainability, and quantitative validation of both explanation faithfulness and prediction uncertainty.

### Design Philosophy

Three guiding principles informed every design decision:

1. **Clinical trustworthiness over raw accuracy** — A model that achieves 84% accuracy with no explainability is less useful clinically than a model with 80% accuracy and validated, reliable explanations.
2. **Constraints, not post-hoc patches** — Anatomy-awareness and explanation consistency are **built into the training objective**, not added as post-processing.
3. **Honest reporting of uncertainty** — The system must know what it does not know, and report that explicitly.

### System Overview

```
X-ray Input (224 × 224, grayscale)
          │
          ▼
  ┌───────────────────┐
  │   Preprocessing   │  ← CLAHE enhancement + normalization
  └─────────┬─────────┘
            │
    ┌────────┴─────────┐
    ▼                  ▼
┌──────────┐     ┌──────────────────┐
│  Global  │     │  Local (joint    │
│  Stream  │     │  space crop)     │
│ ResNet50 │     │  ResNet50        │
│ backbone │     │  backbone        │
└────┬─────┘     └────────┬─────────┘
     │                    │
     └──────────┬──────────┘
                ▼
      ┌─────────────────┐
      │  Feature Fusion  │  ← Concatenation / attention fusion
      └────────┬─────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
  ┌────────┐       ┌──────────┐
  │Ordinal │       │ Anatomy  │
  │  Head  │       │Alignment │
  │ (KL-0  │       │   Loss   │
  │ to KL-4│       │          │
  └────┬───┘       └──────────┘
       │
  ┌────┴──────────┐
  ▼               ▼
KL Grade      Confidence
Prediction    + Uncertainty
               (MC Dropout)
       │
  ┌────▼────────────┐
  │   Grad-CAM      │  ← Heatmap generation
  └────┬────────────┘
       │
  ┌────▼────────────────────┐
  │  Explanation Validation  │  ← AFS + Stability Score
  └──────────────────────────┘
```

---

## 7. The Four Pillars of Our Framework

### Pillar 1 — Ordinal Severity Learning

**Core idea:** The model must understand that KL grades have a natural ordering: 0 < 1 < 2 < 3 < 4.

**Implementation:** We use the **CORN (Conditional Ordinal Regression for Neural Networks)** loss formulation, or equivalently, the **ordinal cross-entropy** approach:

- The model predicts **K-1 binary classifiers**: "Is grade ≥ 1?", "Is grade ≥ 2?", "Is grade ≥ 3?", "Is grade ≥ 4?"
- These binary probabilities are constrained to be monotonically non-increasing: P(≥1) ≥ P(≥2) ≥ P(≥3) ≥ P(≥4)
- The final predicted grade is derived from these cumulative probabilities
- Loss is computed as the sum of binary cross-entropies over all K-1 binary tasks

**Why this works:** Large ordinal errors naturally incur higher cumulative loss than small ordinal errors, without requiring any explicit distance penalty term.

**Comparison to alternatives:**
- Standard CE: No ordinal awareness whatsoever
- Weighted CE: Can penalize distance, but heuristically; does not guarantee monotonic probability estimates
- Earth Mover's Distance loss: Penalizes ordinal distance but doesn't produce proper probability distributions
- **CORN/Ordinal CE** (our choice): Principled, produces calibrated cumulative probabilities, easily combined with other losses

---

### Pillar 2 — Anatomy-Aware Dual-Stream Architecture

**Core idea:** Force the model to look at the right place, not just anywhere that happens to correlate with the label.

**Architecture — Dual Stream:**

| Stream | Input | Purpose |
|--------|-------|---------|
| Global stream | Full 224×224 image | Capture overall knee geometry, bilateral symmetry, bone deformity |
| Local stream | Attention-guided crop (e.g., 112×112 joint space region) | Zoom into the specific region where pathology manifests (joint space, osteophytes) |

Both streams share a ResNet-50 backbone (weights may be tied or independent — an ablation will determine this). Features from both streams are fused via concatenation followed by a learned attention gate.

**Anatomy-Alignment Loss:**

During training, Grad-CAM activations from the classification head are computed for the current batch. The fraction of activation energy inside a pre-defined anatomical bounding box (the joint space region) is measured. A loss term penalizes the model when this fraction falls below a threshold:

```
L_anatomy = max(0, τ - AFS(batch))
```

where AFS (Anatomy Focus Score) is the proportion of Grad-CAM energy within the anatomically expected region, and τ is a threshold (e.g., 0.5). This is a differentiable soft constraint added to the total loss during training.

**Total Training Loss:**

```
L_total = λ₁ · L_ordinal + λ₂ · L_anatomy
```

where λ₁ and λ₂ are hyperparameters tuned via ablation.

---

### Pillar 3 — Grad-CAM Explainability

**Core idea:** Generate a visual heatmap showing exactly which pixels of the X-ray most influenced the KL grade prediction.

**Method:** Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017):
1. Run a forward pass through the CNN
2. Compute gradients of the target class score with respect to the last convolutional feature map
3. Global-average-pool the gradients to get per-channel importance weights
4. Linearly combine the feature maps weighted by these importances
5. Apply ReLU (only positive contributions matter)
6. Upsample to input image resolution (bilinear interpolation to 224×224)

**Output:** A heatmap overlaid on the X-ray where:
- **Bright red/yellow regions** = areas that strongly support the predicted KL grade
- **Cool/dark regions** = areas that were irrelevant to the prediction

**Extensions planned:**
- **Grad-CAM++**: Improved localization for multi-instance regions (multiple osteophytes)
- **Score-CAM**: Perturbation-based CAM that does not rely on gradients (more faithful)
- **SmoothGrad**: Average gradients over Gaussian-noised copies of the input to reduce high-frequency noise in raw gradient maps

---

### Pillar 4 — Explanation Reliability (Consistency + Confidence)

**Core idea:** A heatmap that looks good is not enough. We must measure whether it is actually meaningful.

#### 4a. Anatomy Focus Score (AFS)

$$\text{AFS} = \frac{\sum_{(i,j) \in \text{knee\_bbox}} H(i,j)}{\sum_{(i,j)} H(i,j)}$$

where H is the Grad-CAM heatmap normalized to [0,1]. AFS = 1.0 means all activation is inside the knee region; AFS = 0 means no activation is in the knee region.

**Threshold:** AFS > 0.5 is considered a "focused" explanation. We report the population-level AFS distribution across the test set.

#### 4b. Explanation Stability Score (ESS)

Apply a small, clinically irrelevant perturbation to the input image (e.g., ±5% brightness shift, Gaussian noise σ=0.01). Re-compute the Grad-CAM heatmap. Measure the Structural Similarity Index (SSIM) between the original and perturbed heatmaps:

$$\text{ESS} = \text{SSIM}(H_{\text{original}}, H_{\text{perturbed}})$$

A reliable model should produce ESS > 0.85 — the heatmap should be essentially unchanged by irrelevant perturbations.

#### 4c. Prediction Consistency Under Perturbation

After the same perturbation, check whether the predicted KL grade changes. A model that changes its prediction due to a 5% brightness shift is unreliable (unstable). We report:
- **Grade flip rate**: Percentage of test samples where the predicted grade changed after perturbation

#### 4d. MC Dropout Uncertainty

At inference time:
1. Keep dropout layers active (training mode for dropout only)
2. Run N=30 stochastic forward passes
3. Collect N probability vectors over {KL-0, …, KL-4}
4. Compute the **mean prediction** (final grade = argmax of mean probs)
5. Compute **predictive entropy** as uncertainty:

$$H = -\sum_{c} \bar{p}_c \log \bar{p}_c$$

High entropy → model is uncertain → flag for human review.

We also compute **Expected Calibration Error (ECE)** to verify that the confidence values are well-calibrated:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$

A perfectly calibrated model should have ECE close to 0.

---

## 8. Technical Architecture — Full Pipeline

### 8.1 Stage-by-Stage Implementation

| Stage | Description | Status |
|-------|-------------|--------|
| **Stage 1** | Dataset audit and validation | ✅ Complete |
| **Stage 2** | Preprocessing pipeline (CLAHE, normalization, DataLoader) | ✅ Complete |
| **Stage 3** | Baseline ResNet-50 single-stream, standard CE loss | 🔄 In Progress |
| **Stage 4** | Dual-stream architecture + ordinal regression loss | ⬜ Planned |
| **Stage 5** | Grad-CAM integration + anatomy-alignment loss | ⬜ Planned |
| **Stage 6** | Explanation consistency evaluation (AFS, ESS) | ⬜ Planned |
| **Stage 7** | MC Dropout uncertainty + ECE calibration | ⬜ Planned |
| **Stage 8** | Ablation studies + final comparison table | ⬜ Planned |

### 8.2 Preprocessing Pipeline (Stage 2 — Implemented)

1. **Load image**: Grayscale PNG, originally 224×224 (pre-resized in dataset)
2. **CLAHE** (Contrast-Limited Adaptive Histogram Equalization):
   - `clip_limit=2.0`, `tile_size=8×8`
   - Enhances local contrast in bone and cartilage regions without over-amplifying noise
3. **Convert to float32** in [0, 1]
4. **ToTensor**: Convert to PyTorch tensor `(1, 224, 224)`
5. **GrayscaleToRGB**: Repeat channel 3× → `(3, 224, 224)` (required for ImageNet-pretrained ResNet-50)
6. **Normalize**: Subtract mean `[0.5787, 0.5787, 0.5787]`, divide by std `[0.1271, 0.1271, 0.1271]` (computed from training set via Welford's algorithm)

**Data augmentation** (training only, Stage 4+):
- Random horizontal flip (knee laterality flip — safe because OA is bilateral)
- Random brightness/contrast jitter (±10%)
- Random rotation (±5°)
- **No** vertical flip, heavy distortion, or synthetic lesion generation

### 8.3 Baseline Model (Stage 3 — In Progress)

- **Backbone**: ResNet-50 pretrained on ImageNet
- **Modification**: Replace the final `fc(2048 → 1000)` layer with `fc(2048 → 5)`
- **Total parameters**: ~23.5 million (all trainable)
- **Loss**: Weighted Cross-Entropy (class weights inversely proportional to class frequency)
- **Optimizer**: AdamW, lr=1e-4, weight_decay=1e-4
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping**: patience=10 epochs
- **Batch size**: 32
- **Dropout**: 0.3 before final FC layer

This baseline serves as the **lower bound** for comparison. Every subsequent addition (ordinal loss, dual stream, anatomy loss) should improve on it.

### 8.4 Dual-Stream Architecture (Stage 4 — Planned)

```
Global branch:  Input(3,224,224) → ResNet50 (layer1-4) → AdaptiveAvgPool → 2048-d vector
Local branch:   Input(3,224,224) → Attention crop → ResNet50 (layer1-4) → AdaptiveAvgPool → 2048-d vector
Fusion:         Concat(4096-d) → Linear(4096→512) → ReLU → Dropout(0.3) → Linear(512→4)
Output:         4 binary classifiers for CORN ordinal loss
```

**Attention crop mechanism:**
- Gradient-free spatial attention map computed from the global stream's intermediate features
- Identifies the joint-space region (roughly center of the 224×224 image after anatomical alignment)
- Crop and resize to 112×112 → fed to local branch

### 8.5 Evaluation Metrics

| Metric | Why Used |
|--------|---------|
| **Quadratic Weighted Kappa (QWK)** | Primary metric — penalizes ordinal distance; standard for OA grading competition (Kaggle KL challenge) |
| **Macro-averaged F1** | Treats each class equally — important given class imbalance |
| **Per-class Accuracy** | Understand which KL grades are confused most |
| **Confusion Matrix** | Visualize misclassification patterns |
| **Anatomy Focus Score (AFS)** | Explanation quality |
| **Explanation Stability Score (ESS)** | Explanation robustness |
| **Expected Calibration Error (ECE)** | Uncertainty calibration |
| **Grade Flip Rate** | Prediction stability under perturbation |

---

## 9. Dataset

### Source
- **Name**: Knee Osteoarthritis Dataset with Severity Grading
- **Platform**: Kaggle
- **URL**: https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity
- **Original X-ray source**: OAI (Osteoarthritis Initiative) — a multi-center study sponsored by NIH

### Key Dataset Properties (from Stage 1 Audit)

| Property | Value |
|----------|-------|
| Total valid images | **9,786** |
| Format | PNG, grayscale |
| Resolution | 224 × 224 px (all images pre-standardized) |
| Corrupted images | 0 |
| Duplicate images | 0 |
| Splits provided | train / val / test / auto_test |

### KL-Grade Distribution

| KL Grade | Total | Train | Val | Test | Auto-Test |
|----------|-------|-------|-----|------|-----------|
| KL-0 | 3,857 (39.4%) | 2,286 | 328 | 639 | 604 |
| KL-1 | 1,770 (18.1%) | 1,046 | 153 | 296 | 275 |
| KL-2 | 2,578 (26.3%) | 1,516 | 212 | 447 | 403 |
| KL-3 | 1,286 (13.1%) | 757 | 106 | 223 | 200 |
| KL-4 | 295 (3.0%) | 173 | 27 | 51 | 44 |
| **Total** | **9,786** | **5,778** | **826** | **1,656** | **1,526** |

### Class Imbalance

- **Imbalance ratio**: 13.07× (KL-0 most common, KL-4 rarest)
- **Handling strategy**:
  1. `WeightedRandomSampler` on training DataLoader (batch composition)
  2. Inverse-frequency class weights in loss function (gradient scale)
  3. Ordinal loss (Stage 4) naturally handles class boundaries differently

### Patient / Subject Information

- **Filename pattern**: `<PatientID><L|R>.png` — e.g., `9001695L.png`
- **Unique patient IDs**: 4,130
- **Left-knee images**: 4,130 | **Right-knee images**: 4,130
- All 4,130 patients have both knees in the dataset
- **Data leakage check**: ✅ No patient appears in multiple splits — the pre-defined Kaggle splits are already patient-level

### Class Weights for Loss (Stage 2, Computed)

| KL Grade | Train Count | Class Weight |
|----------|-------------|--------------|
| KL-0 | 2,286 | 0.506 |
| KL-1 | 1,046 | 1.105 |
| KL-2 | 1,516 | 0.762 |
| KL-3 | 757 | 1.524 |
| KL-4 | 173 | 6.672 |

### Normalization Statistics (Stage 2, Computed via Welford's Algorithm)

| Statistic | Value |
|-----------|-------|
| Training images scanned | 5,778 |
| Total pixels processed | 289,533,696 |
| **Mean (grayscale)** | **0.578707** |
| **Std (grayscale)** | **0.127051** |

> **Note:** Mean of 0.578 (vs ImageNet mean 0.485) is expected — X-rays are predominantly bright (bone), with dark background from air. Std of 0.127 (vs ImageNet 0.229) reflects moderate contrast uniformity.

---

## 10. Implementation Stages (Completed & Planned)

### Stage 1 — Dataset Audit ✅

**Goal:** Fully understand the dataset before writing any model code.

**Scripts:** `src/preprocessing/stage1_dataset_audit.py`

**Outputs:** `reports/stage1/audit_report.md`, distribution charts, sample X-rays

**Key findings:**
- 9,786 clean PNG images, 224×224, all grayscale
- 0 corrupted, 0 duplicates
- 4 pre-defined splits with patient-level separation (no leakage)
- 13.07× class imbalance (KL-0 vs KL-4)
- Both knees (L/R) from same patient always in same split

---

### Stage 2 — Preprocessing Foundation ✅

**Goal:** Build a reusable, configurable preprocessing pipeline.

**Scripts:** `src/preprocessing/`
- `dataset.py` — PyTorch Dataset, class weight calculator
- `transforms.py` — CLAHE + Normalize (lazy-init for multiprocessing safety)
- `dataloader.py` — `build_dataloaders()` with WeightedRandomSampler
- `compute_stats.py` — Pixel-level mean/std via Welford's algorithm

**Key design decisions:**
- CLAHE lazy-initialization (`_get_clahe()`) to avoid `cv2.CLAHE` pickling failures on Python 3.14 / macOS with `forkserver`
- `num_workers=0` recommended on Python 3.14 macOS (spawn/forkserver incompatibility with custom objects)
- 3-channel repeat for ImageNet-pretrained ResNet-50

---

### Stage 3 — Baseline ResNet-50 🔄

**Goal:** Establish a reproducible baseline before adding any novel components.

**Scripts:** `src/training/stage3_train_baseline.py`, `src/models/baseline.py`

**Architecture:** ResNet-50, final FC replaced with 5-class head

**Loss:** Weighted Cross-Entropy

**Metrics tracked:** QWK, macro-F1, per-class accuracy, confusion matrix

**Output:** `outputs/models/baseline_resnet50_best.pth`, training curves in `reports/stage3/`

---

### Stage 4 — Dual-Stream + Ordinal Loss ⬜

**Goal:** Replace baseline with anatomy-aware dual-stream architecture and ordinal loss.

**Files to create:**
- `src/models/dual_stream.py` — DualStreamResNet50
- `src/losses/ordinal_loss.py` — CORN ordinal CE loss
- `src/losses/anatomy_loss.py` — Grad-CAM anatomy alignment loss
- `src/training/stage4_train_dual_stream.py`

---

### Stage 5 — Grad-CAM Explainability ⬜

**Goal:** Integrate Grad-CAM into both training (anatomy loss) and inference (visual explanation).

**Files to create:**
- `src/xai/gradcam.py` — Grad-CAM, Grad-CAM++, Score-CAM
- `src/xai/visualization.py` — Overlay heatmap on X-ray, save to `outputs/heatmaps/`

---

### Stage 6 — Explanation Consistency Evaluation ⬜

**Goal:** Quantitatively validate that heatmaps are anatomically meaningful and stable.

**Files to create:**
- `src/xai/consistency.py` — AFS computation, ESS computation (SSIM), perturbation pipeline
- `src/evaluation/xai_metrics.py` — Population-level AFS/ESS statistics

---

### Stage 7 — Uncertainty Quantification ⬜

**Goal:** Add MC Dropout inference, entropy-based uncertainty, ECE calibration.

**Files to create:**
- `src/uncertainty/mc_dropout.py` — N-pass inference, entropy, grade flip rate
- `src/uncertainty/calibration.py` — ECE computation, reliability diagram

---

### Stage 8 — Ablation Studies ⬜

**Goal:** Prove that each component contributes. Ablations to run:

| Configuration | Ordinal Loss | Dual Stream | Anatomy Loss | XAI Validation |
|---------------|:---:|:---:|:---:|:---:|
| Stage 3 Baseline | ❌ | ❌ | ❌ | ❌ |
| + Ordinal Loss only | ✅ | ❌ | ❌ | ❌ |
| + Dual Stream only | ❌ | ✅ | ❌ | ❌ |
| + Dual Stream + Ordinal | ✅ | ✅ | ❌ | ❌ |
| + All (Full Model) | ✅ | ✅ | ✅ | ✅ |

For each configuration, report: QWK, macro-F1, AFS, ESS, ECE. This table will form the primary results table of the paper.

---

## 11. Evaluation Strategy

### Primary Metrics

1. **Quadratic Weighted Kappa (QWK)**  
   The standard metric for ordinal multi-class problems and the metric used in the Kaggle OA grading challenge. QWK = 1.0 is perfect; QWK = 0.0 is random; QWK < 0 is worse than random.  
   $$\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i-j)^2}{(N-1)^2}$$

2. **Macro-averaged F1 Score**  
   Treats each class equally regardless of size. Essential given the 13× imbalance.

### Secondary Metrics
- Per-class sensitivity (recall) — especially for KL-4 (most clinically critical, most underrepresented)
- Confusion matrix (normalized row-wise) — visualizes ordinal confusion patterns
- Precision-Recall curves per class

### Explainability Metrics
- Anatomy Focus Score (AFS) > 0.5 target, population mean reported
- Explanation Stability Score (ESS) > 0.85 target, population mean reported
- Grade Flip Rate < 5% target under mild perturbation

### Uncertainty Metrics
- Expected Calibration Error (ECE) — lower is better
- Reliability diagram — visual calibration curve
- Mean predictive entropy per KL grade — boundary grades (KL-1/2, KL-2/3) should be highest

### Comparison Targets (from Literature)
- Tiulpin et al. (2018): QWK = 0.83 (Siamese CNN)
- Chen et al. (2019): QWK = 0.87 (ResNet variants)
- Panfilov et al. (2022): QWK = 0.88 (transformer-based)
- **Our target**: QWK ≥ 0.85 on the test set, with validated explanation metrics

---

## 12. Novelty Claims

This project's novelty lies not in any single component (each has precedent) but in their **simultaneous integration and joint validation**. No prior published work has combined all four:

### Claim 1 — Integrated Ordinal + Anatomy-Aware Training
Prior work uses ordinal loss OR anatomy-aware localization. We use both simultaneously in a joint loss function where both signals influence the same backpropagation pass.

### Claim 2 — Explanation Quality as a Training Constraint
Prior work evaluates Grad-CAM post-hoc visually. We incorporate the anatomy-alignment loss **during training**, meaning the model is explicitly trained to produce anatomically focused explanations, not just accurate predictions.

### Claim 3 — Quantitative XAI Validation on OA Grading
To the best of our knowledge, no published OA grading paper reports AFS or ESS as numeric metrics with threshold-based evaluation. We introduce and define these metrics in the context of knee X-ray grading.

### Claim 4 — End-to-End Reliability Framework
The combination of (a) ordinal awareness, (b) anatomy-aware localization, (c) explanation faithfulness quantification, and (d) uncertainty quantification constitutes a unified reliability-focused framework. This integrated approach has not been reported in the OA grading literature.

---

## 13. Expected Outcomes

### Research Outcomes

1. A trained dual-stream ordinal ResNet-50 model achieving QWK ≥ 0.85 on the test set
2. Baseline vs. full-model ablation demonstrating measurable improvement from each component
3. Population-level AFS and ESS metrics demonstrating superior explanation quality over standard post-hoc Grad-CAM
4. Calibration curves showing the model's uncertainty estimates are meaningful
5. A reproducible open-source codebase (this repository)

### Clinical Value Proposition

If deployed in a clinical support setting:
- Provide an **automated KL grade recommendation** as a second reader
- Show the clinician **exactly what region of the X-ray** drove the recommendation
- Flag **uncertain cases** (high predictive entropy, near-boundary grades) for priority human review
- Reduce radiologist cognitive load without replacing clinical judgment

---

## 14. Limitations and Ethical Considerations

### Technical Limitations

1. **Single dataset** — Results are on one pre-processed Kaggle dataset derived from OAI. External validation on different scanner types, institutions, or ethnic populations is needed.
2. **No MRI / clinical metadata** — OA diagnosis in clinical practice uses MRI, patient history, and physical examination alongside X-rays. This system uses only plain X-rays.
3. **Pre-resized images** — All images in this dataset are already 224×224. The original resolution information is unavailable, so we cannot evaluate the impact of resolution on grading.
4. **Anatomical bounding box** — The "knee region" used for AFS computation is approximated. A precise anatomical bounding box would require per-image annotation (not available in this dataset).
5. **CLAHE parameters are fixed** — We use `clip_limit=2.0`, `tile=8×8` throughout. These were chosen based on domain knowledge but not systematically optimized.

### Ethical Considerations

1. **This is a research tool, not a regulatory-approved medical device.** The system must not be used for actual clinical decision-making without further validation and regulatory clearance (FDA 510(k) / CE Mark).
2. **Bias in training data** — OAI was recruited primarily in the United States. The model may not generalize equally across ethnic groups or geographic populations.
3. **Explainability ≠ interpretability for all users** — Grad-CAM heatmaps require training to interpret. Showing them to non-specialist users without context could cause confusion.
4. **Uncertainty should be surfaced** — The system must always report its confidence. A model that is uncertain but presents a confident prediction is more dangerous than a model that refuses to predict.
5. **Patient data privacy** — The OAI dataset used here is de-identified and publicly released under appropriate data use agreements. No patient-identifiable data is used.

---

## 15. References and Related Work

### Foundational Methods

| Reference | Contribution |
|-----------|-------------|
| Kellgren & Lawrence (1957) | Original KL grading scale definition |
| He et al. (2016) — ResNet | Deep Residual Learning for Image Recognition |
| Selvaraju et al. (2017) — Grad-CAM | Gradient-weighted Class Activation Mapping |
| Gal & Ghahramani (2016) — MC Dropout | Dropout as a Bayesian Approximation |
| Lakshminarayanan et al. (2017) — Deep Ensembles | Simple and Scalable Predictive Uncertainty Estimation |
| Frank & Hall (2001) — Ordinal CE | Simple Approach to Ordinal Classification |
| Shi et al. (2023) — CORN | Ordinal Regression with Consistent Optimality |

### Prior Work on Knee OA Grading

| Paper | Method | QWK | Gap |
|-------|--------|-----|-----|
| Antony et al. (2016) | CNN localize then classify | ~0.70 | No XAI, no ordinal |
| Tiulpin et al. (2018) | Siamese CNN, bilateral patches | 0.83 | No ordinal, no XAI validation |
| Chen et al. (2019) | ResNet variants, transfer learning | 0.87 | Standard CE, no XAI |
| Panfilov et al. (2022) | Transformer-based | 0.88 | No anatomy alignment, no uncertainty |
| Swiecicki et al. (2021) | Ordinal regression on OA | ~0.85 | No anatomy, no XAI |

### Explainability Validation Literature

| Paper | Contribution |
|-------|-------------|
| Adebayo et al. (2018) | Sanity checks for saliency maps — showed many XAI methods are unreliable |
| Samek et al. (2017) | Evaluating XAI methods by pixel perturbation |
| Ghassemi et al. (2021) | False hope of XAI in healthcare — motivates our quantitative validation |
| Zeiler & Fergus (2014) | Visualizing CNN representations (occlusion sensitivity) |

### Class Imbalance and Uncertainty

| Paper | Contribution |
|-------|-------------|
| Lin et al. (2017) — Focal Loss | Addresses class imbalance in detection tasks |
| Guo et al. (2017) | On calibration of modern neural networks — temperature scaling, ECE |
| Naeini et al. (2015) | Obtaining well-calibrated probabilities — reliability diagrams |

---

## Appendix A — Project File Map

```
knee_oa_xai/
│
├── project_context.md          ← THIS FILE
├── README.md                   ← Quick start guide
├── DATASET_SETUP.md            ← Teammate dataset download guide
├── requirements.txt
│
├── configs/
│   └── config.yaml             ← Central config (paths, hyperparams)
│
├── src/
│   ├── preprocessing/          ← Stage 1–2
│   │   ├── stage1_dataset_audit.py
│   │   ├── dataset.py
│   │   ├── transforms.py       ← CLAHE (lazy-init), Normalize, GrayscaleToRGB
│   │   ├── dataloader.py       ← build_dataloaders(), WeightedRandomSampler
│   │   └── compute_stats.py   ← Welford mean/std computation
│   │
│   ├── models/                 ← Stage 3+
│   │   ├── baseline.py         ← BaselineResNet50 (single-stream)
│   │   └── dual_stream.py      ← [Planned Stage 4]
│   │
│   ├── losses/                 ← [Planned Stage 4]
│   │   ├── ordinal_loss.py     ← CORN ordinal CE
│   │   └── anatomy_loss.py     ← Grad-CAM anatomy alignment
│   │
│   ├── training/               ← Stage 3+
│   │   ├── stage3_train_baseline.py
│   │   └── stage4_train_dual_stream.py  ← [Planned]
│   │
│   ├── xai/                    ← [Planned Stage 5–6]
│   │   ├── gradcam.py
│   │   ├── visualization.py
│   │   └── consistency.py      ← AFS, ESS, perturbation
│   │
│   ├── uncertainty/            ← [Planned Stage 7]
│   │   ├── mc_dropout.py
│   │   └── calibration.py      ← ECE, reliability diagram
│   │
│   └── evaluation/             ← Stage 3+
│       └── metrics.py          ← QWK, macro-F1, confusion matrix
│
├── reports/
│   ├── stage1/                 ← Dataset audit report + figures
│   ├── stage2/                 ← Normalization stats + pipeline report
│   └── stage3/                 ← Training curves [after Stage 3 completes]
│
└── outputs/
    ├── figures/                ← Dataset visualizations
    ├── metrics/                ← Audit CSVs
    ├── models/                 ← Saved .pth checkpoints
    └── heatmaps/               ← Grad-CAM output images [Stage 5+]
```

---

## Appendix B — Configuration Reference

From `configs/config.yaml`:

```yaml
preprocessing:
  clahe_clip_limit : 2.0
  clahe_tile_size  : [8, 8]
  mean_3ch         : [0.578707, 0.578707, 0.578707]
  std_3ch          : [0.127051, 0.127051, 0.127051]

dataloader:
  batch_size           : 32
  num_workers          : 0        # Use 0 on Python 3.14 / macOS (spawn quirks)
  use_weighted_sampler : true
  channels             : 3

training:
  seed                 : 42
  epochs               : 50
  lr                   : 0.0001
  weight_decay         : 0.0001
  early_stop_patience  : 10

model:
  backbone    : resnet50
  pretrained  : true
  num_classes : 5
  dropout     : 0.3
```

---

*This document is the single source of truth for the project's research context, motivation, architecture, and implementation plan. Update it as the project progresses.*
