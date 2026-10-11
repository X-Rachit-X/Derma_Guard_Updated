# 04 — Data Pipeline
### *MILK10k Structure, Leak-Safe Splits, Label Grouping, and Preprocessing*

> **Prerequisites:** [`03_Mathematical_Foundations.md`](./03_Mathematical_Foundations.md)  
> **Next:** [`05_Architecture_Deep_Dive.md`](./05_Architecture_Deep_Dive.md)

---

## 1. Primary Dataset: MILK10k

### 1.1 Why MILK10k?

MILK10k was selected over the v1.0 choice of Derm7pt for the following reasons:

| Criterion | Derm7pt | MILK10k |
|-----------|---------|---------|
| Number of lesions | ~1,000 | **5,240** |
| Paired clinical + dermoscopy | ✅ | ✅ |
| Histopathology confirmation | Partial | ~**96%** |
| Skin-tone metadata | ❌ | ✅ (6 levels) |
| MONET concept annotations | ❌ | ✅ |
| "Acquire dermoscopy" is a real experiment | Simulation only | **Real** (both always present) |
| Centre information | 1 centre | **5 centres** |
| Benchmark leaderboard | ❌ | ✅ (MILK10k Benchmark, 479 lesions) |

The fact that every lesion in MILK10k has both a clinical photograph and a dermoscopic image is critical: it makes "acquire dermoscopy" a **controlled experiment** rather than a simulation.

### 1.2 Dataset structure

```
MILK10k
├── 5,240 lesions
│   ├── Each lesion: 1 clinical close-up + 1 dermoscopic image = 10,480 images total
│   ├── 5 collecting centres
│   ├── ~96% histopathology-confirmed
│   └── ISIC IDs (unique per lesion)
│
├── Diagnostic categories (11 broad categories):
│   ├── Melanoma (MEL)
│   ├── Basal cell carcinoma (BCC)
│   ├── Squamous cell carcinoma (SCC)
│   ├── Actinic keratosis (AK)
│   ├── Nevus (NEV)
│   ├── Seborrheic keratosis (SK)
│   ├── Solar lentigo (SL)
│   ├── Dermatofibroma (DF)
│   ├── Vascular lesion (VAS)
│   ├── Other benign (OB)
│   └── Other malignant (OM)  ← very small class (~7 training lesions!)
│
└── Metadata per lesion:
    ├── Age: 5-year bins (e.g., 45-49)
    ├── Sex: male/female/unknown
    ├── Anatomic site: face, back, arm, leg, etc.
    ├── Skin tone: 6 levels (not Fitzpatrick scale — MILK10k-specific)
    └── MONET concept probabilities: ~30 morphological features
```

### 1.3 The MILK10k Benchmark (separate)

A separate test set of 479 lesions with hidden labels exists as a live challenge. Submitting to this challenge gives an external, unbiased performance estimate. Check challenge rules before submitting.

---

## 2. Label Grouping Strategy

### 2.1 The problem with 11 classes

Some classes have very few samples (e.g., "Other malignant" ≈ 7 training lesions). Training a classifier on these classes will result in:
- Very high variance in per-class metrics
- Near-random predictions for rare classes
- Misleading aggregate metrics

### 2.2 Coarse grouping (used in classification)

Merge rare classes:

```
Original → Coarse Group
─────────────────────────────
MEL              → Malignant-specific
BCC, SCC, AK, OM → Other Malignant (merged)
NEV, SK, SL, DF  → Benign-specific
OB, VAS          → Other Benign (merged)
```

Report both coarse-group and binary malignancy metrics.

### 2.3 Binary grouping (used for the LTT guarantee)

```
Malignancy group:
  MALIGNANT: MEL, BCC, SCC, AK, OM
  BENIGN:    NEV, SK, SL, DF, OB, VAS
```

The LTT guarantee is stated in terms of the binary malignancy miss rate:

```
R(λ) = P( g(y)=MALIGNANT  AND  g(ŷ)=BENIGN  AND  a*=PROCEED )
```

This is more statistically stable than per-class guarantees (more samples in each group) and clinically meaningful (the key decision is malignant vs benign).

---

## 3. Data Splits

### 3.1 The four-way split

```
MILK10k lesions (5,240 total)
├── Train       60% = ~3,144 lesions  → fit model weights
├── Validation  15% = ~786 lesions    → fit temperature T, hyperparameters
├── Calibration 10% = ~524 lesions    → fit conformal quantile, LTT thresholds
└── Test        15% = ~786 lesions    → final evaluation (touch only once)

⚠️  The test split is frozen after the pre-registration file is committed.
    No one may inspect test-set performance until the protocol is frozen.
```

### 3.2 Why a separate calibration split?

Conformal prediction and LTT require fresh, held-out data that was not used for model training OR hyperparameter selection. Using the validation split for calibration would introduce subtle bias.

**Key rule:** calibration split is used *only* for:
- Computing RAPS conformal scores and quantile q̂
- Running the LTT fixed-sequence test to select λ*

**Nothing else.** Temperature scaling uses the validation split. Model weights use the train split.

### 3.3 Stratification

Splits are stratified by:
1. Coarse diagnostic category (ensure each split has representation of each class)
2. Skin-tone level (ensure fairness analysis has enough samples per group per split)

### 3.4 The leakage risk

MILK10k does **not** provide patient identifiers. It is possible that multiple lesions from the same patient appear in the dataset (e.g., photographed at different visits). If so, a patient's lesions could end up in both train and test — this is data leakage.

**Mitigation:**
- Split by ISIC lesion ID (each lesion gets exactly one split assignment)
- Deduplicate via ISIC IDs to avoid the same image appearing twice
- State this residual leakage risk explicitly in the paper's limitations section

### 3.5 Three seeds

Repeat splits with seeds 0, 1, 2. Report mean ± 95% confidence interval across seeds to demonstrate robustness of results.

### 3.6 Statistical power note

```
Test split: ~786 lesions
Estimated melanomas in test: ~50-60 (melanoma prevalence ≈ 8-10%)

→ Confidence intervals on per-class melanoma metrics will be WIDE
→ Do NOT make strong claims about rare classes
→ Use bootstrap CIs (≥ 2,000 resamples) for all method comparisons
→ Binary malignancy AUROC is more stable than per-class sensitivity
```

---

## 4. Secondary and Shift Datasets

### 4.1 Derm7pt (secondary)

```
Derm7pt (Kawahara et al., 2019)
├── ~1,000 lesions
├── Clinical + dermoscopic + 7-point checklist
├── 1 centre
└── Use: replication of MILK10k results if time allows
```

⚠️ Verify licence and class mapping before use. Do NOT assume the same diagnostic categories apply.

### 4.2 PAD-UFES-20 (distribution shift dataset)

```
PAD-UFES-20 (Pacheco et al., 2020)
├── Brazilian patient population
├── Smartphone clinical images ONLY (no dermoscopy)
├── 6 diagnostic categories
└── Use: ONLY for Family D (clinical-branch distribution shift tests)
         NOT as a multimodal benchmark
         Map labels explicitly to MILK10k categories — document any gaps
```

This dataset represents a real distribution shift: different device, different population, different lighting, different demographic.

### 4.3 HAM10000 (optional)

```
HAM10000 (Tschandl et al., 2018)
├── 10,015 dermoscopic images
├── No clinical close-up
└── Use: ONLY for dermoscopy-branch pretraining or OOD evaluation
         NOT for multimodal experiments
```

---

## 5. Preprocessing

### 5.1 Image preprocessing

```
Standard pipeline for clinical and dermoscopic images:
1. Load image (RGB)
2. Resize to 224×224 (or 384×384 for ViT-based backbones)
3. Apply ImageNet normalization:
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]

Training augmentations (applied only during training):
- Random horizontal flip
- Random vertical flip
- Random rotation (±30°)
- Color jitter (brightness, contrast, saturation)
- Random crop and resize
```

### 5.2 Metadata preprocessing

```
Age:       One-hot encode 5-year bins (or numeric: midpoint of bin)
Sex:       One-hot encode {male, female, unknown}
Site:      One-hot encode anatomic site categories
Skin tone: One-hot encode levels 1-6 (or ordinal)
MONET:     Use raw probabilities (already normalized)
```

**Important:** never impute missing clinical facts. If a metadata field is absent, record it as `MISSING` and include a binary indicator in the feature vector.

### 5.3 Versioning preprocessing

Every preprocessing step must be versioned. The `preprocessing_version` is stored in each `ModelRun` record so that any saved prediction can be exactly reproduced.

---

## 6. Out-of-Fold Targets for the Acquisition Estimator

This is a special preprocessing step for training the acquisition-value estimator `h_d`.

### 6.1 Why out-of-fold?

If we train the main classifier on all training data, then compute R_proc on the same training data, we get optimistic (too low) expected-loss values — the model has memorised those samples. The estimated benefit of dermoscopy would be unreliable.

### 6.2 5-fold procedure

```
For each fold k = 1, ..., 5:
    Train set: all training lesions except fold k
    
    Train three models (for k-th fold):
        - Clinical-only model on train set (minus fold k)
        - Dermoscopy-only model on train set (minus fold k)
        - Fused model on train set (minus fold k)
    
    Get predictions on fold k:
        P_{c,z}(i) = clinical+metadata posterior for lesion i in fold k
        P_{c,d,z}(i) = fused posterior for lesion i in fold k
    
    Compute target:
        t_i = R_proc({c,z})_i - R_proc({c,d,z})_i
             = benefit of adding dermoscopy for lesion i

After all 5 folds, every training lesion has an OOF target t_i.
Train h_d (LightGBM) to predict t_i from pre-acquisition features.
```

### 6.3 What OOF targets look like

```
t_i > 0: adding dermoscopy reduced expected loss → dermoscopy was helpful
t_i < 0: adding dermoscopy increased expected loss → dermoscopy was harmful (NIR case)
t_i = 0: dermoscopy made no difference
```

The distribution of t_i values is used to:
1. Train h_d
2. Compute the NIR (fraction of cases where t_i < 0)
3. Compute the MSES (which evidence subsets are sufficient for each case)

---

## 7. Data Leakage Rules (Critical)

```
LEAKAGE PREVENTION RULES — violating these invalidates the research

Rule 1: Temperature scaling parameters → calibrated on VALIDATION split only
Rule 2: Conformal quantile q̂ → computed on CALIBRATION split only  
Rule 3: LTT threshold λ* → computed on CALIBRATION split only
Rule 4: OOF targets for h_d → must use out-of-fold procedure
Rule 5: Test split → touched ONCE, after pre-registration file is committed
Rule 6: If using external test (HAM10000, PAD-UFES-20) → keep separate from MILK10k pipeline
Rule 7: Synthetic text → never encodes the diagnosis or outcome
Rule 8: DERMA-STRESS corruptions → applied AFTER all splits are created
```

If any of these rules are violated, the corresponding experiment is invalid and must be rerun.

---

## 8. MONET Concept Annotations

MONET (Morphological Annotations) provides probabilities for ~30 dermoscopic concepts for each lesion. Examples:

| Concept | Description | Values |
|---------|-------------|--------|
| Atypical pigment network | Irregular pigment structures | [0, 1] |
| Blue-white veil | Blue-white overlay | [0, 1] |
| Regression structures | White scar-like areas | [0, 1] |
| Vascular structures | Blood vessel patterns | [0, 1] |
| ... | ... | ... |

These can be used as:
- Additional metadata features in the fusion model
- Features for the acquisition-value estimator
- Ground truth for the RAG explanation module (cite concept annotations as evidence)

---

## 9. Data Documentation Checklist

Before any experiment, document:

```
☐ Dataset version and source URL
☐ Split version (e.g., v1-lesion-60-15-10-15)
☐ Number of samples per class per split
☐ Class imbalance ratios
☐ Label mapping applied (if any)
☐ Preprocessing version
☐ Random seeds used for splits
☐ Deduplication method
☐ Missing value handling
☐ Licence and data-use conditions
☐ Separate storage of synthetic vs real data
```

---

## Summary

| Topic | Key Decision |
|-------|-------------|
| Primary dataset | MILK10k (5,240 lesions, paired, 96% confirmed) |
| Split | 60/15/10/15 train/val/cal/test, stratified, lesion-level |
| Label grouping | 11 classes → coarse groups + binary malignancy |
| Statistical power | ~786 test lesions; binary AUROC more stable than per-class |
| Leakage mitigation | By ISIC lesion ID; separate calibration split; OOF targets |
| Secondary | Derm7pt for replication (if time allows) |
| Shift dataset | PAD-UFES-20 for Family D only (clinical branch, no dermoscopy) |
| Preprocessing | Versioned; never impute clinical facts; record MISSING explicitly |

---

*Next: [`05_Architecture_Deep_Dive.md`](./05_Architecture_Deep_Dive.md) — Every system component explained in depth →*
