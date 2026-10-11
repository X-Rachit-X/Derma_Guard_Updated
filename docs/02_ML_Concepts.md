# 02 — ML Concepts
### *Every AI/ML Concept Used in DERMA-GUARD, Explained With Intuition First*

> **Prerequisites:** [`01_System_Overview.md`](./01_System_Overview.md)  
> **Next:** [`03_Mathematical_Foundations.md`](./03_Mathematical_Foundations.md)

---

## Part A — Classification Fundamentals

### A1. What is a classifier?

A classifier is a function that takes an input (e.g., an image) and outputs a probability for each possible class label.

**Example in DERMA-GUARD:**
```
Input: dermoscopic image
Output: {
  melanoma:        0.62,
  basal_cell:      0.18,
  seborrheic_kerat: 0.12,
  nevus:           0.05,
  ...
}
```

The class with the highest probability is the prediction. But DERMA-GUARD uses the **entire probability distribution**, not just the argmax, to compute expected loss.

### A2. What are "logits" vs "probabilities"?

A neural network's last layer outputs raw numbers called **logits** — unbounded, can be negative or very large. These are converted to probabilities via the **softmax** function:

```
P(class k) = exp(logit_k) / sum_over_all_classes( exp(logit_j) )
```

Key property: all probabilities sum to 1.

### A3. Calibration — why raw softmax probabilities are misleading

Modern deep networks are **overconfident**: they output `P(malignant) = 0.97` even for borderline cases. When you say "my model is 97% confident," that should mean it's correct 97% of the time on such cases. A poorly calibrated model isn't.

**Calibration** is the process of adjusting outputs so that confidence actually matches empirical accuracy.

```
Well calibrated:   "70% confident" → correct 70% of the time  ✅
Overconfident:     "70% confident" → correct 55% of the time  ❌
Underconfident:    "70% confident" → correct 90% of the time  ❌
```

**Why it matters for DERMA-GUARD:** The expected-loss computation and conformal prediction sets both require accurate probabilities. Miscalibrated probabilities produce incorrect expected losses and unreliable conformal guarantees.

### A4. Temperature Scaling — the calibration method used

Temperature scaling adds a single learnable scalar T (the "temperature") to the logits:

```
Calibrated P(class k) = exp(logit_k / T) / sum( exp(logit_j / T) )
```

- T > 1 → probabilities become more uniform (softer) → less overconfident
- T < 1 → probabilities become more peaked → more overconfident
- T = 1 → no change (original softmax)

T is fitted on the **validation split** by minimising the log-loss (negative log-likelihood). It is a separate scalar per modality configuration (clinical-only, dermoscopy-only, fused).

---

## Part B — Uncertainty Quantification

### B1. Two types of uncertainty

| Type | Name | What causes it | Example |
|------|------|---------------|---------|
| **Aleatoric** | Data uncertainty | Inherent noise in the input | Blurry image of an ambiguous lesion |
| **Epistemic** | Model uncertainty | Gaps in training data | Case unlike anything in training set |

Aleatoric uncertainty cannot be reduced by collecting more data. Epistemic uncertainty can be reduced by training on more similar cases.

DERMA-GUARD estimates both:
- Aleatoric: captured in calibrated class probabilities
- Epistemic: captured by **ensemble disagreement**

### B2. Deep Ensembles

Train the same model architecture **3 times** with different random seeds (different random initialisation, different data shuffling). Each produces a different posterior:

```
Seed 0 → P₀(malignant) = 0.71
Seed 1 → P₁(malignant) = 0.68
Seed 2 → P₂(malignant) = 0.82
```

**Ensemble average:** P̄ = (0.71 + 0.68 + 0.82) / 3 = 0.737

The **spread** of individual predictions measures epistemic uncertainty.

### B3. Predictive Entropy

Entropy measures how "spread out" a probability distribution is.

```
H[P̄] = -∑_k  P̄(k) × log(P̄(k))
```

- Maximum entropy (uniform distribution across K classes) → completely uncertain
- Zero entropy (all mass on one class) → completely certain

**Example:**
```
P̄ = {malignant: 0.74, benign: 0.26}  →  H = 0.86  (fairly certain)
P̄ = {malignant: 0.51, benign: 0.49}  →  H = 1.00  (very uncertain)
```

High entropy triggers acquisition in HOT-style baselines (B5).

### B4. Mutual Information (Epistemic Uncertainty)

Mutual information separates *total* uncertainty from *data* uncertainty:

```
MI = H[P̄] - (1/M) × ∑_j H[Pⱼ]
     ↑              ↑
  total entropy   average entropy of individual models
```

- High MI → ensemble members disagree → high epistemic uncertainty
- Low MI despite high H[P̄] → all models agree the case is ambiguous → aleatoric

DERMA-GUARD uses MI as a feature in the acquisition-value estimator.

---

## Part C — Out-of-Distribution Detection

### C1. What is OOD?

A model is trained on data from a certain distribution (e.g., European skin tones, certain camera devices, certain lighting conditions). When it encounters a case from a **different distribution**, its predictions become unreliable — even if it outputs high confidence.

```
Training data: mostly light-skinned patients, standard dermoscopes
OOD case:     dark-skinned patient, smartphone dermoscopy adapter
→ Model may confidently misclassify
```

### C2. Energy-Based OOD Score

The **energy score** (Liu et al., 2020) uses the raw logits without going through softmax:

```
E(x) = -T × log( ∑_k exp(fₖ(x) / T) )
```

- In-distribution inputs: typically large negative energies (concentrated logits)
- OOD inputs: typically smaller negative energies (diffuse logits)

The energy score can be computed for any input without needing a separate OOD model.

### C3. Mahalanobis Distance OOD Score

This method uses the feature embeddings from the penultimate layer (just before the classification head).

During training, compute the mean and covariance of features for each class. For a new input:

```
d_M(x) = √[(φ(x) - μ_ŷ)ᵀ × Σ⁻¹ × (φ(x) - μ_ŷ)]
```

- φ(x) = feature embedding of the new input
- μ_ŷ = mean feature vector for the predicted class
- Σ = shared covariance matrix across classes

OOD inputs have embeddings that don't cluster near any class mean → high Mahalanobis distance.

### C4. How DERMA-GUARD uses OOD scores

OOD scores are:
1. **A feature in the acquisition-value estimator** (high OOD score → more likely to benefit from dermoscopy)
2. **A reason code for DEFER** (if OOD score is very high, the safety layer can force deferral)
3. **Evaluated in DERMA-STRESS Family E** (leave-class-out experiments)

---

## Part D — Image Quality Estimation

### D1. Why image quality matters

A blurry photograph can produce high-confidence but wrong predictions. Standard classifiers have no way to detect this. DERMA-GUARD explicitly estimates image quality.

### D2. Laplacian variance (blur detection)

```python
laplacian = cv2.Laplacian(image, cv2.CV_64F)
blur_score = laplacian.var()
```

Sharp images have high variance in their second derivative (Laplacian). Blurry images have low variance. A low blur score triggers the quality-signal pipeline.

### D3. Other quality metrics

| Metric | What it detects | Method |
|--------|----------------|--------|
| Blur score | Defocus, motion blur | Laplacian variance |
| Exposure | Overexposed / underexposed | Mean pixel intensity |
| Resolution | Insufficient detail | Image dimensions |
| Contrast | Low dynamic range | Pixel value range |
| Artifact score | JPEG compression artifacts | High-frequency noise |

Quality scores become features for the acquisition-value estimator and are displayed in the UI as quality badges.

---

## Part E — Multimodal Fusion

### E1. What is multimodal fusion?

"Fusion" means combining information from multiple sources (modalities). In DERMA-GUARD:
- **Clinical image** (`xᶜ`) — surface appearance
- **Dermoscopic image** (`xᵈ`) — sub-surface structures
- **Structured metadata** (`z`) — age, sex, anatomic site, skin tone

### E2. Late Fusion (used in DERMA-GUARD)

Each modality is processed by its own model branch. The outputs (probability vectors) are combined after classification:

```
Clinical branch:    P_c = [0.62, 0.18, 0.12, ...]
Dermoscopy branch:  P_d = [0.71, 0.14, 0.09, ...]
Metadata branch:    P_z = [0.58, 0.21, 0.14, ...]

Fused:  P_fused = weighted combination of P_c, P_d, P_z
```

### E3. Modality Dropout (key training trick)

During training, randomly drop one or more modalities from each batch with some probability. This forces the model to learn to work with any subset of inputs.

**Why it matters:** At test time, dermoscopy may be missing. A model trained with modality dropout gracefully handles this; one trained only on complete cases does not.

### E4. Cross-Modal Consistency

Even when both modalities are present, they may disagree. DERMA-GUARD measures this with **Jensen-Shannon divergence (JSD)**:

```
JSD(P_c || P_z) = ½ KL(P_c || m) + ½ KL(P_z || m)
where m = ½(P_c + P_z)
```

KL divergence measures how different one distribution is from another. JSD is a symmetric, bounded (0 to ln2) version.

- JSD ≈ 0 → modalities agree strongly
- JSD ≈ ln2 → modalities completely disagree → triggers ACQUIRE or DEFER

---

## Part F — Selective Prediction

### F1. What is selective prediction?

A classifier that can **abstain** (refuse to answer) when uncertain. Instead of always producing a label, it can say "I don't know" and pass the case to a human.

```
Without selective prediction:  answer rate = 100%, error rate = X%
With selective prediction:     answer rate < 100%, error rate < X%
```

**Coverage** = fraction of cases the system answers (does not abstain)
**Selective risk** = error rate among answered cases

The goal: achieve low risk at high coverage.

### F2. Risk-Coverage Curves

By sweeping a threshold parameter, you trade coverage for risk:

```
More permissive threshold → higher coverage, higher risk
More restrictive threshold → lower coverage, lower risk
```

Plotting selective risk vs coverage gives a risk-coverage curve. A good method has lower risk at the same coverage — its curve is below and to the right of a weaker method.

### F3. AURC and AUGRC

**AURC** (Area Under Risk-Coverage Curve) integrates the curve across all coverage levels. Lower = better.

**AUGRC** (Area Under *Generalised* Risk-Coverage Curve, Traub et al. 2024) fixes known flaws in AURC. It is interpretable as "average risk of undetected failures." DERMA-GUARD uses AUGRC as the primary multi-threshold metric.

---

## Part G — Conformal Prediction

### G1. The basic idea

Conformal prediction is a framework for constructing **prediction sets** that are guaranteed to contain the true label with a specified probability, without making assumptions about the data distribution.

```
Standard prediction: "I predict class 3"
Conformal prediction: "The true class is in {class 3, class 7} with probability ≥ 95%"
```

The set is larger when uncertainty is higher.

### G2. How a conformal set is constructed

**Step 1 — Calibration.** On a held-out calibration set (not used for training), compute a "nonconformity score" for each sample — a measure of how surprising the sample is given the model.

Common choice: **APS/RAPS score** — based on the sorted softmax probabilities.

**Step 2 — Quantile.** Find the α-quantile of the nonconformity scores across the calibration set. Call this q̂.

**Step 3 — Prediction.** For a new test point, include all classes whose cumulative softmax probability (sorted descending) falls within q̂.

**Guarantee:** The resulting set contains the true class with probability ≥ 1 - α over the calibration draw.

### G3. The Mixed-Set Rule (critical for DERMA-GUARD)

A conformal set that contains **both benign and malignant classes** means the model is genuinely uncertain about the malignancy grouping.

**DERMA-GUARD's rule:**
```
If conformal set ∩ Benign ≠ ∅  AND  conformal set ∩ Malignant ≠ ∅:
    → Block PROCEED
    → Trigger ACQUIRE (if dermoscopy available)
    → Otherwise DEFER
```

This is intuitive: when we can't even agree on whether the lesion is benign or malignant, we should not auto-proceed.

### G4. Marginal vs Conditional Coverage

**Marginal coverage:** The guarantee holds *on average* over all cases. A fraction (1 - α) of all cases will have their true label in the conformal set.

**Conditional coverage:** The guarantee holds *for each subgroup*. This is stronger but harder to achieve.

**Group-conditional conformal** (used for fairness in DERMA-GUARD): run separate calibration per skin-tone group, ensuring the guarantee holds for each group independently.

---

## Part H — Learn-then-Test (LTT)

### H1. Why we need LTT

Conformal prediction gives a coverage guarantee (set contains true label). But we want a **risk guarantee** — specifically, that the malignancy miss rate is bounded.

Standard conformal methods don't directly give this. LTT does.

### H2. The LTT idea

You want to select a threshold λ (a policy parameter) such that the "bad event" probability (e.g., proceeding and missing malignancy) is ≤ α with high probability.

**Problem:** if you try many thresholds and pick the one that looks best on calibration data, you overfit and the guarantee is violated on test data.

**LTT solution:** treat threshold selection as a multiple-testing problem. For a grid of thresholds λ₁ < λ₂ < ... < λJ:

1. Compute empirical risk at each threshold on calibration data
2. Run a statistical test (Hoeffding-Bentkus bound or binomial tail) for each threshold
3. Use fixed-sequence testing: walk from most conservative to most permissive, and select the most permissive threshold that passes the test

This controls the Type I error rate across the entire grid, giving a **valid** guarantee: with probability ≥ 1 - δ, the selected λ achieves risk ≤ α.

### H3. The key limitation

The LTT guarantee assumes the calibration and test data are **exchangeable** (roughly: drawn from the same distribution). Under distribution shift, the guarantee may be violated. DERMA-STRESS measures this violation — that measurement is a core result.

---

## Part I — Active Feature Acquisition

### I1. The concept

**Active Feature Acquisition (AFA):** decide at test time, for each sample, whether to observe additional features (at a cost) before making a prediction.

In DERMA-GUARD, the "feature" is the dermoscopic image.

```
Given: clinical image + metadata
Question: is it worth paying cost(dermoscopy) to observe the dermoscopic image?
```

### I2. Expected benefit estimation

We want to estimate: "if I acquire dermoscopy for this case, by how much will my expected loss decrease?"

This is the **acquisition-value estimator** hₐ(xₛ). It is trained using **out-of-fold predictions** on training cases (where we know what dermoscopy would have shown).

If estimated benefit > acquisition cost → ACQUIRE
Otherwise → PROCEED or DEFER

### I3. Oracle and random bounds

| Bound | Description | AE value |
|-------|------------|---------|
| **Oracle** | Always acquires the cases with the highest realised benefit | AE = 1.0 |
| **Random** | Acquires a random subset of cases | AE = 0.0 |
| **Our policy** | Uses the estimator to decide | 0 < AE < 1 |
| **HOT-style** | Acquires when entropy > threshold | 0 < AE < 1 |

**Acquisition Efficiency AE(r)** at rate r measures how much of the oracle benefit the policy recovers.

---

## Part J — Learning to Defer

### J1. What is learning-to-defer?

Learning-to-defer (L2D, Mozannar & Sontag 2020) jointly trains a classifier and a deferral rule. Unlike selective prediction (which defers based on post-hoc thresholds), L2D trains the model to recognise cases where the human expert would do better.

### J2. In DERMA-GUARD

L2D is implemented as **Baseline B4**. It is a strong comparison because:
- It uses a principled training objective (consistent surrogate loss)
- It can potentially defer on different cases than confidence-threshold abstention
- It doesn't use an explicit cost model or acquire evidence

DERMA-GUARD's EADE should outperform B4 specifically by its combination of (a) evidence acquisition and (b) cost-aware risk control.

---

## Part K — RAG: Retrieval-Augmented Generation

### K1. What is RAG?

RAG (Retrieval-Augmented Generation) combines a retrieval system (search in a document database) with a language model (generates text). The idea: instead of relying on the LLM's parametric knowledge, retrieve relevant passages and use them as context.

```
Query: "What are the dermoscopic features of melanoma?"
→ Retrieve: 3 passages from curated dermatology guidelines
→ Generate: "According to [passage 1], melanoma shows..."
```

### K2. How it works in DERMA-GUARD

The RAG module is an **explanation feature** — it provides source-grounded context for the model's assessment. It is **not** in the decision path (does not influence PROCEED/ACQUIRE/DEFER).

**Retrieval pipeline:**
1. BM25 (lexical search) retrieves passages matching medical terms
2. pgvector (semantic search) retrieves passages based on meaning
3. Results are reranked and the top passages are cited

**Important constraints:**
- Every claim is linked to a specific passage
- Unsupported claims are flagged
- Citations are never invented
- RAG output is labelled as "supporting context", not "clinical validation"

### K3. Hybrid retrieval

**BM25** works on exact keyword matches and is fast, interpretable, and good for specific terminology.
**Vector search** works on semantic similarity and is good for paraphrases and related concepts.
**Hybrid** = both, with a reranker to combine results. Better coverage and precision than either alone.

---

## Summary Table: ML Concepts in DERMA-GUARD

| Concept | Where used | Purpose |
|---------|-----------|---------|
| Temperature scaling | After each classifier | Calibrate probabilities |
| Deep ensemble (3 seeds) | Vision Intelligence | Epistemic uncertainty |
| Predictive entropy | Acquisition estimator feature | Measure total uncertainty |
| Mutual information | Acquisition estimator feature | Separate epistemic uncertainty |
| Energy OOD score | Safety layer feature | Detect distribution shift |
| Mahalanobis OOD | Safety layer feature | Detect distribution shift |
| JS divergence | Consistency module | Measure cross-modal disagreement |
| Image quality scores | Evidence Manager | Flag low-quality inputs |
| Conformal sets (RAPS/APS) | Safety layer | Mixed-set blocking rule |
| LTT calibration | Safety layer | Statistical risk guarantee |
| Expected-loss rule | EADE | Main decision criterion |
| Acquisition estimator | EADE | Predict benefit of dermoscopy |
| Active feature acquisition | EADE | Request dermoscopy if beneficial |
| Learning-to-defer | Baseline B4 | Comparison method |
| AUGRC | Evaluation | Primary multi-threshold metric |
| Risk-coverage curves | Evaluation | Visualise coverage-risk tradeoff |
| BM25 + pgvector | RAG explanation | Retrieve supporting passages |
| Group-conditional conformal | Fairness | Per-skin-tone guarantees |

---

*Next: [`03_Mathematical_Foundations.md`](./03_Mathematical_Foundations.md) — All equations derived step by step →*
