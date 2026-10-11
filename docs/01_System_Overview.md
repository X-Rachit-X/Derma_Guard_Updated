# 01 — System Overview
### *The Clinical Problem, the Big Picture, and Why DERMA-GUARD Exists*

> **Prerequisites:** None. This is the starting point.  
> **Next:** [`02_ML_Concepts.md`](./02_ML_Concepts.md)

---

## 1. The Clinical Reality

### What is a skin lesion?

A skin lesion is any abnormal growth or discolouration on the skin. The word covers everything from an ordinary mole to a potentially life-threatening melanoma. The key challenge is that many dangerous lesions look unremarkable to the naked eye — and many harmless ones look alarming.

### Why does early detection matter?

Melanoma, the deadliest form of skin cancer, has a **5-year survival rate above 98%** when caught at Stage I, but drops to under **20%** at Stage IV. The bottleneck is access to skilled dermatologists — there are not enough, especially in low-resource settings.

### The two imaging modalities

| Modality | How taken | What it shows | Cost/availability |
|----------|-----------|---------------|------------------|
| **Clinical close-up** | Any smartphone camera | Surface colour, shape, border | Free, immediate |
| **Dermoscopy** | Handheld dermoscope device | Sub-surface structures (pigment networks, vessels) | Requires device + trained operator |

A 2026 reader study on MILK10k (Müller et al.) found that **both modalities carry independent diagnostic value**: dermoscopy has higher sensitivity for malignancy; clinical close-up has higher specificity. Neither alone is sufficient in all cases.

---

## 2. The Core Decision Problem

Imagine a screening AI receives a clinical photograph. It has three options:

```
┌─────────────────────────────────────────────────────────────┐
│  Clinical photograph arrives                                 │
│                                                              │
│  Option 1 — PROCEED                                          │
│    "I have enough confidence. Here is my assessment."        │
│    Risk: I might be wrong and miss a malignancy.             │
│                                                              │
│  Option 2 — ACQUIRE (request dermoscopy)                     │
│    "The dermoscopic image would help. Please upload it."     │
│    Cost: extra device, extra operator time, extra wait.      │
│                                                              │
│  Option 3 — DEFER (route to a human)                         │
│    "This case is beyond automated analysis. A dermatologist  │
│     should review it."                                       │
│    Cost: human expert time, potential delay.                 │
└─────────────────────────────────────────────────────────────┘
```

**The wrong strategy for each option:**
- Always PROCEED → misses malignancies when evidence is poor
- Always ACQUIRE → wastes dermoscope resources for easy cases  
- Always DEFER → overloads human reviewers, defeats the purpose of AI

**DERMA-GUARD's approach:** choose the action that minimises *expected clinical loss*, with a mathematical guarantee that the miss rate stays below a pre-specified level.

---

## 3. What Makes DERMA-GUARD Different

Most dermatology AI systems are **classifiers** — they take an image and output a label or probability. DERMA-GUARD adds an **evidence management and decision layer** on top of classification.

### The gap in existing work

| Approach | What it does | What it misses |
|----------|-------------|----------------|
| Standard classifier | Predicts label | No abstention, no acquisition |
| HOT (Mehta et al.) | Recommends dermoscopy when uncertain | No cost model, no statistical guarantee |
| Selective classification | Abstains when uncertain | Cannot ask for more evidence |
| Active feature acquisition | Acquires features under budget | Not applied in dermatology with risk guarantees |
| Conformal prediction (Fayyad et al.) | Gives uncertainty sets | Used for UQ, not to drive actions |
| **DERMA-GUARD** | All three actions + cost model + statistical guarantee + stress tests | See limitations section |

### The three scientific contributions

**C1. Risk-controlled EADE**  
An expected-loss minimisation policy with thresholds calibrated by **Learn-then-Test (LTT)**, giving a finite-sample guarantee: *malignancy miss rate on auto-proceeded cases ≤ α with probability ≥ 1 − δ*.

**C2. DERMA-STRESS**  
A versioned, reproducible stress-test protocol across 7 failure families — including the first systematic measurement of **where the statistical guarantee breaks** under distribution shift.

**C3. Acquisition analysis**  
An acquisition-value estimator evaluated against oracle and random bounds, with two custom metrics: MSES and NIR.

---

## 4. The System at 10,000 Feet

```
INPUTS
  ├── Clinical photograph (always present ideally)
  ├── Dermoscopic image (may be missing)
  └── Metadata: age, sex, anatomic site, skin tone

       ↓
VISION INTELLIGENCE
  ├── 3-seed ensemble classifier per modality
  ├── Temperature-scaled calibrated probabilities
  ├── Quality estimators (blur, exposure, resolution)
  └── OOD scores (energy, Mahalanobis)

       ↓
CONSISTENCY MODULE
  ├── Jensen-Shannon divergence between modalities
  └── Rule-based contradiction flags (metadata vs text)

       ↓
EVIDENCE MANAGER
  └── Aggregates all signals → versioned feature record

       ↓
EADE (Evidence-Adaptive Decision Engine)
  ├── Expected-loss computation (Eq. 1–4)
  ├── Conformal mixed-set safety rule
  └── LTT-calibrated threshold check

       ↓
ACTION
  ├── PROCEED → Assessment shown to user with caveats
  ├── ACQUIRE → UI prompts for dermoscopy upload
  └── DEFER   → Case queued for human reviewer

       ↓
PERSISTENCE & AUDIT
  └── Append-only records: evidence, model run, action, reviewer

       ↓
DERMA-STRESS (separate evaluation pipeline)
  └── Runs same cases through all baselines under 7 failure families
```

---

## 5. The Three Research Hypotheses

These are **predictions to be tested**, not assumed results.

### H1 — Risk control
> On the clean MILK10k test split, EADE achieves the target malignancy miss rate α on proceeded cases at **higher coverage** than confidence-threshold abstention calibrated to the same α.

### H2 — Acquisition efficiency
> At a matched acquisition rate, EADE recovers a larger fraction of oracle acquisition benefit than random acquisition, entropy-threshold (HOT-style), and greedy mutual-information acquisition.

### H3 — Robustness
> Under combined evidence failures, EADE's selective risk degrades less than the baselines', and its risk guarantee degrades **measurably** (reported) rather than silently.

> **Important:** If these hypotheses are not supported, the team will report that finding honestly. A negative result is still publishable if the benchmark and guarantee-violation analysis are rigorous.

---

## 6. What DERMA-GUARD Is NOT

| Claim | Status |
|-------|--------|
| A medical device | 🔴 No. Research prototype only |
| An autonomous diagnosis system | 🔴 No. AI provides bounded assessments |
| Proven to outperform all baselines | 🔴 Not yet — it's a hypothesis |
| Trained from scratch | 🔴 No. Fine-tuned pretrained models |
| Clinically validated | 🔴 No. Public benchmark results only |
| Foundation model | 🔴 No. ConvNeXt-Tiny / EfficientNet-B0 |

---

## 7. The Evidence Management Philosophy

The core insight of DERMA-GUARD is that **evidence quality is distinct from prediction confidence**.

| Signal | What it measures | Example failure |
|--------|-----------------|-----------------|
| Image quality | Technical usability of the photo | Blurry, dark, low resolution |
| Model reliability | Empirical validation of the model | Trained on different demographics |
| Predictive uncertainty | Ambiguity of the prediction | Close call between benign and malignant |
| Cross-modal consistency | Agreement between clinical and dermoscopy | Image says benign, dermoscopy says malignant |
| Missingness | Explicitly absent inputs | No dermoscopy available |
| Evidence value | Estimated benefit of acquiring more evidence | Would dermoscopy change the decision? |
| OOD signal | Does this case look like the training distribution? | Different device, different population |

A confident model can still have poor-quality inputs or be completely out of distribution. DERMA-GUARD tracks all these signals separately and feeds them to the decision engine.

---

## 8. Scope Boundaries (What Will and Won't Be Built)

### In scope ✅
- Patient-case-centred web application
- Clinical-only, dermoscopy-only, and fused image classifiers on MILK10k with modality dropout
- Probability calibration (temperature scaling) and uncertainty (deep ensemble)
- Conformal prediction sets and LTT-calibrated thresholds
- Expected-loss proceed/acquire/defer policy with explicit cost model
- Acquisition-value estimator
- Cross-modal consistency signal and contradiction detection
- OOD scoring (energy + Mahalanobis)
- DERMA-STRESS protocol and experiment runner
- Baseline comparisons and ablations
- Fairness analysis across 6 skin-tone levels
- Small RAG-based explanation feature
- Persistent case history and audit log

### Out of scope for initial release 🔴
- Training a foundation model from scratch
- Claiming clinical benefit or regulatory readiness
- Autonomous diagnosis or treatment recommendations
- Microservices, Kafka, Kubernetes
- Reinforcement-learning acquisition policies
- LLM authority over predictions or routing
- Synthetic text presented as genuine patient history

---

## 9. Summary

| Concept | One-Line Summary |
|---------|-----------------|
| Clinical problem | Skin lesion AI must decide when to answer, acquire dermoscopy, or defer |
| Key insight | Evidence quality ≠ prediction confidence; manage both explicitly |
| EADE | Expected-loss decision rule with a statistical guarantee |
| LTT | Calibration method that provides a finite-sample miss-rate bound |
| DERMA-STRESS | 7-family stress test protocol including guarantee-violation measurement |
| Fairness | Group-conditional conformal calibration across 6 skin-tone levels |
| Honest science | Negative results are equally valid — report everything |

---

*Next: [`02_ML_Concepts.md`](./02_ML_Concepts.md) — Every AI concept explained with intuition →*
