# 🧭 DERMA-GUARD — Complete Learning Guide
### *Read This First. Everything Else Branches From Here.*

---

## What Is This Documentation?

This folder contains **10 comprehensive learning documents** that teach the entire DERMA-GUARD system from scratch — from the clinical problem all the way to deployment and comparative analysis. You do not need an advanced machine learning background to start. Each document builds on the previous one.

---

## 📚 Recommended Learning Order

> Follow this order exactly if you are new to the project. Skip sections you already know.

| Step | File | What You Will Learn | Time |
|------|------|---------------------|------|
| **1** | [`01_System_Overview.md`](./01_System_Overview.md) | The clinical problem, why DERMA-GUARD exists, the 3-action framework (PROCEED / ACQUIRE / DEFER), and the big picture of every component | 30 min |
| **2** | [`02_ML_Concepts.md`](./02_ML_Concepts.md) | Every AI/ML concept used — classification, calibration, uncertainty, OOD detection, conformal prediction, ensembles — explained with intuition before math | 60 min |
| **3** | [`03_Mathematical_Foundations.md`](./03_Mathematical_Foundations.md) | All equations derived step by step: expected loss, LTT theorem, conformal sets, acquisition efficiency, NIR, MSES | 45 min |
| **4** | [`04_Data_Pipeline.md`](./04_Data_Pipeline.md) | How MILK10k is structured, how splits are made leak-free, label grouping, and preprocessing | 30 min |
| **5** | [`05_Architecture_Deep_Dive.md`](./05_Architecture_Deep_Dive.md) | Every system component (Vision Intelligence, Evidence Manager, EADE, RAG, Persistence) explained with data flow diagrams | 60 min |
| **6** | [`06_DERMA_STRESS_Protocol.md`](./06_DERMA_STRESS_Protocol.md) | The 7 stress-test families, why each matters, and how results and guarantee violations are interpreted | 30 min |
| **7** | [`07_Research_Methodology.md`](./07_Research_Methodology.md) | How to run experiments, compare baselines B1–B8, interpret metrics, and write honest results | 45 min |
| **8** | [`08_Platform_Engineering.md`](./08_Platform_Engineering.md) | Tech stack, API design, database schema, background jobs, deployment, and security | 45 min |
| **9** | [`09_Glossary_and_References.md`](./09_Glossary_and_References.md) | Every clinical & ML term defined, mathematical symbols, and 15 key academic references | Reference |
| **10** | [`10_Comparison_With_D_Guard.md`](./10_Comparison_With_D_Guard.md) | Architectural and algorithmic comparison with `D-Guard_System` (Evidential Dempster-Shafer on PAD-UFES-20) | 20 min |

**Total estimated reading time: ~6 hours**

---

## 🗺️ Concept Map — How Everything Connects

```
CLINICAL PROBLEM
│
│  "A skin lesion photo arrives. What should the AI do?"
│
▼
EVIDENCE SIGNALS (what we measure)
├── Image Quality          → Is the photo usable?
├── Model Uncertainty      → How confident is the prediction?
├── Cross-Modal Consistency → Do clinical and dermoscopy agree?
├── OOD Score              → Is this unlike anything trained on?
└── Missingness            → What evidence is absent?
│
▼
EVIDENCE MANAGER (aggregates all signals into a versioned feature record)
│
▼
EADE — Evidence-Adaptive Decision Engine
│
│  Expected-Loss Rule (Eq. 1–4)
│  + Conformal Mixed-Set Safety Rule
│  + LTT-Calibrated Threshold (statistical guarantee)
│
├──► PROCEED  → Return bounded assessment + limitations
├──► ACQUIRE  → Request dermoscopic image (if benefit > cost)
└──► DEFER    → Route to human reviewer
│
▼
DERMA-STRESS (tests the policy under 7 failure families)
│
▼
RESULTS + GUARANTEE-VIOLATION ANALYSIS + FAIRNESS
```

---

## ❓ Quick Orientation: The 5 Most Important Concepts

If you only have 15 minutes, understand these five things:

1. **The 3-action policy** — The system never just predicts a label. It always asks: *should I answer, ask for more evidence, or hand over to a human?*

2. **Expected-loss minimisation** — Each action has a mathematically computed expected cost. The cheapest action wins. This replaces hand-tuned rules.

3. **The LTT guarantee** — After calibration, the system can say: *"The probability I miss a malignancy while auto-proceeding is ≤ 5%, and this holds with 90% statistical confidence."* This is the key scientific contribution.

4. **DERMA-STRESS** — The guarantee only holds when test data looks like calibration data. DERMA-STRESS systematically measures *by how much* the guarantee breaks under realistic failures.

5. **Honest science** — Every result is labelled. The system may NOT outperform baselines. That is a valid, publishable finding.

---

## 👥 Who Owns What (Three-Member Team)

| Member | Primary Role | Core Deliverable |
|--------|-------------|-----------------|
| **Member 1** | ML, calibration, EADE | Reproducible Python package + result tables |
| **Member 2** | Consistency module, DERMA-STRESS, fairness, RAG | DERMA-STRESS package + consistency module + literature audit |
| **Member 3** | Web platform, API, deployment | Deployable web application |

> Every member must be able to explain the **full pipeline** and the **LTT guarantee** in plain words.

---

## 🚦 Project Status Flags Used in Documentation

| Flag | Meaning |
|------|---------|
| ✅ Verified | Existence and key claim confirmed in this revision |
| 🔍 To verify | Cited in v1.0 but not yet confirmed — do NOT cite without checking |
| 📌 Pre-registered | Value is locked; do not change after test split is touched |
| ⚠️ Limitation | Known weakness — must be disclosed in paper |
| 🔴 Out of scope | Will not be built in this version |

---

## 📁 Other Files in This Repository

| File | Description |
|------|-------------|
| [`DERMA_GUARD_v2_Project_Specification.md`](./DERMA_GUARD_v2_Project_Specification.md) | **Authoritative v2.0 specification** — primary source of truth |
| [`DERMA_GUARD_Literature_Survey_v2.md`](./DERMA_GUARD_Literature_Survey_v2.md) | Complete related work audit and base paper dossier |
| [`DERMA_GUARD_v1_Specification_Archive.md`](./DERMA_GUARD_v1_Specification_Archive.md) | Archived v1.0 specification (superseded by v2.0) |
| [`../paper/main.tex`](../paper/main.tex) | Overleaf-ready IEEE formatted research paper draft |

---

*Start with [`01_System_Overview.md`](./01_System_Overview.md) →*
