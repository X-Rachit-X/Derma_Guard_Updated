# 🛡️ DERMA-GUARD (v2.0)
### *Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for Multimodal Dermatology AI*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%2Bpgvector-336791.svg)](https://www.postgresql.org/)
[![Paper](https://img.shields.io/badge/IEEE-Paper_Draft-informational.svg)](./paper/derma_guard_ieee_paper.tex)

> ⚠️ **Research Prototype Notice:** DERMA-GUARD is an academic research platform and teledermatology decision framework. It is **not** a certified medical device and does not provide autonomous medical diagnoses.

---

## 🔬 The Clinical Problem

A teledermatology AI system does not fail simply because its neural network accuracy is a few points short. It fails because it **answers confidently when it should not** — on a blurred smartphone photo, an incomplete patient history, or a suspicious lesion where clinical photography is ambiguous and contact dermoscopy is missing.

Most medical AI forces a choice: predict a label on whatever data arrives. DERMA-GUARD asks a fundamentally different clinical question:

> **"Given a clinical close-up photograph of a skin lesion, should the AI answer now, request a dermoscopic image first under explicit acquisition costs, or defer to a human specialist — with a mathematically proven statistical guarantee on how often it misses a malignancy?"**

---

## 🚦 The 3-Action Decision Framework

Instead of forced-choice classification, the Evidence-Adaptive Decision Engine (**EADE**) chooses dynamically among three distinct actions:

```
                              LESION INTAKE
                       (Clinical Smartphone Photo)
                                    │
                                    ▼
                     EVIDENCE-ADAPTIVE DECISION ENGINE
                   [Expected-Loss Minimization + LTT]
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
    1. PROCEED                 2. ACQUIRE                  3. DEFER
   Return bounded            Actively request           Route to human
    assessment +           contact dermoscopy         dermatologist with
  conformal set &           if expected gain >         immutable evidence
    limitations             acquisition cost                 packet
```

1. **`PROCEED`** — The clinical photo is sufficient and unambiguous. Returns calibrated posterior probabilities, conformal prediction set $C(x)$, and explicit boundary limitations.
2. **`ACQUIRE`** — The clinical photo is ambiguous, but an out-of-fold value estimator predicts that acquiring a high-resolution dermoscopic image will significantly reduce expected clinical risk exceeding the cost $c_{\text{acq}}$.
3. **`DEFER`** — Severe image degradation, cross-modal contradiction, out-of-distribution pathology, or mixed conformal set. Escalates immediately to human clinical review.

---

## ⚡ Brief: How This Repo Differs From `D-Guard_System`

This repository ([`Derma_Guard_Updated`](https://github.com/X-Rachit-X/Derma_Guard_Updated)) represents the **v2.0 evolution** of our dermatology safety research, significantly expanding upon the earlier prototype in [`D-Guard_System`](https://github.com/X-Rachit-X/D-Guard_System):

| Feature / Dimension | [`D-Guard_System`](https://github.com/X-Rachit-X/D-Guard_System) | **`Derma_Guard_Updated` (This Repo)** |
|---|---|---|
| **Core Research Question** | Which passive evidence modalities should the AI trust vs discount? | **When should the AI answer, ask for dermoscopy, or defer to a doctor?** |
| **Action Space** | **Binary (2 Actions):** `PREDICT` vs `DEFER` | **Ternary (3 Actions):** `PROCEED` vs `ACQUIRE` vs `DEFER` |
| **Dataset & Scale** | **PAD-UFES-20** (2,298 clinical smartphone images, single scale) | **MILK10k** (5,240 lesions, 10,480 paired clinical + dermoscopy images) |
| **Multimodal Scope** | Smartphone photo + Tabular metadata + Narrative text | **Dual visual scales:** Clinical Close-Up + Contact Dermoscopy + Metadata |
| **Mathematical Engine** | Evidential Deep Learning (Dirichlet) + Dempster–Shafer fusion | **Bayes Expected-Loss Minimization** over asymmetric clinical cost matrix |
| **Statistical Guarantees** | Conformal risk control on general prediction error | **Learn-Then-Test (LTT) finite-sample bound on Malignancy Miss Rate** ($\le 5\%$) |
| **Safety Override** | Heuristic abstention ladder based on ignorance & conflict | **Conformal Mixed-Set Rule:** If set contains benign AND malignant $\implies$ No Proceed |
| **Active Sensing** | ❌ Not supported (evidence is static at intake) | ✅ **Learned Value of Information Estimator** for active dermoscopy acquisition |
| **Failure Benchmark** | Evaluated on clean disjoint splits | **DERMA-STRESS Protocol:** 7 systematic failure families (A through G) |
| **Deliverables** | Python research experiment scripts | **Full Platform Monolith:** Next.js UI, FastAPI, Celery/Redis, PostgreSQL, & **IEEE Paper** |

> 📖 *For full mathematical derivations, algebraic proofs, and in-depth architectural comparisons, see [`docs/10_Comparison_With_D_Guard.md`](./docs/10_Comparison_With_D_Guard.md).*

---

## 🏗️ System Architecture

DERMA-GUARD is structured as a robust **modular monolith** with an asynchronous worker pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NEXT.JS / REACT FRONTEND                        │
│   Case Intake │ Evidence Review │ Conformal Visualizer │ Stress Lab   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTPS / REST (Typed OpenAPI)
┌───────────────────────────────────▼────────────────────────────────────┐
│                       FASTAPI BACKEND SERVICE                          │
│   Auth & RBAC  │ Case Controller │ Assessment Engine │ Audit Middleware│
└───────────┬───────────────────────┬─────────────────────────┬──────────┘
            │ SQL                   │ Enqueue Jobs            │ S3 API
┌───────────▼───────────┐ ┌─────────▼─────────┐ ┌─────────────▼──────────┐
│  POSTGRESQL + PGVECTOR │ │  REDIS MESSAGE   │ │     OBJECT STORAGE     │
│   Immutable Audit     │ │      BROKER       │ │   (Local / MinIO)      │
│   Case & Model State  │ └─────────┬─────────┘ │   Dermoscopy & Clinical│
│   RAG Vector Chunks   │           │ Pull Jobs └────────────────────────┘
└───────────────────────┘ ┌─────────▼─────────┐
                          │   CELERY / RQ     │
                          │   WORKER DAEMON   │
                          │ PyTorch Inference │
                          │ EADE Optimization │
                          │ DERMA-STRESS Run  │
                          └───────────────────┘
```

---

## 📚 Complete Learning Documentation Suite

The documentation is organized into a sequential, concept-building learning curriculum:

| Step | Guide | Summary |
|:---:|---|---|
| **00** | [🧭 Master Learning Guide](./docs/00_START_HERE.md) | Reading order, concept map, and project orientation. |
| **01** | [🏥 System Overview](./docs/01_System_Overview.md) | Clinical motivation, 3-action policy, evidence signals, and core hypotheses. |
| **02** | [🧠 ML Concepts](./docs/02_ML_Concepts.md) | Intuition-first breakdown of calibration, uncertainty, OOD, and ensembles. |
| **03** | [📐 Mathematical Foundations](./docs/03_Mathematical_Foundations.md) | Rigorous step-by-step mathematical derivations of all equations (Eq 1–14). |
| **04** | [📊 Data Pipeline](./docs/04_Data_Pipeline.md) | MILK10k cohort structure, leak-free lesion-level splitting, and preprocessing. |
| **05** | [⚙️ Architecture Deep Dive](./docs/05_Architecture_Deep_Dive.md) | Component data flow, Pydantic schemas, and database entity relationships. |
| **06** | [🌪️ DERMA-STRESS Protocol](./docs/06_DERMA_STRESS_Protocol.md) | 7 failure families (A–G), combined perturbations, and guarantee-violation metric. |
| **07** | [🧪 Research Methodology](./docs/07_Research_Methodology.md) | Pre-registration rules, baselines B1–B8, ablation matrix, and bootstrap testing. |
| **08** | [💻 Platform Engineering](./docs/08_Platform_Engineering.md) | Modular monolith architecture, PostgreSQL schema, REST API, and safeguards. |
| **09** | [📖 Glossary & References](./docs/09_Glossary_and_References.md) | Definitions of clinical/ML terms, notation guide, and 15 academic citations. |
| **10** | [🔄 Comparison with D-Guard](./docs/10_Comparison_With_D_Guard.md) | Detailed comparison between `Derma_Guard_Updated` and `D-Guard_System`. |

---

## 📄 Research Paper (IEEE Overleaf Ready)

The full scientific paper has been compiled into an IEEE-format LaTeX document ready for immediate Overleaf compilation:

- **Source File:** [`paper/derma_guard_ieee_paper.tex`](./paper/derma_guard_ieee_paper.tex)
- **Compilation Guide:** [`paper/README.md`](./paper/README.md)
- **Features:** Double-column `IEEEtran` format, 29 academic bibliography citations, baseline comparisons across B1–B8, and formal LTT theorem statements.

---

## 📁 Repository Directory Structure

```text
Derma_Guard_Updated/
├── README.md                           # Master project README & high-level overview
├── .gitignore                          # Standard ignore rules for Python, Node, LaTeX, OS
├── configs/
│   └── prereg.yaml                     # Pre-registered experimental parameters & cost matrix
├── paper/
│   ├── derma_guard_ieee_paper.tex      # Full IEEE research paper LaTeX draft
│   └── README.md                       # Overleaf compilation & setup instructions
└── docs/
    ├── 00_START_HERE.md                # Curriculum index & learning roadmap
    ├── 01_System_Overview.md           # Clinical motivation & 3-action architecture
    ├── 02_ML_Concepts.md               # Intuition-first machine learning fundamentals
    ├── 03_Mathematical_Foundations.md   # Step-by-step derivations of all core equations
    ├── 04_Data_Pipeline.md             # MILK10k processing, splits, and leakage prevention
    ├── 05_Architecture_Deep_Dive.md    # Component specifications & internal data flow
    ├── 06_DERMA_STRESS_Protocol.md     # 7 stress failure families & guarantee violation
    ├── 07_Research_Methodology.md      # Baselines B1–B8, ablations, & statistical tests
    ├── 08_Platform_Engineering.md      # FastAPI backend, PostgreSQL DDL, & REST APIs
    ├── 09_Glossary_and_References.md   # Terminology definitions & scientific citations
    ├── 10_Comparison_With_D_Guard.md   # Deep architectural comparison with D-Guard_System
    ├── DERMA_GUARD_v2_Project_Specification.md  # Official v2.0 Authoritative Specification
    ├── DERMA_GUARD_Literature_Survey_v2.md      # Full literature survey & related work audit
    └── DERMA_GUARD_v1_Specification_Archive.md  # Archived v1.0 initial specification
```

---

## 👥 Three-Member Ownership & Responsibilities

| Team Member | Core Focus Areas | Key Deliverables |
|---|---|---|
| **Member 1** | Machine Learning, Calibration, EADE | Reproducible PyTorch training package, LTT calibration engine, baseline benchmark results. |
| **Member 2** | Cross-Modal Consistency, DERMA-STRESS, Fairness | Stress generation harness, divergence metrics, skin-tone subgroup analysis, literature audit. |
| **Member 3** | Platform Architecture, Web App, DevOps | FastAPI backend, Next.js dashboard, PostgreSQL schema, Docker deployment, CI/CD pipeline. |

---

## ⚖️ Pre-Registration & Scientific Integrity

All primary hypotheses, evaluation metrics, and clinical loss parameters are frozen in [`configs/prereg.yaml`](./configs/prereg.yaml). The evaluation test split is held completely sequestered and evaluated exactly once per frozen model configuration to eliminate cherry-picking and p-hacking.
