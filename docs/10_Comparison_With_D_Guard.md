# 10 — Architectural Comparison: DERMA-GUARD v2 vs D-Guard System
### *Technical Differentiation Between `Derma_Guard_Updated` and `D-Guard_System`*

> **Prerequisites:** All prior documents ([`01_System_Overview.md`](./01_System_Overview.md) through [`09_Glossary_and_References.md`](./09_Glossary_and_References.md)).  
> **Source Repositories:**  
> - Reference Repo: [D-Guard_System](https://github.com/X-Rachit-X/D-Guard_System)  
> - Current Repo: [Derma_Guard_Updated](https://github.com/X-Rachit-X/Derma_Guard_Updated)

---

## 1. Executive Summary: What Changed and Why

Both projects tackle a fundamental vulnerability in medical artificial intelligence: **deep neural networks fail silently by making overly confident predictions on degraded, conflicting, or incomplete evidence.**

However, they operate under fundamentally different clinical settings, mathematical frameworks, and action spaces:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          THE CORE DISTINCTION                          │
├────────────────────────────────────────────────────────────────────────┤
│ D-Guard_System asks:                                                   │
│ "Given a smartphone image, tabular metadata, and narrative notes from  │
│  a patient, which modality should I believe, and when should I DEFER?" │
│ ➔ Mathematical Tool: Evidential Deep Learning + Dempster–Shafer Fusion │
│ ➔ Action Space: 2 Actions (PREDICT vs DEFER)                           │
│ ➔ Dataset: PAD-UFES-20 (2,298 single-scale clinical images)            │
├────────────────────────────────────────────────────────────────────────┤
│ Derma_Guard_Updated (v2.0) asks:                                       │
│ "Given an initial smartphone photo, can the AI answer safely, should   │
│  it actively REQUEST high-resolution dermoscopy under explicit costs,  │
│  or must it ESCALATE to a dermatologist with a statistical guarantee?" │
│ ➔ Mathematical Tool: Expected-Loss Minimization + Conformal LTT        │
│ ➔ Action Space: 3 Actions (PROCEED vs ACQUIRE vs DEFER)                │
│ ➔ Dataset: MILK10k (10,480 paired clinical + dermoscopy images)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. In-Depth Architectural & Technical Comparison

### 2.1 Dataset & Modality Scope

| Dimension | `D-Guard_System` | `Derma_Guard_Updated` (v2.0) |
|---|---|---|
| **Primary Dataset** | **PAD-UFES-20** (2,298 clinical smartphone images, 1,373 patients, 6 classes: BCC, ACK, NEV, SEK, SCC, MEL). | **MILK10k** (5,240 lesions, 10,480 paired images, 11 classes, ~96% biopsy confirmed). |
| **Visual Modalities** | Single visual scale: Smartphone clinical macro photos. | **Dual visual scales:** Paired **Clinical Photography** + **Contact Dermoscopy** for every lesion. |
| **Auxiliary Modalities** | Tabular clinical metadata (age, sex, itch, bleed) + Free-text clinical narrative (BERT embeddings). | Tabular clinical metadata (age, sex, anatomic site, 6-level skin tone) + **MONET semantic concepts**. |
| **Out-of-Distribution** | Evaluation restricted to PAD-UFES-20 patient splits. | Systematic OOD evaluation on **HAM10000**, **Fitzpatrick17k**, and PAD-UFES-20 as shift benchmarks. |

---

### 2.2 Fusion & Reasoning Engine

#### `D-Guard_System`: Evidential Dempster–Shafer Theory
- Models each modality via an **Evidential Deep Learning head** outputting Dirichlet parameters $\boldsymbol{\alpha}_m$.
- Computes per-modality belief mass $b_m$, vacuous ignorance mass $u_m = K / S_m$, and fuses them via Dempster’s rule of combination:
  $$m_{1 \oplus 2}(A) = \frac{1}{1 - C} \sum_{X \cap Y = A} m_1(X) m_2(Y)$$
- Detects modality conflict mass $C$. When modalities contradict, conflict increases and forces deferral.
- Missing modalities are represented by the **vacuous opinion** ($u = 1$), acting as the algebraic identity element.

#### `Derma_Guard_Updated`: Multi-Scale Vision Foundation + Cross-Modal Consistency
- Employs pre-trained visual foundation backbones (`timm` ConvNeXt / EfficientNet / BiomedCLIP) trained with **modality dropout**.
- Computes **Cross-Modal Consistency** between clinical close-up and contact dermoscopy via:
  - Jensen-Shannon Divergence $D_{\text{JS}}(p_{\text{clinical}} \parallel p_{\text{dermo}})$
  - Representation cosine similarity $S_{\text{rep}} = \frac{\mathbf{z}_c \cdot \mathbf{z}_d}{\|\mathbf{z}_c\|\|\mathbf{z}_d\|}$
  - Mutual Information $\text{MI}$ across a 3-seed epistemic ensemble.
- Aggregates all signals into a structured, versioned **Evidence Feature Record** rather than merging Dirichlet masses.

---

### 2.3 Decision Space & Action Policy

```
D-Guard_System (Binary Policy):
Evidence ────────► [Evidential Head] ───► Decision ──┬──► PREDICT (Autonomous Output)
                                                     └──► DEFER   (Hand to Doctor)

Derma_Guard_Updated (3-Action Policy):
Evidence ────────► [Expected-Loss + LTT] ───► Action ┬──► PROCEED  (Safe Assessment)
                                                     ├──► ACQUIRE  (Request Dermoscopy)
                                                     └──► DEFER    (Escalate to Specialist)
```

1. **`D-Guard_System` (Binary Abstention):**
   - The system is purely reactive to the evidence present at intake.
   - If the smartphone photo or text is untrustworthy, its only fallback is to **DEFER**.
   - Cannot request additional clinical tests.

2. **`Derma_Guard_Updated` (Active Acquisition + Risk Control):**
   - Introduces **Active Feature Acquisition**: if a smartphone photo is ambiguous, it predicts the expected diagnostic benefit of ordering a dermoscopy image.
   - **Cost-Aware Optimization:** Solves Bayes expected-loss minimization across asymmetric clinical penalties:
     $$a^*(S) = \arg\min \{ R_{\text{proceed}}(S),\, R_{\text{acquire}}(S, \text{dermo}),\, R_{\text{defer}} \}$$
   - Allows up to $T_{\max}$ acquisition iterations before finalizing or deferring.

---

### 2.4 Statistical Safety Guarantees

| Safety Layer | `D-Guard_System` | `Derma_Guard_Updated` (v2.0) |
|---|---|---|
| **Abstention Criterion** | Evidential ignorance $u > \tau_u$ and conflict mass $C > \tau_C$. | **Conformal Prediction Sets** + **Learn-Then-Test (LTT)**. |
| **Statistical Bound** | Conformal risk control proposed on classification error. | **Finite-sample theorem** bounding the **malignancy miss rate** on auto-proceeded cases: $\mathbb{P}(\mathcal{R}_{\text{proceed}} \le \alpha) \ge 1 - \delta$. |
| **Safety Override** | Heuristic abstention ladder. | **Mixed-Set Rule:** If conformal prediction set $C(x)$ contains *both* benign and malignant classes, `PROCEED` is strictly prohibited. |
| **Fairness Guarantee** | Marginal calibration across dataset. | **Group-Conditional Conformal Calibration** across 6 skin-tone groups. |

---

### 2.5 Stress-Testing & Robustness Verification

- **`D-Guard_System`:**
  - Audited and corrected severe data leakage in previous benchmarks (demonstrating honest linear probe baselines on disjoint patient splits).
  - Evaluates standard risk-coverage curves (AURC) on clean PAD-UFES-20 data.

- **`Derma_Guard_Updated`:**
  - Establishes the **DERMA-STRESS Protocol**: 7 standardized stress families (Blur/JPEG degradation, Modality Missingness, Contradictory Metadata, Distribution Shift, Leave-Class OOD, Combined Failures, Operational Downtime).
  - Formally measures **Guarantee Violation Rate** ($\Delta_{\text{violation}} = \max(0, \hat{\mathcal{R}}_{\text{test}} - \alpha)$) when test conditions break exchangeability.
  - Replaces flawed classic AURC with **AUGRC** (Traub et al., NeurIPS 2024).

---

### 2.6 Software Architecture & Deliverables

| Layer | `D-Guard_System` | `Derma_Guard_Updated` |
|---|---|---|
| **Project Structure** | Python research codebase (`scripts/`, `src/dguard/`, PyTorch models). | **Full Production Monolith:** Next.js UI, FastAPI REST API, Celery/Redis queue, PostgreSQL + `pgvector`. |
| **Auditability** | File-based experiment scripts. | **Tamper-evident append-only database:** `audit_events`, `model_runs`, `evidence_snapshots`. |
| **Paper Integration** | Internal research documentation. | **Complete IEEE-formatted Overleaf research paper** with pre-registered hypotheses and baseline tables. |

---

## 3. High-Level Summary Table

| Feature / Dimension | `D-Guard_System` | `Derma_Guard_Updated` |
|---|---|---|
| **Primary Mission** | Multimodal evidence weighting & deferral on smartphone images. | Cost-aware active dermoscopy acquisition & risk-bounded deferral. |
| **Benchmark Cohort** | PAD-UFES-20 (Brazil) | MILK10k (Europe/Multicenter) + Derm7pt + PAD-UFES-20 |
| **Key Mathematical Model** | Evidential Deep Learning (Dirichlet) + Dempster–Shafer | Expected Loss Minimization + Learn-Then-Test (LTT) |
| **Action Outputs** | 2 (`PREDICT` or `DEFER`) | 3 (`PROCEED`, `ACQUIRE`, or `DEFER`) |
| **Dermoscopy Acquisition** | ❌ Not modelled (Single-scale) | ✅ Formally modelled with out-of-fold value estimator |
| **Guaranteed Endpoint** | Selective error rate | Malignancy miss rate $\le 5\%$ with $90\%$ confidence |
| **Stress Benchmark** | Split validation audit | DERMA-STRESS (7 failure families A–G) |
| **System Delivery** | Research Python package | Enterprise Web Platform + API + IEEE Paper |

---

## 4. How They Complement Each Other

The two repositories form an evolutionary research continuum:
1. `D-Guard_System` demonstrated that **naive multimodal concatenation degrades performance** on clinical dermatology data, proving that evidence must be discounted when uninformative or contradictory.
2. `Derma_Guard_Updated` builds directly on this insight, elevating the problem from *passive evidence filtering* to *active, cost-aware acquisition of gold-standard dermoscopy*, backed by finite-sample statistical guarantees and systematic failure-mode stress testing.
