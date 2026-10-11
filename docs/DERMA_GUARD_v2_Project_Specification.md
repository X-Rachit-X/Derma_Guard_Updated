 DERMA-GUARD — Project Specification v2.0

**Project type:** Three-member academic research and software-engineering project
**Working title:** *DERMA-GUARD: Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for Multimodal Dermatology AI*
**Version:** 2.0 — Revised after literature check and methodology review
**Supersedes:** v1.0 ("Evidence-Adaptive Dermatology Intelligence Framework")
**Status:** Proposed implementation and evaluation plan. Research outcomes and novelty are not yet established.

---

## 0. What changed from v1.0 (read this first)

| # | Change | Why |
|---|--------|-----|
| 1 | **Primary dataset changed from Derm7pt to MILK10k** | MILK10k has paired clinical close-up + dermoscopic images for 5,240 lesions (≈5× Derm7pt), ~96% histopathology-confirmed, with skin-tone metadata and MONET concept annotations. Every lesion actually has the dermoscopic image, so "acquire dermoscopy" becomes a **real** experiment instead of a simulation. |
| 2 | **Research question narrowed** to "When should the AI ask for dermoscopy, and when should it defer?" | One sharp question is easier to defend, evaluate and present than seven loosely coupled signals. |
| 3 | **Novelty repositioned against HOT (Mehta et al.)** | HOT already recommends dermoscopy when a clinical image is insufficient. "AI requests more evidence" is therefore **not** new. Our differentiation is: (a) statistical risk guarantees, (b) explicit acquisition costs, (c) combined-failure stress testing. |
| 4 | **Hand-tuned policy replaced by an expected-loss (Bayes) decision rule** | Principled, explainable, and has a clear mathematical form for the paper. |
| 5 | **Thresholds calibrated with Learn-then-Test / conformal risk control** | Gives a finite-sample guarantee such as "malignancy miss rate on auto-proceeded cases ≤ α with probability ≥ 1 − δ". This is the strongest differentiator. |
| 6 | **Conformal prediction sets drive actions** | "If the set contains both benign and malignant classes → do not proceed." Intuitive for viva and demo. |
| 7 | **AUGRC replaces AURC as the main multi-threshold metric** | AUGRC (Traub et al., NeurIPS 2024) fixes known flaws of AURC and is interpretable as the average risk of undetected failures. |
| 8 | **New baselines:** learning-to-defer, greedy information-gain acquisition, random and oracle acquisition | Required to show the policy is better than obvious alternatives, and to bound it from above and below. |
| 9 | **"MSE" renamed to MSES** (Minimum Sufficient Evidence Set) | Avoids collision with mean squared error. |
| 10 | **Member 2 now owns the cross-modal consistency signal** | In v1.0 the RAG subsystem never influenced the decision, so it would look decorative. RAG is kept as a small explanation feature. |
| 11 | **Stack simplified** | pgvector instead of Qdrant; local/MinIO storage; one worker queue; no OpenTelemetry/Grafana initially. Time goes to evaluation. |
| 12 | **Fairness analysis added** | MILK10k provides a 6-level skin-tone field; group-conditional conformal calibration is a cheap, meaningful addition. |

---

## 1. Executive summary

DERMA-GUARD is a multimodal dermatology research platform that studies a narrow, clinically motivated decision:

> Given a clinical close-up photograph of a skin lesion, should an AI system **answer now**, **ask for a dermoscopic image first**, or **defer to a human** — and can it make that choice with a **statistical guarantee** on how often it misses a malignancy?

The system computes calibrated class probabilities, uncertainty, cross-modal consistency, input quality and an out-of-distribution (OOD) score. A decision engine then chooses among three actions:

1. **PROCEED** — return a bounded assessment with its limitations.
2. **ACQUIRE** — request the dermoscopic image (or another available input) because its expected benefit exceeds its cost.
3. **DEFER** — abstain and route to human review.

The action rule is an **expected-loss minimisation** over a clinically asymmetric cost matrix. Its thresholds are calibrated using **Learn-then-Test (LTT) / conformal risk control**, so that the malignancy miss rate on auto-proceeded cases is bounded with high probability under exchangeability.

**DERMA-STRESS** is a reproducible protocol that tests the policy under image degradation, missing modalities, contradictory metadata, distribution shift and **combined** failures — including measuring **when the statistical guarantee breaks** under shift, which is itself a reportable finding.

The integrated web application provides case intake, evidence storage, model execution, assessment history, review workflows, experiment tracking and audit logs.

**Central research question:**
Can a cost-aware, risk-controlled acquire/defer policy achieve a better reliability–coverage–cost trade-off than confidence-threshold abstention, learning-to-defer, and HOT-style "recommend dermoscopy when uncertain" rules — on clean data and under controlled evidence failures?

This is a research prototype, not a medical device. The proposed method may not beat its baselines; results will be reported honestly.

**One-line pitch:**
*"An AI that knows when to answer, when to ask for dermoscopy, and when to hand over — with a statistical guarantee on how often it is wrong."*

---

## 2. Scope and non-goals

### 2.1 In scope

- Patient-case-centred web application (intake → assessment → review → timeline).
- Clinical-only, dermoscopy-only and fused image classifiers on MILK10k, trained with modality dropout.
- Probability calibration (temperature scaling) and uncertainty (small deep ensemble).
- Conformal prediction sets and LTT-calibrated decision thresholds.
- Expected-loss proceed/acquire/defer policy with an explicit cost model.
- Acquisition-value estimator (predicts benefit of obtaining dermoscopy).
- Cross-modal consistency signal and contradiction detection.
- OOD scoring (energy / Mahalanobis) with leave-class-out and external-dataset tests.
- DERMA-STRESS protocol and experiment runner.
- Baseline comparisons, ablations, subgroup (skin-tone) analysis.
- Small RAG-based explanation feature with source attribution.
- Persistent case history, model-run records, review actions and audit events.

### 2.2 Out of scope (initial release)

- Training a foundation model from scratch.
- Claiming the system replaces dermatologists, or claiming clinical benefit.
- Autonomous diagnosis or treatment recommendations.
- Microservices, Kafka, Kubernetes, OpenTelemetry/Grafana dashboards.
- Reinforcement-learning acquisition policies (listed as future work; too costly to tune reliably in 12 weeks).
- LLM authority over predictions, thresholds or routing.
- Using synthetic text as if it were genuine patient history.

### 2.3 Success philosophy

A working platform and a confirmed hypothesis are separate outcomes. If the policy does not beat baselines on the pre-registered endpoints, the team reports that and emphasises the benchmark, the guarantee-violation analysis under shift, and the ablations.

---

## 3. Problem statement and motivation

Clinical close-up photographs are cheap and available from any smartphone. Dermoscopy needs a device and a trained operator. A recent reader study on MILK10k images (Müller et al., JAAD 2026) found that the two modalities carry **complementary** information: dermoscopy alone gave higher sensitivity for malignant lesions, while clinical close-ups gave higher specificity, and each added independent diagnostic value.

This creates a natural decision problem for any screening system:

- Answering every case from the clinical image alone risks missing malignancies.
- Requiring dermoscopy for every case wastes a scarce resource.
- Deferring every uncertain case overloads human reviewers.

Existing work addresses parts of this. HOT (Mehta et al.) recommends dermoscopy when the clinical image is insufficient; selective classification literature studies abstention; active feature acquisition studies cost-aware test-time acquisition; conformal prediction gives distribution-free uncertainty. **What is not established in the dermatology literature we have checked so far** is a policy that (a) jointly decides proceed/acquire/defer under an explicit cost model, (b) carries a finite-sample risk guarantee, and (c) is stress-tested under combined evidence failures with the guarantee's breakdown measured. Whether this combination is publishably novel must be confirmed by a full related-work audit (Section 6.4).

---

## 4. Research objectives and hypotheses

### Objective 1 — Risk-controlled decision engine (EADE)

Build a deterministic, versioned policy that minimises expected clinical loss across proceed/acquire/defer, with thresholds calibrated by Learn-then-Test.

**H1 (risk control):** On the clean MILK10k test split, the EADE achieves the target malignancy miss rate α on proceeded cases (within statistical tolerance) at **higher coverage** than confidence-threshold abstention calibrated to the same α.

### Objective 2 — Cost-aware acquisition

Estimate, per case, the benefit of acquiring dermoscopy, and acquire only when benefit exceeds cost.

**H2 (acquisition efficiency):** At a matched acquisition rate, the EADE recovers a larger fraction of the oracle acquisition benefit than random acquisition, entropy-threshold acquisition (HOT-style), and greedy information-gain acquisition.

### Objective 3 — DERMA-STRESS robustness

**H3 (robustness):** Under combined evidence failures, the EADE's selective risk degrades less than the baselines', and its risk guarantee degrades gracefully (measured violation) rather than silently.

### Engineering objective

An auditable platform that links every assessment to its exact evidence snapshot, model versions, calibration version, cost-matrix version and policy version. This is an acceptance objective, not a novelty claim.

### Hypothesis discipline

- Endpoints, α, δ, cost matrix and acquisition costs are fixed **before** the final test run and recorded in a pre-registration file in the repository.
- Thresholds are tuned on validation/calibration splits only.
- The test split is touched once per frozen configuration.

---

## 5. Proposed research contributions

### Contribution A — Risk-controlled, cost-aware decision engine

An expected-loss decision rule over three actions, with a conformal / Learn-then-Test calibration layer that provides a finite-sample bound on malignancy miss rate among proceeded cases. Full formulation in Section 7.

### Contribution B — DERMA-STRESS

A versioned, reproducible stress protocol covering image degradation, missing modalities, contradictory metadata, distribution shift, leave-class-out OOD, combined failures and operational failures — evaluated not only on accuracy but on **guarantee violation**, acquisition behaviour and cost.

### Contribution C — Acquisition-value analysis

A learned estimator of the benefit of acquiring dermoscopy, evaluated against **oracle** (upper bound) and **random** (lower bound) acquisition at matched rates, plus two project-defined diagnostic metrics:

- **MSES — Minimum Sufficient Evidence Set** (renamed from "MSE").
- **NIR — Negative Information Rate.**

Both are kept only if they add information beyond simpler statistics.

---

## 6. Literature positioning

### 6.1 Verification status

Each paper below is marked:
- ✅ **Verified** — existence and key claim checked during this revision.
- 🔍 **To verify** — cited in v1.0 but not yet checked; must be confirmed before it appears in the paper.

### 6.2 Primary comparison and threat papers

| Paper | Status | Role | What we must do |
|-------|--------|------|-----------------|
| **HOT** — Mehta et al., *Revamping AI Models in Dermatology* (arXiv 2311.01009) | ✅ | **Closest threat.** Hierarchical prediction + OOD alert + recommends dermoscopy when clinical image is insufficient, then fuses both. | Implement a HOT-style baseline (entropy/uncertainty-triggered dermoscopy request). State clearly what we add: cost model, risk guarantee, combined-failure stress tests. |
| **Traub et al.**, *Overcoming Common Flaws in the Evaluation of Selective Classification Systems* (NeurIPS 2024) | ✅ | Source of AUGRC metric. | Use AUGRC as primary multi-threshold metric. |
| **JI-ADF** — Joint-Individual Learning with Adaptive Decision Fusion (arXiv 2604.27343) | ✅ | Recent multimodal fusion on MILK10k. | Use as a fusion reference point on the same dataset (label as paper-reported unless reimplemented). |
| **Müller et al.**, complementary value of dermatoscopy and clinical close-ups, JAAD 2026 | ✅ | Clinical motivation for the acquisition question. | Cite in introduction. |
| **Fayyad et al.**, *Empirical Validation of Conformal Prediction for Trustworthy Skin Lesions Classification* (arXiv 2312.07460) | ✅ | Prior conformal work in dermatology. | Position: they evaluate CP as UQ; we use CP/LTT to drive actions. |
| **Lu et al.**, *Fair Conformal Predictors for Applications in Medical Imaging* (arXiv 2109.04392) | ✅ | Group-conditional conformal by skin type. | Basis for our skin-tone subgroup calibration. |
| **Zoravar et al.**, *Conformal Ensemble of Vision Transformers* (arXiv 2505.15997) | ✅ | Conformal under domain shift in dermatology. | Compare shift behaviour. |
| **CAMA** — Rheude et al., *Cohort-Based Active Modality Acquisition* (arXiv 2505.16791) | ✅ | Test-time modality acquisition under budget. | Related-work positioning; possible cohort-budget variant. |
| **ACO** — Valancius et al., *Acquisition Conditioned Oracle for Nongreedy AFA* (arXiv 2302.13960) | ✅ | Active feature acquisition methodology. | Related work; motivates oracle/greedy baselines. |
| **Shim et al.**, *Joint AFA and classification with variable-size set encoding* (NeurIPS 2018) | ✅ | RL-based cost-sensitive AFA. | Related work; RL is future work for us. |
| **SAGEAgent** (arXiv 2607.09521) | ✅ | Cost-aware sequential modality acquisition with LLM agent, non-dermatology. | Related work; shows the area is active. |
| DermFormer (2025) | 🔍 | Multimodal ViT for skin cancer with robustness. | Verify before citing. |
| Zuo et al. / CosCatNet (2025) | 🔍 | Adaptive multimodal fusion. | Verify before citing. |
| QA-MoE (IJCAI 2026) | 🔍 | Quality-aware routing, non-dermatology. | Verify before citing. |
| SAGE (npj Digital Medicine 2026) | 🔍 | Multi-criterion uncertainty in skin lesions. | Verify before citing. Note name clash with SAGEAgent. |
| PanDerm (dermatology foundation model) | 🔍 | Possible stronger backbone. | Verify availability, licence, compute before use. |

### 6.3 Methodological foundations (well established)

- Selective classification — Geifman & El-Yaniv (2017).
- Calibration / temperature scaling — Guo et al. (2017).
- Deep ensembles — Lakshminarayanan et al. (2017).
- Conformal prediction intro — Angelopoulos & Bates (2021).
- RAPS / APS prediction sets — Angelopoulos et al. (2020).
- Learn-then-Test — Angelopoulos et al. (2021).
- Conformal risk control — Angelopoulos et al. (2022).
- Learning to defer — Mozannar & Sontag (2020).
- Energy-based OOD — Liu et al. (2020); Mahalanobis OOD — Lee et al. (2018).
- Decision curve analysis — Vickers & Elkin (2006).

### 6.4 Novelty audit procedure (Week 1–2)

1. Search Google Scholar / arXiv / PubMed / MICCAI / ISBI for: *dermoscopy acquisition*, *selective prediction dermatology*, *conformal dermatology deferral*, *active modality acquisition skin*, *learning to defer dermatology*.
2. Read every paper citing HOT.
3. Record each hit in `docs/related_work_audit.md` with: task, data, actions, guarantees, costs, stress tests.
4. If a paper already does all three of (cost model + risk guarantee + acquire/defer in dermatology), narrow the claim to the stress-test and guarantee-violation analysis.

### 6.5 Novelty wording

Use: *"We propose and evaluate a risk-controlled, cost-aware proceed/acquire/defer policy for multimodal dermatology and measure its behaviour under controlled evidence failures."*
Do **not** use "first", "state of the art" or "novel" without audit evidence.

### 6.6 Result labelling

Every number in a table is labelled `paper-reported`, `our-reimplementation`, or `our-method`.

---

## 7. Mathematical formulation

### 7.1 Notation

| Symbol | Meaning |
|--------|---------|
| $x^{c}$ | clinical close-up image |
| $x^{d}$ | dermoscopic image |
| $z$ | structured metadata (age bin, sex, site, skin tone) |
| $S \subseteq \{c, d, z\}$ | currently available evidence |
| $y \in \{1,\dots,K\}$ | diagnostic class |
| $g(y) \in \{\text{benign}, \text{malignant}\}$ | malignancy grouping |
| $p_S(y \mid x_S)$ | calibrated posterior using evidence $S$ |
| $L(y, \hat y)$ | loss of predicting $\hat y$ when truth is $y$ |
| $c_{\text{acq}}(m)$ | cost of acquiring modality $m$ |
| $c_{\text{def}}$ | cost of deferring to a human |

### 7.2 Cost matrix

Asymmetric, clinically motivated, fixed in the pre-registration file:

| Truth \ Prediction | correct class | wrong class, same malignancy group | benign predicted for malignant | malignant predicted for benign |
|---|---|---|---|---|
| Loss | 0 | $\ell_{\text{sub}}$ (e.g. 1) | $\ell_{\text{miss}}$ (e.g. 10) | $\ell_{\text{fa}}$ (e.g. 2) |

Sensitivity analysis over $\ell_{\text{miss}} \in \{5, 10, 20\}$ is mandatory, because results depend on these values.

### 7.3 Expected loss of each action

**Proceed** with the Bayes-optimal prediction:

$$
R_{\text{proceed}}(S) = \min_{\hat y} \sum_{y} p_S(y \mid x_S)\, L(y, \hat y)
$$

**Acquire** modality $m \notin S$:

$$
R_{\text{acquire}}(S, m) = \hat{\mathbb{E}}\left[R_{\text{proceed}}(S \cup \{m\})\right] + c_{\text{acq}}(m)
$$

where the expectation is estimated by the **acquisition-value estimator** $h_m(x_S)$ (Section 7.4).

**Defer:**

$$
R_{\text{defer}} = c_{\text{def}}
$$

**Policy:**

$$
a^*(S) = \arg\min \{\, R_{\text{proceed}}(S),\ \min_{m} R_{\text{acquire}}(S,m),\ R_{\text{defer}} \,\}
$$

subject to: at most $T_{\max}$ acquisition rounds, only modalities that are actually available, and the safety layer (7.6) overriding PROCEED when its condition fails.

### 7.4 Acquisition-value estimator

For training lesions where both $x^c$ and $x^d$ exist:

1. Produce **out-of-fold** predictions for $p_{\{c,z\}}$ and $p_{\{c,d,z\}}$ (5-fold, to avoid optimistic targets).
2. Target: $t_i = R_{\text{proceed}}(\{c,z\})_i - R_{\text{proceed}}(\{c,d,z\})_i$ — the realised reduction in expected loss from adding dermoscopy.
3. Fit $h_d$ (gradient-boosted trees or a small MLP) on features available **before** acquisition: clinical-model posterior, entropy, ensemble disagreement, quality scores, OOD score, metadata.
4. Then $\hat{\mathbb{E}}[R_{\text{proceed}}(S \cup \{d\})] = R_{\text{proceed}}(S) - h_d(x_S)$.

Simpler fallback baselines: acquire iff entropy > τ (HOT-style); acquire iff mutual information (ensemble) > τ.

### 7.5 Calibration and uncertainty

- **Temperature scaling** per modality configuration, fit on the validation split.
- **Deep ensemble** of $M = 3$ seeds (cost-limited) for epistemic uncertainty:
  - Predictive entropy $H[\bar p]$
  - Mutual information $\text{MI} = H[\bar p] - \frac{1}{M}\sum_j H[p_j]$
- **Consistency signal** (owned by Member 2): Jensen–Shannon divergence between modality-specific posteriors, e.g. $\text{JS}(p_{\{c\}} \,\|\, p_{\{z\}})$, plus rule-based contradiction flags between metadata and extracted text.
- **OOD score:** energy score $E(x) = -T \log \sum_k e^{f_k(x)/T}$ and/or Mahalanobis distance on penultimate features.

### 7.6 Safety layer: conformal sets and Learn-then-Test

**Conformal prediction sets.** Using the calibration split, build APS or RAPS sets $C_\alpha(x_S)$ with marginal coverage $\ge 1-\alpha$. Rule:

> If $C_\alpha(x_S)$ contains at least one benign **and** at least one malignant class, PROCEED is not allowed.

**Learn-then-Test threshold calibration.** Let $\lambda$ parameterise how permissive PROCEED is (e.g. a cap on $R_{\text{proceed}}$). Define the controlled risk:

$$
\mathcal{R}(\lambda) = \Pr\big(g(y)=\text{malignant},\ g(\hat y)=\text{benign},\ a^*=\text{PROCEED} \big)
$$

or its conditional version on proceeded cases. For a grid $\lambda_1 < \dots < \lambda_J$ ordered from conservative to permissive:

1. Compute the empirical risk $\hat{\mathcal{R}}(\lambda_j)$ on the calibration split.
2. Compute a valid p-value for $H_j: \mathcal{R}(\lambda_j) > \alpha$ (binomial tail or Hoeffding–Bentkus bound).
3. Apply **fixed-sequence testing**: walk the grid while $p_j \le \delta$; select the most permissive rejected $\lambda$.

Result: with probability $\ge 1-\delta$ over the calibration draw, $\mathcal{R}(\hat\lambda) \le \alpha$.

**Important limitation:** this guarantee assumes calibration and test data are exchangeable. It **does not** hold under distribution shift. DERMA-STRESS measures the size of the violation under each shift; that measurement is a core result, not a footnote.

**Fairness variant:** repeat calibration per skin-tone group (group-conditional conformal) and report coverage and miss rate per group. Merge sparse groups and report sample sizes.

### 7.7 Project-defined metrics

**MSES — Minimum Sufficient Evidence Set.** For case $i$ with available evidence $E_i$, the smallest-cost subset $S \subseteq E_i$ whose decision satisfies the pre-registered criterion (e.g. PROCEED with correct malignancy group). Report "none" if no subset qualifies. Compare against simple baselines ("clinical-only sufficient rate").

**NIR — Negative Information Rate.**

$$
\text{NIR}_m = \frac{1}{N}\sum_i \mathbb{1}\left[ L(\hat y_i^{S \cup m}, y_i) > L(\hat y_i^{S}, y_i) \right]
$$

Report together with the mean magnitude of the increase. Keep only if informative.

**Acquisition efficiency (new).** At a matched acquisition rate $r$:

$$
\text{AE}(r) = \frac{G_{\text{policy}}(r) - G_{\text{random}}(r)}{G_{\text{oracle}}(r) - G_{\text{random}}(r)}
$$

where $G$ is the total loss reduction from acquisitions, the oracle acquires exactly the $rN$ cases with largest realised benefit, and random acquires $rN$ cases uniformly. $\text{AE} = 1$ means oracle-level, $0$ means no better than random.

---

## 8. Datasets

### 8.1 Primary: MILK10k (ISIC)

- 5,240 lesions, each with one clinical close-up and one dermoscopic image (10,480 images), five centres.
- ~96% histopathology-confirmed.
- Metadata: age (5-year bins), sex, anatomic site, skin tone (6 levels, deliberately not Fitzpatrick), MONET concept probabilities.
- 11 broad diagnostic categories. Several are very small (e.g. "other malignant" ≈ 7 training lesions).
- A separate **MILK10k Benchmark** (479 lesions) exists as a live challenge with hidden labels.
- Licence and data-use terms: **verify before download and before any redistribution.**

**Handling decisions:**
- **Label scheme:** report both (a) a coarse grouping (merge rare classes into "other benign"/"other malignant") and (b) binary malignant/benign for the guarantee.
- **Patient leakage:** MILK10k lists no patient identifiers. Split **by lesion**, deduplicate via the provided ISIC IDs, and state the residual leakage risk in the limitations.
- **Splits (lesion-level, stratified):** 60% train / 15% validation / 10% calibration / 15% test. Repeat with 3 seeds where compute allows.
- **Optional:** submit final predictions to the MILK10k Benchmark challenge for an external hidden-test number (check challenge rules first).

### 8.2 Secondary: Derm7pt

Clinical + dermoscopic + 7-point checklist. Use as a second paired-modality dataset for replication if time allows. Verify licence and class mapping.

### 8.3 Shift dataset: PAD-UFES-20

Smartphone clinical images from Brazil, no dermoscopy. Use **only** for clinical-image distribution-shift tests, with an explicit label mapping to the shared classes. Do not treat as multimodal with dermoscopy.

### 8.4 Optional: HAM10000 / ISIC 2019

Dermoscopy-only. Use for dermoscopy-branch pretraining or shift tests only.

### 8.5 Leakage rules

- Calibration, temperature, conformal and LTT thresholds use **only** validation/calibration splits.
- Acquisition-value targets use **out-of-fold** predictions.
- Synthetic text (if any) is generated from metadata only and must never encode the diagnosis.
- Test split frozen until the pre-registration file is committed.

### 8.6 Statistical power note

A 15% test split is ~786 lesions, with perhaps 50–60 melanomas. Confidence intervals on per-class melanoma metrics will be wide. Report bootstrap CIs, prefer binary malignant/benign for the guarantee, and avoid claims about rare classes.

---

## 9. DERMA-STRESS protocol

Each family has fixed severities, seeds and config files.

### A — Image degradation
Gaussian blur, JPEG compression, brightness/contrast shift, downscaling, motion blur. 3 severity levels each. Applied separately to clinical and dermoscopic images.

### B — Missing evidence
Remove dermoscopy, remove metadata fields, remove both. Distinguish `MISSING` from `PRESENT_BUT_LOW_QUALITY`.

### C — Contradictory context
Controlled perturbations: swap anatomic site, shift age bin, insert conflicting symptom text. All labelled synthetic; never leak diagnosis.

### D — Distribution shift
- Clinical branch on PAD-UFES-20 (mapped labels).
- **Centre shift** within MILK10k if centre information is available (train on some centres, test on another) — verify field availability.

### E — Leave-class-out OOD (new)
Remove one or more classes (e.g. inflammatory, vascular) from training entirely; measure whether the policy defers on them.

### F — Combined failures
- Degraded clinical image + missing metadata
- Missing dermoscopy + contradictory metadata
- Distribution shift + degradation
- Unseen class + degradation

### G — Operational failures
Inference timeout, worker restart, corrupted upload, LLM provider unavailable. Reported separately from diagnostic metrics.

### Per-condition outputs
Selective risk, coverage, AUGRC, **guarantee violation** ($\hat{\mathcal R}_{\text{test}} - \alpha$), conformal set size and coverage, acquisition rate, acquisition efficiency, deferral rate, latency.

---

## 10. Experimental design

### 10.1 Primary endpoint (pre-registered)

**Coverage at guaranteed risk:** the fraction of test cases auto-proceeded while the malignancy miss rate on proceeded cases is ≤ α (α = 0.05, δ = 0.1, adjustable only before freezing).

### 10.2 Secondary endpoints

**Classification**
- Balanced accuracy, macro-F1, per-class sensitivity/precision, confusion matrix.
- Binary malignancy AUROC (with DeLong CIs).

**Calibration**
- Brier score, negative log-likelihood.
- Adaptive (equal-mass) ECE and classwise ECE. Avoid plain fixed-bin ECE as the only calibration number.

**Selective prediction**
- Risk–coverage curves.
- **AUGRC** (primary multi-threshold metric); AURC reported for comparability only.
- Selective risk at fixed coverage (e.g. 70%, 80%, 90%).

**Conformal**
- Empirical coverage, mean set size, fraction of mixed benign/malignant sets.
- Per-skin-tone coverage and miss rate.

**Acquisition**
- Acquisition rate, acquisition efficiency AE(r), loss before/after acquisition, unnecessary-request rate, NIR, MSES.

**Clinical utility**
- **Decision-curve net benefit** for malignancy referral: $\text{NB} = \frac{TP}{N} - \frac{FP}{N}\cdot\frac{p_t}{1-p_t}$ over a range of threshold probabilities $p_t$.
- Expected cost per case under the cost matrix.

**Operational**
- Job success rate, recovery after failure, end-to-end latency, traceability completeness.

### 10.3 Baselines

| ID | Baseline | Actions |
|----|----------|---------|
| B1 | Clinical-only classifier | proceed always |
| B2 | Fused clinical + dermoscopy (always acquire) | proceed always |
| B3 | Confidence-threshold abstention (max-softmax), LTT-calibrated to same α | proceed / defer |
| B4 | Learning-to-defer (Mozannar & Sontag surrogate) | proceed / defer |
| B5 | HOT-style: acquire dermoscopy if entropy > τ, else proceed | proceed / acquire |
| B6 | Greedy information-gain acquisition (ensemble MI) + abstention | all three |
| B7 | Random acquisition at matched rate | lower bound |
| B8 | Oracle acquisition at matched rate | upper bound |
| **Ours** | Full EADE (expected-loss + acquisition estimator + LTT safety layer) | all three |

### 10.4 Ablations

- Remove LTT safety layer (raw expected-loss policy).
- Remove conformal mixed-set rule.
- Remove acquisition estimator (use entropy trigger instead).
- Remove consistency signal.
- Remove OOD signal.
- Remove quality signals.
- Uncalibrated vs temperature-scaled posteriors.
- Single model vs 3-seed ensemble.
- Cost-matrix sensitivity ($\ell_{\text{miss}} \in \{5,10,20\}$, $c_{\text{acq}}$ sweep).

### 10.5 Statistics

- Paired lesion-level bootstrap (≥ 2,000 resamples) for all differences between methods.
- McNemar's test for paired accuracy; DeLong for AUROC.
- Report means ± 95% CI over 3 seeds where feasible.
- Holm correction when testing several baselines against our method.
- No cherry-picking: every pre-registered metric is reported.

### 10.6 Interpreting outcomes

- **Positive:** higher coverage at the same guaranteed risk, or higher AE at the same acquisition rate.
- **Partial:** gains only under some stress families — report which and why.
- **Negative:** no gain over B3/B5 — the paper then centres on DERMA-STRESS and the guarantee-violation analysis.

---

## 11. System architecture

Modular monolith + one background worker.

### A. Case intake
Create case, upload clinical/dermoscopic images, structured metadata, optional symptom text. Record provenance, file hash, missingness (`MISSING` vs `PRESENT_BUT_LOW_QUALITY`). Validate type and size.

### B. Vision intelligence (Member 1)
Clinical, dermoscopy and fused models (modality dropout), 3-seed ensemble, temperature scaling, quality metrics (blur via Laplacian variance, exposure, resolution), OOD scores. Returns structured outputs with model and calibration versions.

### C. Consistency and evidence intelligence (Member 2)
- **Consistency module (feeds the decision):** JS divergence between modality-specific posteriors; metadata–text contradiction flags; structured symptom extraction (LLM with schema-constrained output, or rule-based fallback).
- **Explanation module (does not feed the decision):** small curated corpus (e.g. public dermatology guideline summaries; licences checked), BM25 + pgvector hybrid retrieval, claim-to-passage linking, unsupported-claim flags. Clearly labelled as context, not validation.

### D. Evidence Manager
Aggregates all signals into a versioned feature record per decision. Logs unavailable signals and why.

### E. Decision engine (EADE)
Applies cost matrix, acquisition estimator, conformal rule and LTT threshold. Enforces $T_{\max}$ acquisition rounds. Records action, reason codes and all versions.

### F. Persistence and audit
Append-only assessments, model runs, reviews and audit events. No silent overwrites.

### G. DERMA-STRESS runner
Loads frozen split + config, applies stress, runs all baselines and EADE on the same cases, saves predictions and actions, computes metrics and CIs, writes JSON + Markdown reports.

### Data flow (one assessment)

```
Upload → Validate → Store asset → Enqueue job
  → Vision models (calibrated posteriors, ensemble, quality, OOD)
  → Consistency module
  → Evidence Manager (feature record)
  → EADE: expected losses → conformal check → LTT threshold
  → Action: PROCEED | ACQUIRE(dermoscopy) | DEFER
  → Persist assessment + audit event → UI
```

---

## 12. User roles and workflows

### Roles
Researcher/admin, case operator, reviewer, viewer. Backend authorisation enforced even in demo mode.

### Workflow 1 — Assess a case
Create case → upload clinical image → request assessment → background job → action shown with reasons, calibrated probabilities, conformal set, limitations.

### Workflow 2 — Acquire dermoscopy
EADE returns ACQUIRE → UI explains expected benefit vs cost (non-medical wording) → operator uploads dermoscopy or marks unavailable → new assessment on new snapshot → system records change in prediction, risk, action and cost → loop stops at $T_{\max}$ or DEFER.

### Workflow 3 — Human review
Deferred case → reviewer sees evidence, outputs, reasons → records decision and rationale → appended to timeline.

### Workflow 4 — Run DERMA-STRESS
Select split + config → worker runs all methods → metrics, CIs, plots → linked to config hash.

---

## 13. Technology stack (simplified)

| Layer | Choice | Notes |
|-------|--------|-------|
| Frontend | Next.js, React, TypeScript, Tailwind | |
| Backend | Python, FastAPI, Pydantic | |
| Database | PostgreSQL + **pgvector** | Replaces Qdrant |
| Object storage | Local filesystem in dev, MinIO if needed | |
| Jobs | Redis + RQ (or Celery) | One queue, one worker type |
| ML | PyTorch, timm, torchvision, scikit-learn, OpenCV | |
| Conformal / UQ | MAPIE or a small in-house implementation (~150 lines) | In-house is easier to explain |
| Trees | LightGBM or scikit-learn GBM | Acquisition estimator |
| Retrieval | rank_bm25 + pgvector | |
| LLM | Provider abstraction with schema-constrained output, timeout, retry | Optional; rule-based fallback required |
| Experiments | YAML configs + MLflow (or plain JSON logs) | |
| Deployment | Docker Compose, GitHub Actions CI | |
| Logging | Structured JSON logs | No PHI in logs |

**Backbone choice:** start with a pretrained ConvNeXt-Tiny or EfficientNet-B0 from timm. Consider PanDerm or another dermatology foundation model only after verifying availability, licence and compute.

---

## 14. Data model

### PatientCase
`id, created_at, updated_at, status, created_by, access_metadata, permitted_metadata`

### EvidenceAsset
`id, case_id, asset_type (clinical|dermoscopy|metadata|text), storage_key, provenance, uploaded_at, quality_status, quality_scores, content_hash`

### EvidenceSnapshot
`id, case_id, asset_ids[], missingness_record, created_at`

### Assessment
`id, case_id, snapshot_id, policy_version, calibration_version, cost_matrix_version, ltt_threshold, action, reason_codes[], conformal_set, expected_losses{proceed, acquire, defer}, result_summary, created_at`

### ModelRun
`id, assessment_id, model_name, model_version, ensemble_seed, preprocessing_version, temperature, output_scores, entropy, mutual_information, ood_score, runtime_ms, status, error_details`

### ConsistencyRecord
`id, assessment_id, js_divergences{}, contradiction_flags[], extractor_version`

### EvidenceSource
`id, assessment_id, document_id, source_url, passage_reference, retrieval_score, index_version, supported (bool)`

### WorkflowJob
`id, case_id, job_type, status, attempt_count, started_at, finished_at, error_code, error_message`

### ReviewDecision
`id, case_id, assessment_id, reviewer_id, decision, rationale, created_at`

### AuditEvent
`id, case_id, actor_type, actor_id, event_type, event_payload, created_at`

### ExperimentRun
`id, dataset_version, split_version, stress_config, config_hash, seed, model_versions, policy_version, metrics, ci, started_at, finished_at, status`

---

## 15. API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/cases` | Create case |
| GET | `/api/cases` | List authorised cases |
| GET | `/api/cases/{id}` | Case details |
| POST | `/api/cases/{id}/assets` | Upload evidence |
| POST | `/api/cases/{id}/assessments` | Request assessment |
| GET | `/api/assessments/{id}` | Status/result |
| GET | `/api/cases/{id}/timeline` | Event history |
| POST | `/api/cases/{id}/reviews` | Record review |
| GET | `/api/policy/current` | Active policy, calibration and cost-matrix versions |
| POST | `/api/experiments` | Start DERMA-STRESS run |
| GET | `/api/experiments/{id}` | Experiment status/results |

Requirements: typed schemas, auth on every case route, async jobs, idempotency keys on assessment creation, clear error codes, versioned contracts, append-only history.

---

## 16. Frontend pages

1. **Overview** — case counts, job states, recent experiments.
2. **Case list** — filter by status/date.
3. **Case creation** — metadata + uploads.
4. **Case detail** — evidence, quality badges, current action.
5. **Assessment detail** — calibrated probabilities, **conformal set visual**, expected-loss bar for the three actions, reason codes, limitations.
6. **Acquisition/review screen.**
7. **Case timeline.**
8. **DERMA-STRESS lab** — config picker, risk–coverage and AUGRC plots, guarantee-violation chart, acquisition-efficiency chart.
9. **Evidence sources** — retrieved passages, support status.
10. **Admin** — model/policy/calibration versions.

The UI must never imply that a confident score, a heatmap or a citation proves clinical correctness.

---

## 17. Security, privacy and safeguards

- Auth + role-based authorisation on every route.
- Upload validation (type, size, image decode check).
- Secrets in environment variables; never in the repo.
- No patient images or text sent to external LLM providers unless a permitted data-handling arrangement exists; prefer local or rule-based extraction for the demo.
- No sensitive content in logs.
- Separate public, synthetic and any real data.
- Ethics approval required before any real patient data.
- Clear "research prototype, not a medical device" banner.
- Subgroup (skin-tone) results reported, including where the guarantee fails for a group.

---

## 18. Testing

### Unit
Schema validation, quality functions, missingness, cost-matrix loss, expected-loss computation, conformal set construction, LTT threshold selection (test against a hand-computed toy example), metric functions (AUGRC, AE, NIR), authorisation.

### Integration
Upload → model run; model failure → visible failed job; assessment → timeline; review → immutable history; experiment config → saved metrics.

### Research
- Lesion-level split has no ID overlap.
- Calibration/test disjoint.
- Out-of-fold targets for acquisition estimator.
- Same cases across all baselines.
- Stress configs reproducible from seed.
- **Guarantee sanity test:** on synthetic exchangeable data, empirical risk ≤ α in ≥ (1 − δ) of repeated calibration draws.

### End-to-end acceptance
1. Create case, upload clinical image, get an action.
2. ACQUIRE path: upload dermoscopy, action updates, history preserved.
3. Invalid file → clear error.
4. Inference timeout → case remains inspectable.
5. Stress run → metrics and config retrievable.
6. Review recorded without altering prior assessment.

---

## 19. Three-member responsibilities

### Member 1 — ML, calibration and decision engine
- MILK10k preparation, label grouping, lesion-level splits.
- Clinical, dermoscopy and fused models; ensemble; temperature scaling.
- Conformal sets, LTT calibration, expected-loss policy, acquisition estimator.
- Baselines B1–B8 and ablations.
- **Deliverable:** reproducible Python package + result tables.

### Member 2 — Consistency, stress testing and evidence intelligence
- Consistency module (JS divergence, contradiction flags, symptom extraction).
- DERMA-STRESS families A–G: generators, configs, runner logic.
- Fairness / skin-tone subgroup analysis.
- Small RAG explanation module with source attribution.
- Related-work audit document.
- **Deliverable:** DERMA-STRESS package + consistency module + audit.

### Member 3 — Platform and systems
- Next.js UI, FastAPI, PostgreSQL/pgvector, object storage.
- Jobs, workflow states, audit, auth.
- Experiment lab UI and plots.
- Docker, CI, integration and E2E tests, deployment.
- **Deliverable:** deployable application.

### Shared
Pre-registration file, code review, integration milestones, paper writing, demo rehearsal. Every member must be able to explain the full pipeline and the guarantee in plain words.

---

## 20. Twelve-week plan

| Week | Member 1 | Member 2 | Member 3 | Shared milestone |
|------|----------|----------|----------|------------------|
| 1 | Download MILK10k, check licence, EDA, label grouping | Novelty audit starts; stress family A design | Repo, Docker, CI, FastAPI + Next.js skeleton | — |
| 2 | Lesion-level splits; clinical-only baseline | Novelty audit done; family A + B generators | Case + asset upload, DB schema | **M1: vertical slice** |
| 3 | Dermoscopy and fused models (modality dropout) | Consistency module v1 | Background jobs, model-run records | — |
| 4 | Ensemble + temperature scaling; calibration metrics | Families C + D | Assessment page, timeline | **M2: trustworthy baseline** |
| 5 | Conformal sets; baselines B1–B3 | Family E (leave-class-out) | Review workflow, auth | — |
| 6 | Acquisition estimator (out-of-fold); B5–B8 | Family F combined; runner | Acquisition UI flow | **M3: fair comparison** |
| 7 | Expected-loss policy + LTT layer | Fairness analysis tooling | Experiment lab UI | **Pre-registration frozen** |
| 8 | B4 learning-to-defer; ablations | Family G operational; RAG explanation | Plots: AUGRC, risk–coverage, AE | — |
| 9 | Full runs on validation; sensitivity analysis | Full stress runs | Integration hardening | **M4: policy value check** |
| 10 | **Single test-split run** | Stress results + subgroup tables | E2E tests, failure recovery | — |
| 11 | Paper results section | Paper related work + stress section | Deployment, docs | **M5: integrated system** |
| 12 | Final report, slides, demo rehearsal (all) | | | **Submission** |

Rule: evaluation quality before features. Anything unstable in Week 11 is cut.

---

## 21. Milestones and go/no-go

| Milestone | Go criterion | If no-go |
|-----------|--------------|----------|
| M1 Vertical slice | Case created, image uploaded, inference run, result persisted, failure shown | No new models or agents until it works |
| M2 Trustworthy baseline | Splits verified leak-free; baseline metrics reproducible across seeds | Fix data issues first |
| M3 Fair comparison | B1–B3, B5, B7, B8 run on identical cases | Simplify policy |
| Pre-registration | α, δ, cost matrix, endpoints committed to repo | Do not run test split |
| M4 Policy value | Gain on ≥ 1 pre-registered endpoint on **validation** without unacceptable regressions | Pivot paper to DERMA-STRESS + guarantee-violation analysis |
| M5 Integrated system | All acceptance tests pass | Cut non-essential features |

---

## 22. Demo script (≈ 8 minutes)

1. Create a case; upload a clinical close-up.
2. System returns **ACQUIRE**: show the expected-loss bars and the mixed benign/malignant conformal set that triggered it.
3. Upload the dermoscopic image → system returns **PROCEED**, conformal set shrinks to one group.
4. Upload a different, clean, easy case → **PROCEED** immediately (no unnecessary acquisition).
5. Upload a blurred image of an unseen class → **DEFER** with OOD and quality reasons.
6. Open the timeline: exact snapshot, model versions, calibration and policy versions.
7. Open DERMA-STRESS lab: risk–coverage curves vs baselines, AUGRC table, acquisition-efficiency chart.
8. Show the **guarantee-violation chart** under shift — honest about where the guarantee breaks.
9. Kill the worker mid-job → case remains inspectable, job retried.

Never present one hand-picked case as proof of reliability.

---

## 23. Deliverables

1. Working web application.
2. Versioned API and schema.
3. Vision models + calibration + conformal + LTT code.
4. EADE policy and acquisition estimator.
5. Consistency module.
6. DERMA-STRESS configs and runner.
7. Baseline, ablation, sensitivity and subgroup results.
8. Pre-registration file.
9. Related-work audit.
10. Dataset/split documentation and reproduction guide.
11. Architecture diagram and API docs.
12. Security, privacy and limitations statement.
13. Final report and presentation.
14. Optional paper submission (e.g. a MICCAI/ISBI workshop on uncertainty or fairness) if results justify it.

---

## 24. Paper structure

1. Introduction — the proceed/acquire/defer problem; clinical motivation (complementary value of dermoscopy).
2. Related work — multimodal dermatology, HOT, selective classification, conformal prediction, learning to defer, active feature/modality acquisition.
3. Problem formulation — notation, cost matrix, actions.
4. Method — expected-loss policy, acquisition estimator, conformal + LTT safety layer.
5. DERMA-STRESS.
6. Experimental setup — MILK10k, splits, baselines, metrics, pre-registration.
7. Results — coverage at guaranteed risk, AUGRC, acquisition efficiency, net benefit.
8. Robustness — stress families, guarantee violation under shift.
9. Fairness — per-skin-tone coverage and miss rate.
10. Ablations and cost sensitivity.
11. Limitations, ethics, threats to validity.
12. Conclusion.

---

## 25. Risks and mitigation

| Risk | Mitigation |
|------|------------|
| HOT or later work already covers our contribution | Novelty audit in Weeks 1–2; implement HOT-style baseline; narrow claim to guarantee + cost + stress analysis |
| MILK10k licence restricts use | Check terms in Week 1; fall back to Derm7pt |
| No patient IDs → leakage | Lesion-level split, ISIC-ID dedup, stated in limitations |
| Small test set → wide CIs | Binary malignancy for guarantee; bootstrap CIs; 3 seeds; no rare-class claims |
| Guarantee fails under shift | Expected; it is measured and reported as a finding |
| Cost matrix seen as arbitrary | Pre-register values; mandatory sensitivity sweep |
| Acquisition estimator overfits | Out-of-fold targets; compare against oracle and random |
| Limited GPU | Small backbones, 3-seed ensemble max, cache features |
| RAG produces unsupported claims | Kept out of the decision path; unsupported-claim flags |
| Integration delays | Vertical slice first; modular monolith |
| Team silos | Weekly integration demo; shared pre-registration |
| Overclaiming | Hypothesis language; report every pre-registered metric |

---

## 26. Final project statement

**DERMA-GUARD is a multimodal dermatology research platform that tests whether a cost-aware, risk-controlled policy can decide when to answer, when to request dermoscopy, and when to defer — achieving higher coverage at a guaranteed malignancy miss rate than simpler abstention and acquisition rules, and degrading measurably rather than silently under evidence failures.**

**Research pillars**
1. Risk-controlled expected-loss decision engine (EADE).
2. Cost-aware acquisition with oracle-bounded efficiency.
3. DERMA-STRESS, including guarantee-violation analysis.

**Engineering pillars**
Case-centred workflow · calibrated vision models · consistency module · append-only audit trail · reproducible experiment runner.

**Next action:** download MILK10k, verify its licence, build lesion-level splits and the clinical-only baseline, and ship the vertical slice (one case, one upload, one saved model run, one visible failure state).

---

## Appendix A — Glossary

| Term | Meaning |
|------|---------|
| Coverage | Fraction of cases the system answers automatically |
| Selective risk | Error (or loss) rate among answered cases |
| AUGRC | Area under the generalized risk–coverage curve; average risk of undetected failures |
| Conformal set | A set of labels guaranteed to contain the truth with probability ≥ 1 − α |
| LTT | Learn-then-Test; selects thresholds with a high-probability risk bound |
| OOD | Out-of-distribution input |
| AE | Acquisition efficiency relative to random and oracle |
| MSES | Minimum Sufficient Evidence Set |
| NIR | Negative Information Rate |
| Net benefit | Decision-curve measure of clinical utility at a threshold probability |

## Appendix B — Pre-registration template (`prereg.yaml`)

```yaml
dataset: MILK10k
split_version: v1-lesion-stratified-60-15-10-15
label_scheme: coarse + binary_malignancy
alpha: 0.05
delta: 0.10
controlled_risk: malignant_miss_rate_on_proceeded
cost_matrix:
  correct: 0
  wrong_same_group: 1
  malignant_missed: 10
  false_alarm: 2
acquisition_cost:
  dermoscopy: 0.5
defer_cost: 1.5
max_acquisition_rounds: 1
primary_endpoint: coverage_at_guaranteed_risk
secondary_endpoints: [augrc, acquisition_efficiency, net_benefit, brier, adaptive_ece]
sensitivity:
  malignant_missed: [5, 10, 20]
  dermoscopy_cost: [0.1, 0.5, 1.0]
seeds: [0, 1, 2]
```

(Values above are placeholders; the team must agree and commit them before Week 7.)

## Appendix C — References

**Verified during this revision**
1. Tschandl et al. — MILK10k dataset, ISIC Archive. https://api.isic-archive.com/collections/425/ (DOI 10.34970/648456)
2. MILK10k Benchmark challenge. https://challenge.isic-archive.com/landing/milk10k/
3. Müller et al. — Comparative and complementary diagnostic value of dermatoscopy and clinical close-up photography, *JAAD* 2026. DOI 10.1016/j.jaad.2026.03.069
4. Mehta et al. — Revamping AI Models in Dermatology (HOT). https://arxiv.org/abs/2311.01009
5. Traub et al. — Overcoming Common Flaws in the Evaluation of Selective Classification Systems, NeurIPS 2024. https://arxiv.org/abs/2407.01032
6. JI-ADF — Joint-Individual Learning with Adaptive Decision Fusion. https://arxiv.org/abs/2604.27343
7. Fayyad et al. — Empirical Validation of Conformal Prediction for Trustworthy Skin Lesions Classification. https://arxiv.org/abs/2312.07460
8. Lu et al. — Fair Conformal Predictors for Applications in Medical Imaging. https://arxiv.org/abs/2109.04392
9. Zoravar et al. — Domain Adaptive Skin Lesion Classification via Conformal Ensemble of ViTs. https://arxiv.org/abs/2505.15997
10. Rheude et al. — Cohort-Based Active Modality Acquisition. https://arxiv.org/abs/2505.16791
11. Valancius et al. — Acquisition Conditioned Oracle for Nongreedy AFA. https://arxiv.org/abs/2302.13960
12. Shim et al. — Joint Active Feature Acquisition and Classification, NeurIPS 2018.
13. SAGEAgent — Cost-aware modality acquisition for survival prediction. https://arxiv.org/abs/2607.09521

**Established methods (standard references)**
14. Geifman & El-Yaniv — Selective Classification for Deep Neural Networks, NeurIPS 2017. arXiv 1705.08500
15. Guo et al. — On Calibration of Modern Neural Networks, ICML 2017. arXiv 1706.04599
16. Lakshminarayanan et al. — Deep Ensembles, NeurIPS 2017. arXiv 1612.01474
17. Angelopoulos & Bates — A Gentle Introduction to Conformal Prediction. arXiv 2107.07511
18. Angelopoulos et al. — Uncertainty Sets for Image Classifiers using Conformal Prediction (RAPS). arXiv 2009.14193
19. Angelopoulos et al. — Learn then Test. arXiv 2110.01052
20. Angelopoulos et al. — Conformal Risk Control. arXiv 2208.02814
21. Mozannar & Sontag — Consistent Estimators for Learning to Defer to an Expert, ICML 2020. arXiv 2006.01862
22. Liu et al. — Energy-based Out-of-distribution Detection, NeurIPS 2020. arXiv 2010.03759
23. Lee et al. — Mahalanobis OOD detector, NeurIPS 2018. arXiv 1807.03888
24. Vickers & Elkin — Decision Curve Analysis, *Medical Decision Making* 2006.
25. Pacheco et al. — PAD-UFES-20, *Data in Brief* 2020.
26. Kawahara et al. — Derm7pt, *IEEE JBHI* 2019.
27. Tschandl et al. — HAM10000, *Scientific Data* 2018. arXiv 1803.10417

**To verify before citing (from v1.0)**
DermFormer (2025); Zuo et al. / CosCatNet (2025); QA-MoE (IJCAI 2026); SAGE (npj Digital Medicine 2026); PanDerm; DermEVAL; DermFusion; UAGF.
