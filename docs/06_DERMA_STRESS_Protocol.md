# 06 — DERMA-STRESS Protocol
### *Systematic Stress-Testing, Failure Modes, and Measuring Guarantee Violation Under Shift*

> **Prerequisites:** [`05_Architecture_Deep_Dive.md`](./05_Architecture_Deep_Dive.md)  
> **Next:** [`07_Research_Methodology.md`](./07_Research_Methodology.md)

---

## 1. Why DERMA-STRESS Exists

In standard machine learning evaluations, models are tested on clean, independent, and identically distributed (i.i.d.) test splits drawn from the same clinical centers as training data. Under this idealized setting, modern deep neural networks achieve deceptive clinical parity.

However, real-world teledermatology deployment introduces systematic perturbations:
1. Patients take unfocused smartphone photos in dim lighting.
2. Anatomic metadata is omitted or miskeyed by triage staff.
3. High-resolution contact dermoscopy is frequently unavailable during primary remote encounters.
4. Patient populations vary across skin types (Fitzpatrick I–VI) and clinical centers.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THE CORE SCIENTIFIC REALIZATION                      │
├────────────────────────────────────────────────────────────────────────┤
│ Mathematical guarantees (e.g., LTT / Conformal Risk Control) rely      │
│ strictly on the EXCHANGEABILITY assumption between calibration and     │
│ test distributions.                                                    │
│                                                                        │
│ When data shifts, the theoretical guarantee CANNOT be proven to hold.   │
│                                                                        │
│ Rather than hiding this failure, DERMA-STRESS systematically measures: │
│ 1. By exactly how much the empirical risk violates the bound α        │
│ 2. Whether the policy fails silently (proceeds with errors) or        │
│    safely falls back to ACQUIRE or DEFER                              │
│ 3. The exact degradation of Coverage and Cost under combined stressors │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 7 Stress Families (A through G)

DERMA-STRESS specifies 7 distinct failure families. Each family is strictly versioned, reproducible via fixed random seeds, and parameterised across 3 severity levels ($s \in \{1, 2, 3\}$).

```
                      ┌─────────────────────────────────┐
                      │     DERMA-STRESS BENCHMARK      │
                      └────────────────┬────────────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        │              │               │               │              │
   ▼ Family A     ▼ Family B      ▼ Family C      ▼ Family D     ▼ Family E
  Image Blur     Missingness    Contradictory     Distribution   Leave-Class
 & Compression   of Modality       Context           Shift           OOD
        │              │               │               │              │
        └──────────────┴───────┬───────┴───────────────┴──────────────┘
                               │
                       ┌───────┴───────┐
                       │               │
                  ▼ Family F      ▼ Family G
                   Combined       Operational
                   Failures        Failures
```

---

### Family A: Image Quality Degradation

Evaluates robustness to smartphone camera artifacts and compression. Applied independently to clinical photography $x^c$ and dermoscopy $x^d$.

| Perturbation | Severity 1 (Mild) | Severity 2 (Moderate) | Severity 3 (Severe) | Implementation |
|---|---|---|---|---|
| **Gaussian Blur** | $\sigma = 1.0$, $k = 5$ | $\sigma = 2.5$, $k = 11$ | $\sigma = 5.0$, $k = 21$ | `cv2.GaussianBlur` |
| **Motion Blur** | Angle $\theta \sim \mathcal{U}(0, 180^\circ)$, length $5\text{px}$ | Length $15\text{px}$ | Length $30\text{px}$ | Directional linear convolution kernel |
| **JPEG Compression** | Quality $Q = 60$ | Quality $Q = 35$ | Quality $Q = 15$ | PIL save with quality parameter |
| **Exposure / Contrast** | Contrast $\times 0.8$, Brightness $+20$ | Contrast $\times 0.5$, Brightness $+40$ | Contrast $\times 0.3$, Brightness $+60$ | Linear intensity remapping |
| **Resolution Downsampling** | Downscale to $128 \times 128 \to$ bicubic up | Downscale to $64 \times 64 \to$ bicubic up | Downscale to $32 \times 32 \to$ bicubic up | Spatial down/up-sampling |

#### Degradation Function:
$$\tilde{x} = \mathcal{T}_{\text{deg}}(x; \text{type}, s)$$

The Vision Intelligence quality module must detect these perturbations:
- Laplacian Variance: $\sigma^2_{\text{Laplacian}}(\tilde{x}) < \tau_{\text{blur}}$
- Exposure Score: Fraction of pixels in saturated bins $[0, 5] \cup [250, 255]$

---

### Family B: Missing Evidence & Modality Dropout

Real teledermatology cases often lack contact dermoscopy or complete patient records. DERMA-STRESS formalises modality absence:

1. **Condition B1 — Dermoscopy Missing:**
   Case has $S = \{c, z\}$ (Clinical image + Metadata). Modality $d$ is marked `MISSING`. The system must decide between `PROCEED` (on clinical evidence alone), `ACQUIRE` (request dermoscopy), or `DEFER`.
2. **Condition B2 — Metadata Missing:**
   Case has $S = \{c, d\}$ (Images present, tabular metadata omitted). Missingness vector $m_{\text{missing}} = 1$.
3. **Condition B3 — Clinical Only:**
   Case has $S = \{c\}$. Both dermoscopy and metadata are absent.
4. **Critical Semantic Distinction:**
   - `MISSING`: Asset was never collected. Missingness flag is active; prior imputation or vacant representation is used.
   - `PRESENT_BUT_LOW_QUALITY`: Asset exists but quality score is below acceptable threshold ($q < \tau_q$).

---

### Family C: Contradictory Clinical Context

Simulates clerical errors and contradictory triage information without leaking true diagnostic labels.

1. **Anatomic Site Perturbation:**
   Swaps true anatomic location with discordant site (e.g., facial lentigo maligna relabelled as "plantar foot").
   $$\tilde{z}_{\text{site}} \sim \text{Categorical}(\mathcal{S} \setminus \{z_{\text{site}}\})$$
2. **Demographic Inversion:**
   Shifts age bracket by $\pm 30$ years (e.g., geriatric melanoma relabelled as pediatric 0–10 years).
3. **Conflicting Clinical Narrative:**
   Inserts synthetically generated discordant history (e.g., patient text states *"stable unchanged brown spot for 15 years"* while image displays rapidly evolving ulcerated nodule).
4. **Consistency Module Response:**
   Jensen-Shannon Divergence $D_{\text{JS}}(p_c \parallel p_z)$ and text-metadata contradiction flags fire:
   $$\text{Flag}_{\text{contradiction}} = \mathbb{1}[D_{\text{JS}}(p_c \parallel p_z) > \tau_{\text{conflict}}] \implies \text{Force DEFER}$$

---

### Family D: Distribution Shift

Evaluates out-of-distribution generalization across geography, devices, and clinical centers.

```
Distribution Shifts:
├── Center Shift: Train on MILK10k Centers 1–4 ──────► Test on Center 5
└── Device / Population Shift: MILK10k (Spain/Europe) ──► PAD-UFES-20 (Brazil Smartphone)
```

1. **Center Shift (Intra-dataset):**
   MILK10k originates from 5 distinct clinical centers. Models are trained on centers $\{1, 2, 3, 4\}$ and evaluated on holdout Center 5 to test hospital covariate shift.
2. **Geographic & Camera Shift (PAD-UFES-20):**
   Smartphone clinical images collected in Espírito Santo, Brazil. Tested strictly on the overlapping 6 diagnostic classes (BCC, SCC, MEL, NEV, ACK, SEK). Note: PAD-UFES-20 does not include dermoscopy, isolating clinical-branch generalization.

---

### Family E: Leave-Class-Out OOD (Novel Pathologies)

In clinical practice, lesions arrive that do not belong to the 7 or 11 pre-defined training classes (e.g., Kaposi sarcoma, cutaneous lymphoma, atypical vascular tumors).

```
Training Classes:   {MEL, NV, BCC, BKL, AKIEC, SCC, VASC}  (K classes)
Excluded OOD Class: {DF (Dermatofibroma)} or external classes (Fitzpatrick17k rare classes)
```

1. **Protocol:**
   - Remove class $k^*$ completely from training splits $\mathcal{D}_{\text{train}}$ and calibration splits $\mathcal{D}_{\text{cal}}$.
   - Feed cases with ground truth $y = k^*$ to the system at test time.
2. **Required Behavior:**
   - Energy Score $E(x) = -T \log \sum_k e^{f_k(x)/T} > \tau_{\text{energy}}$
   - Mahalanobis Distance $D_M(z) = (z - \mu)^T \Sigma^{-1} (z - \mu) > \tau_M$
   - Conformal mixed-set or wide empty prediction set triggers **instant DEFER**.
   - `PROCEED` on an OOD class is scored as a high-penalty failure.

---

### Family F: Combined Multi-Stress Failures

Real clinical disasters rarely occur in isolation. Family F pairs multiple simultaneous failure modes to evaluate worst-case catastrophic failure modes:

| Test ID | Combined Stress Configuration | Clinical Analogy |
|---|---|---|
| **F1** | Degraded Clinical Image (A: Blur $s=2$) + Missing Dermoscopy (B1) | Remote rural patient submits blurry selfie; no dermatoscope available. |
| **F2** | Missing Dermoscopy (B1) + Contradictory Metadata (C: Age shift) | Triage clerk enters wrong patient chart on a smartphone submission. |
| **F3** | Distribution Shift (D: PAD-UFES) + Degraded Image (A: JPEG $s=3$) | Cheap smartphone camera in an external, under-resourced hospital. |
| **F4** | Leave-Class-Out OOD (E) + Motion Blur (A: Blur $s=2$) | Rare unknown lesion photographed poorly. |

---

### Family G: Operational and System Failures

Evaluates platform resilience and fault tolerance under infrastructure breakdown:
1. **Inference Timeout:** Vision inference worker hangs or exceeds 15,000ms latency budget.
2. **Worker Crash:** Celery/RQ container terminated via `SIGKILL` during active assessment.
3. **Corrupted File Upload:** Uploaded payload has valid `.jpg` extension but corrupt magic bytes.
4. **External LLM Outage:** Timeout during symptom extraction fallback.
5. **Acceptance Criteria:** Zero silent failures, zero unhandled 500 crashes. All state stored as `FAILED_RETRIEVABLE` with full audit trace.

---

## 3. Formal Definition of Guarantee Violation

The primary theoretical contribution of DERMA-GUARD v2.0 is the formal quantification of statistical breakdown under shift.

Under the Learn-Then-Test (LTT) calibration on exchangeable split $\mathcal{D}_{\text{cal}}$:
$$\mathbb{P}\left(\mathcal{R}_{\text{cal}}(\hat{\lambda}) \le \alpha\right) \ge 1 - \delta$$

When evaluated on stress condition $\mathcal{S}$, the empirical test risk on auto-proceeded cases is:
$$\hat{\mathcal{R}}_{\text{test}}^{(\mathcal{S})}(\hat{\lambda}) = \frac{\sum_{i \in \text{Test}^{(\mathcal{S})}} \mathbb{1}[y_i = \text{mal}, \hat{y}_i = \text{ben}, a_i = \text{PROCEED}]}{\sum_{i \in \text{Test}^{(\mathcal{S})}} \mathbb{1}[a_i = \text{PROCEED}]}$$

### The Guarantee Violation Metric:
$$\Delta_{\text{violation}}^{(\mathcal{S})} = \max\left(0,\, \hat{\mathcal{R}}_{\text{test}}^{(\mathcal{S})}(\hat{\lambda}) - \alpha\right)$$

```
VIOLATION INTERPRETATION SPECTRUM:
┌─────────────────────────────────────────────────────────────────────────┐
│ Δ_violation = 0.000 : GUARANTEE PRESERVED                               │
│ The policy maintained safe malignancy miss rates despite perturbation.  │
├─────────────────────────────────────────────────────────────────────────┤
│ 0.000 < Δ_violation ≤ 0.020 : GRACEFUL DEGRADATION                      │
│ Minor statistical leakage; empirical risk stayed close to nominal α.    │
├─────────────────────────────────────────────────────────────────────────┤
│ Δ_violation > 0.050 : CATASTROPHIC GUARANTEE FAILURE                   │
│ Covariate shift invalidates exchangeability; system over-proceeded      │
│ on uncalibrated errors. Mandatory reporting in paper findings.         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Per-Condition Benchmark Output Schema

Every execution of the DERMA-STRESS runner generates an immutable, structured JSON experiment log matching this schema:

```json
{
  "experiment_id": "exp_stress_2026_F1_seed42",
  "dataset_version": "milk10k_v2.1",
  "split_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "stress_family": "F",
  "stress_condition": "F1_degraded_clinical_missing_dermoscopy",
  "severity_level": 2,
  "config_params": {
    "blur_sigma": 2.5,
    "missing_modality": "dermoscopy",
    "alpha_target": 0.05,
    "delta_confidence": 0.10
  },
  "metrics": {
    "proceed_coverage": 0.642,
    "empirical_risk": 0.068,
    "guarantee_violation": 0.018,
    "conformal_mean_set_size": 1.42,
    "conformal_mixed_set_fraction": 0.18,
    "acquire_rate": 0.210,
    "defer_rate": 0.148,
    "acquisition_efficiency_AE": 0.74,
    "negative_information_rate_NIR": 0.03,
    "augrc": 0.041,
    "mean_cost_per_case": 0.284
  },
  "confidence_intervals_bootstrap_95": {
    "empirical_risk": [0.052, 0.086],
    "proceed_coverage": [0.608, 0.675]
  },
  "execution_metadata": {
    "runtime_total_seconds": 184.2,
    "gpu_device": "NVIDIA RTX 4090",
    "git_commit": "7a9f2bc"
  }
}
```

---

## 5. Summary: DERMA-STRESS vs Conventional Benchmarking

| Dimension | Standard AI Benchmarking | DERMA-STRESS Protocol |
|---|---|---|
| **Test Distribution** | Clean, curated i.i.d. holdout | 7 systematic perturbation families |
| **Decision Space** | Fixed 100% predictions (Forced choice) | 3-action policy (Proceed / Acquire / Defer) |
| **Statistical Rigor** | Average accuracy / AUC numbers | Measured **Guarantee Violation** ($\hat{\mathcal{R}} - \alpha$) |
| **Failure Handling** | Silent errors in confident predictions | Transparent fallback rates to Defer |
| **Auditability** | Notebook script evaluation | Version-locked, hashed, pre-registered test harness |

*Proceed to [`07_Research_Methodology.md`](./07_Research_Methodology.md) for experimental design, baselines, and statistical test protocols.*
