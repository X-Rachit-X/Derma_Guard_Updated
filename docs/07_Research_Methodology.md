# 07 — Research Methodology
### *Experimental Design, Baselines B1–B8, Ablation Protocol, and Statistical Testing*

> **Prerequisites:** [`06_DERMA_STRESS_Protocol.md`](./06_DERMA_STRESS_Protocol.md)  
> **Next:** [`08_Platform_Engineering.md`](./08_Platform_Engineering.md)

---

## 1. The Research Principles: Honest, Verifiable Science

DERMA-GUARD is designed around strict research integrity to eliminate common traps in medical AI literature:
1. **No Data Snooping:** The test split is sequestered and touched **only once** per frozen model configuration.
2. **Pre-Registration Discipline:** All primary endpoints, cost parameters ($c_{\text{acq}}, c_{\text{def}}, \ell_{\text{miss}}$), risk targets ($\alpha = 0.05$), and confidence levels ($\delta = 0.10$) are locked in `prereg.yaml` before evaluation.
3. **Identical Evaluation Cohort:** All baselines (B1 through B8) and ablation models are evaluated on the **exact same lesion cases** using identical random seeds.
4. **Honest Outcome Reporting:** A negative result (e.g., if EADE fails to beat Learning-to-Defer under specific conditions) is not hidden or post-hoc refactored; it is reported with full guarantee-violation diagnostics.

---

## 2. Experimental Endpoints

```
                               EXPERIMENTAL ENDPOINTS
                                         │
               ┌─────────────────────────┴─────────────────────────┐
               ▼                                                   ▼
     PRIMARY ENDPOINT (Pre-registered)                    SECONDARY ENDPOINTS
   Coverage at Guaranteed Risk:                         ├── Selective AUGRC
   Fraction of cases proceeded under α ≤ 0.05            ├── Acquisition Efficiency (AE)
   with 1 − δ = 0.90 statistical confidence            ├── Conformal Mixed-Set Fraction
                                                        ├── Clinical Net Benefit (DCA)
                                                        └── Subgroup Fairness (Skin Tone)
```

### 2.1 Primary Pre-Registered Endpoint
- **Coverage at Guaranteed Risk ($C_{\alpha,\delta}^*$):**
  The fraction of test cohort cases auto-proceeded by the policy while strictly satisfying the finite-sample risk constraint:
  $$\mathbb{P}\left(\mathcal{R}_{\text{test}}(\hat{\lambda}) \le \alpha\right) \ge 1 - \delta \quad \text{for } \alpha = 0.05, \, \delta = 0.10$$
  A superior policy achieves higher coverage without violating the malignancy miss ceiling.

### 2.2 Secondary Endpoints
1. **Classification Performance:**
   - Balanced Accuracy, Macro-F1 across 11 diagnostic classes.
   - Binary Malignancy AUROC with 95% DeLong confidence intervals.
2. **Calibration Metrics:**
   - Adaptive (equal-mass binning) Expected Calibration Error ($\text{ECE}_{\text{adaptive}}$).
   - Classwise ECE ($\text{cECE}$) and Brier Score.
3. **Selective Prediction:**
   - **AUGRC (Area Under Generalized Risk-Coverage Curve):** Primary multi-threshold metric (Traub et al., NeurIPS 2024), avoiding the mathematical flaws of classic AURC.
   - Selective Risk at fixed coverages: $\mathcal{R}(\text{Cov} = 0.70)$, $\mathcal{R}(\text{Cov} = 0.80)$, $\mathcal{R}(\text{Cov} = 0.90)$.
4. **Acquisition Diagnostics:**
   - **Acquisition Efficiency ($\text{AE}$):** Relative gain over random acquisition bounded by the oracle:
     $$\text{AE}(r) = \frac{G_{\text{policy}}(r) - G_{\text{random}}(r)}{G_{\text{oracle}}(r) - G_{\text{random}}(r)}$$
   - **Negative Information Rate ($\text{NIR}_m$):** Frequency where acquiring dermoscopy *increased* classification loss.
   - **MSES (Minimum Sufficient Evidence Set):** Proportion of cases where clinical image alone satisfies safe risk.
5. **Clinical Decision Utility:**
   - **Decision-Curve Net Benefit:**
     $$\text{NB}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N}\left(\frac{p_t}{1 - p_t}\right)$$
     Evaluated across referral threshold probabilities $p_t \in [0.05, 0.30]$.
   - Mean Expected Clinical Cost under cost matrix $\mathbf{C}$.

---

## 3. The 8 Benchmark Baselines (B1 through B8)

To rigorously isolate the contribution of each component, DERMA-GUARD evaluates 8 formal baselines across identical splits:

| Baseline ID | Name | Permitted Actions | Decision Mechanism | Role in Paper |
|---|---|---|---|---|
| **B1** | Clinical-Only Fixed | `PROCEED` | Always predicts using smartphone image alone. Zero deferral, zero acquisition. | Raw bottom-line clinical baseline. |
| **B2** | Fused Fully-Acquired | `PROCEED` | Assumes dermoscopy is always acquired for every patient ($r = 1.0$). Zero deferral. | Maximum-information non-abstaining upper bound. |
| **B3** | Confidence Abstention | `PROCEED`, `DEFER` | Calibrated Maximum Softmax Probability (MSP). Defers if $\max_k p_k < \tau_{\text{LTT}}$. | Classic selective classification benchmark. |
| **B4** | Learning-to-Defer (L2D) | `PROCEED`, `DEFER` | Mozannar & Sontag surrogate loss with classifier-rejector joint training. | Modern literature standard for human-AI teaming. |
| **B5** | HOT-Style Heuristic | `PROCEED`, `ACQUIRE` | Acquires dermoscopy if predictive entropy $H(p_c) > \tau_{\text{entropy}}$, else proceeds. | Direct comparison against HOT (Mehta et al.). |
| **B6** | Greedy Info-Gain + Abstain | `PROCEED`, `ACQUIRE`, `DEFER` | Acquires if Ensemble Mutual Information $\text{MI} > \tau_{\text{acq}}$, defers if $H > \tau_{\text{def}}$. | Heuristic 3-action competitor. |
| **B7** | Random Acquisition | `ACQUIRE` (random) | Uniform random acquisition at matched rate $r$; defers via B3 rule. | Empirical floor for acquisition testing. |
| **B8** | Oracle Acquisition | `ACQUIRE` (oracle) | Selects top $rN$ cases with largest empirical loss reduction $L(S) - L(S \cup \{d\})$. | Theoretical performance ceiling. |
| **OURS** | **Full EADE Engine** | `PROCEED`, `ACQUIRE`, `DEFER` | **Expected-loss minimization + Out-of-fold Value Estimator + Conformal Safety Layer.** | Proposed novel contribution. |

---

## 4. Ablation Study Protocol

The ablation matrix tests 10 distinct ablations to isolate each mathematical component's necessity:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EADE ABLATION MATRIX                            │
├────┬─────────────────────────────┬────────────────────────────────────┤
│ ID │ Configuration Tested        │ Hypothesis Tested                  │
├────┼─────────────────────────────┼────────────────────────────────────┤
│ A0 │ Full DERMA-GUARD System     │ Complete proposed architecture     │
│ A1 │ No LTT Safety Layer         │ Impact of statistical guarantee    │
│ A2 │ No Conformal Mixed-Set Rule │ Necessity of conformal set bounds  │
│ A3 │ No Value Estimator (Entropy)│ Learned value vs Shannon entropy   │
│ A4 │ No Consistency Module       │ Value of Cross-Modal divergence    │
│ A5 │ No Mahalanobis OOD Module   │ Energy/Distance rejection utility  │
│ A6 │ No Image Quality Filtering  │ Laplacian/Exposure gate value      │
│ A7 │ Uncalibrated Posteriors     │ Impact of Temperature Scaling      │
│ A8 │ Single Seed (No Ensemble)   │ Epistemic uncertainty benefit      │
│ A9 │ Cost Matrix Sensitivity     │ Robustness to clinical miss weight │
└────┴─────────────────────────────┴────────────────────────────────────┘
```

---

## 5. Statistical Inference Protocol

Every metric claim must be supported by formal statistical hypothesis testing:

### 5.1 Lesion-Level Paired Bootstrap
- Minimum 2,000 bootstrap resamples on the test cohort:
  $$\hat{\theta}^*_b \sim \text{Bootstrap}(\mathcal{D}_{\text{test}}), \quad b = 1, \dots, 2000$$
- 95% Confidence Intervals reported via percentile bootstrap $[q_{0.025}, q_{0.975}]$.
- Paired differences $\Delta_b = \theta_{\text{ours}}^* - \theta_{\text{baseline}}^*$ to confirm non-zero improvement.

### 5.2 Hypothesis Significance Tests
- **Accuracy & Categorical Decisions:** McNemar's paired chi-squared test.
- **AUROC Curves:** DeLong's non-parametric paired test.
- **Multiple Comparisons:** Holm-Bonferroni correction applied across all 8 baseline comparisons to ensure Family-Wise Error Rate (FWER) $\le 0.05$.

---

## 6. Subgroup Fairness & Demographic Disparities

MILK10k provides a 6-level skin tone classification (Levels 1–6). 

```
Fairness Protocol:
├── Marginal Calibration (Overall Population) ──► Check subgroup miss rate
└── Group-Conditional Conformal Calibration ────► Compute λ_g per skin tone
```

1. **Evaluation:** Evaluate malignancy miss rate $\mathcal{R}_g$ and proceed coverage $C_g$ for each skin tone group $g \in \{1, \dots, 6\}$.
2. **Mitigation:** Run **Group-Conditional Conformal Prediction**:
   $$\mathbb{P}\left(\mathcal{R}_g(\hat{\lambda}_g) \le \alpha\right) \ge 1 - \delta \quad \forall g$$
3. **Disparity Metric:** Max Risk Gap:
   $$\Delta_{\text{fairness}} = \max_g \hat{\mathcal{R}}_g - \min_g \hat{\mathcal{R}}_g$$
   Report sample sizes and merged bins for sparse darker skin tone categories (Levels 5–6).

---

## 7. Pre-Registration Schema (`prereg.yaml`)

Before touching the test split, the team freezes this exact specification file:

```yaml
# DERMA-GUARD PRE-REGISTRATION SPECIFICATION
version: "2.0.0-frozen"
timestamp: "2026-10-15T00:00:00Z"
git_commit_hash: "3b8a1c94d2e850"

safety_parameters:
  alpha_nominal_risk: 0.05      # 5% malignancy miss rate ceiling
  delta_confidence: 0.10        # 90% finite-sample confidence
  cost_malignancy_miss: 10.0    # Clinical penalty weight
  cost_false_alarm: 1.0
  cost_acquisition_dermoscopy: 0.25
  cost_human_deferral: 0.50

dataset:
  primary_collection: "MILK10k"
  train_split_ratio: 0.60
  val_split_ratio: 0.15
  cal_split_ratio: 0.10
  test_split_ratio: 0.15
  split_stratification: "lesion_id_and_diagnosis"

primary_hypothesis:
  name: "H1_Coverage_Advantage"
  description: "EADE achieves > 10% absolute gain in coverage over B3 at matched alpha = 0.05"
  statistical_test: "paired_bootstrap_2000"
  significance_level: 0.05
```

*Proceed to [`08_Platform_Engineering.md`](./08_Platform_Engineering.md) for full technology stack, backend APIs, and deployment specifications.*
