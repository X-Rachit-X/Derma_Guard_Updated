# 03 — Mathematical Foundations
### *All Equations Derived Step by Step — From First Principles*

> **Prerequisites:** [`02_ML_Concepts.md`](./02_ML_Concepts.md)  
> **Next:** [`04_Data_Pipeline.md`](./04_Data_Pipeline.md)

---

## Notation Reference

| Symbol | Type | Meaning |
|--------|------|---------|
| `xᶜ` | image | Clinical close-up photograph |
| `xᵈ` | image | Dermoscopic image |
| `z` | vector | Structured metadata (age, sex, site, skin tone) |
| `S ⊆ {c, d, z}` | set | Currently available evidence subset |
| `y ∈ {1,...,K}` | integer | True diagnostic class (K classes) |
| `g(y) ∈ {B, M}` | binary | Malignancy grouping: Benign or Malignant |
| `p_S(y \| xₛ)` | vector | Calibrated posterior probability over classes, given evidence S |
| `L(y, ŷ)` | scalar | Loss of predicting ŷ when truth is y |
| `c_acq(m)` | scalar | Cost of acquiring modality m |
| `c_def` | scalar | Fixed cost of deferring to a human |
| `α` | scalar | Target malignancy miss rate (pre-registered: 0.05) |
| `δ` | scalar | Failure probability (pre-registered: 0.10) |
| `λ` | scalar | LTT-calibrated PROCEED permissiveness threshold |
| `T_max` | integer | Maximum acquisition rounds (pre-registered: 1) |
| `M` | integer | Number of ensemble seeds (3) |
| `T` | scalar | Temperature for temperature scaling |

---

## Section 1 — Cost Matrix

### 1.1 Why an asymmetric cost matrix?

In standard multi-class classification, all errors are treated equally. In medicine, this is wrong. Missing a melanoma (false negative) is far more dangerous than misclassifying a benign lesion as another benign type.

DERMA-GUARD uses a clinically motivated asymmetric cost matrix:

### 1.2 The cost matrix

```
                    Predicted class
                 ┌──────────────────────────────────────────────┐
Truth class      │ Correct  Wrong-same-grp  Benign←Malign  Malig←Benign │
─────────────────┼──────────────────────────────────────────────┤
Loss             │   0         ℓ_sub            ℓ_miss          ℓ_fa   │
Default values   │   0           1                 10              2    │
```

**Explanation of each cell:**
- **Correct (0):** Predicted the right class → no cost
- **Wrong, same malignancy group (ℓ_sub = 1):** Called it seborrheic keratosis instead of solar lentigo, but both are benign → small cost
- **Benign predicted for Malignant (ℓ_miss = 10):** The critical error — missed a melanoma → highest cost
- **Malignant predicted for Benign (ℓ_fa = 2):** Unnecessary alarm, patient gets extra testing → moderate cost

### 1.3 Pre-registration requirement

All cost values are locked in `prereg.yaml` before the final test run. A mandatory sensitivity sweep over `ℓ_miss ∈ {5, 10, 20}` and `c_acq ∈ {0.1, 0.5, 1.0}` is required, because results depend on these choices.

---

## Section 2 — Expected Loss of Each Action

### 2.1 Expected loss of PROCEED

Given calibrated probability vector `p_S(y | xₛ)`, the Bayes-optimal expected loss of proceeding (choosing the best possible prediction) is:

```
R_proc(S) = min over ŷ [ Σ_y  p_S(y | xₛ) × L(y, ŷ) ]
```

**Step-by-step:**

1. For each possible prediction ŷ, compute the expected loss if you predict ŷ:
   ```
   Expected_loss(ŷ) = p(class_1) × L(class_1, ŷ)
                    + p(class_2) × L(class_2, ŷ)
                    + ...
                    + p(class_K) × L(class_K, ŷ)
   ```

2. Pick the ŷ that minimises this expected loss.

**Worked example:**
```
Say K = 2 (binary: malignant M, benign B)
p(M) = 0.35, p(B) = 0.65

If we predict ŷ = B:
  E_loss = p(M) × L(M, B) + p(B) × L(B, B)
         = 0.35 × 10      + 0.65 × 0
         = 3.5

If we predict ŷ = M:
  E_loss = p(M) × L(M, M) + p(B) × L(B, M)
         = 0.35 × 0       + 0.65 × 2
         = 1.3

R_proc = min(3.5, 1.3) = 1.3  (best action: predict malignant)
```

Note: even with only 35% probability of malignancy, the optimal prediction is "malignant" because the miss cost (10) is so high.

### 2.2 Expected loss of ACQUIRE

Acquiring modality m reduces the expected proceed-loss by an estimated amount, but costs c_acq(m):

```
R_acq(S, m) = E[ R_proc(S ∪ {m}) ] + c_acq(m)
            = R_proc(S) - h_m(xₛ)  + c_acq(m)
```

Where `h_m(xₛ)` is the **acquisition-value estimator** — it predicts how much adding modality m will reduce R_proc.

- `R_proc(S)` = expected loss without the new modality
- `h_m(xₛ)` = estimated loss reduction from acquiring m (must be positive to be beneficial)
- `c_acq(m)` = cost of acquiring m

**Intuition:** R_acq is low when the estimator says acquiring m will substantially reduce the proceed loss, and the acquisition cost is low.

### 2.3 Expected loss of DEFER

Deferral has a fixed cost (the cost of occupying a human expert):

```
R_def = c_def   (pre-registered: 1.5)
```

### 2.4 The raw policy

The raw policy (before the safety layer) simply picks the action with the lowest expected loss:

```
a_raw(S) = argmin {
    R_proc(S),
    min over m in M_avail\S [ R_acq(S, m) ],
    R_def
}
```

Where `M_avail` = modalities that are actually available for this case (can't acquire dermoscopy if the patient didn't bring the dermoscope).

---

## Section 3 — Acquisition-Value Estimator

### 3.1 The problem

We need `h_m(xₛ)` — a function that predicts how much acquiring modality m will reduce the expected proceed-loss. But at prediction time, we haven't acquired m yet (that's the point — we're deciding whether to).

### 3.2 Out-of-fold targets (avoiding optimistic bias)

On MILK10k, every lesion already has both modalities. So we can compute the **realised benefit** of dermoscopy:

```
t_i = R_proc({c,z})_i - R_proc({c,d,z})_i
```

This is the actual reduction in expected loss for lesion i when dermoscopy is added.

**Problem:** if we train h_d on the same split used for model training, we'll overfit. The model would learn the training cases too well.

**Solution: out-of-fold (OOF) predictions:**

1. Split training data into 5 folds
2. For fold k: train the models on the other 4 folds, get predictions on fold k
3. Compute t_i for each training lesion using held-out predictions
4. Train h_d on these OOF targets

This ensures h_d sees unbiased estimates of the benefit.

### 3.3 Features for h_d

h_d is a gradient-boosted tree (LightGBM) or small MLP trained on:
- Clinical-model posterior probabilities (all K classes)
- Predictive entropy H[P̄]
- Mutual information MI
- Quality scores (blur, exposure, resolution)
- OOD score (energy)
- Structured metadata z

These are all available **before** acquiring dermoscopy.

### 3.4 At test time

```
Estimated expected loss after acquiring dermoscopy:
  E[R_proc(S ∪ {d})] = R_proc(S) - h_d(xₛ)

R_acq(S, d) = R_proc(S) - h_d(xₛ) + c_acq(d)
```

---

## Section 4 — Calibration Mathematics

### 4.1 Temperature scaling

Given model logits `f = [f₁, f₂, ..., f_K]` and temperature T:

```
Calibrated probability:  P(class k) = exp(fₖ/T) / Σⱼ exp(fⱼ/T)
```

T is found by minimising the negative log-likelihood on the validation split:

```
T* = argmin_T  [ -Σᵢ log P_T(yᵢ | xᵢ) ]
```

where yᵢ is the true label for validation sample i.

### 4.2 Ensemble averaging

Given M = 3 seeds, the ensemble posterior is:

```
P̄(y) = (1/M) × Σⱼ₌₁ᴹ  Pⱼ(y | x)
```

### 4.3 Predictive entropy

```
H[P̄] = -Σ_y  P̄(y) × log₂(P̄(y))

Range: [0, log₂(K)]
- 0 = completely certain (all mass on one class)
- log₂(K) = completely uncertain (uniform distribution)
```

### 4.4 Mutual information (epistemic uncertainty)

```
MI = H[P̄] - (1/M) × Σⱼ H[Pⱼ]
   = H[P̄] - (1/M) × Σⱼ [ -Σ_y Pⱼ(y) × log₂(Pⱼ(y)) ]
```

**Interpretation:** 
- `H[P̄]` = how uncertain is the ensemble as a whole?
- `(1/M) × Σⱼ H[Pⱼ]` = how uncertain is each individual model on average?
- `MI` = the extra uncertainty beyond what each model is individually uncertain about = **ensemble disagreement**

---

## Section 5 — Cross-Modal Consistency

### 5.1 Jensen-Shannon Divergence

The consistency module computes the JSD between modality-specific posteriors.

**KL divergence** (asymmetric, unbounded):
```
KL(P || Q) = Σ_y  P(y) × log[ P(y) / Q(y) ]
```

**JSD** (symmetric, bounded between 0 and ln2):
```
m = ½(P + Q)   ← the mixture distribution
JSD(P || Q) = ½ KL(P || m) + ½ KL(Q || m)
```

**Worked example:**
```
P_clinical    = {malignant: 0.80, benign: 0.20}
P_metadata    = {malignant: 0.30, benign: 0.70}

m = {malignant: 0.55, benign: 0.45}

KL(P_clin || m)  = 0.80×log(0.80/0.55) + 0.20×log(0.20/0.45) ≈ 0.30
KL(P_meta || m)  = 0.30×log(0.30/0.55) + 0.70×log(0.70/0.45) ≈ 0.28

JSD = ½×0.30 + ½×0.28 = 0.29   (large disagreement → trigger ACQUIRE)
```

### 5.2 Contradiction flags

Rule-based: if structured metadata says "anatomic site = face" but extracted symptom text says "on the back", flag as contradictory. These boolean flags supplement the continuous JSD score.

---

## Section 6 — OOD Detection Mathematics

### 6.1 Energy score

Given logits `fₖ(x)` for class k and temperature T:

```
E(x) = -T × log [ Σₖ exp(fₖ(x) / T) ]
```

For in-distribution inputs: logits are large and concentrated → sum of exponentials is large → energy is a large negative number (close to 0 when negated).

For OOD inputs: logits are smaller and diffuse → sum of exponentials is smaller → energy is more negative.

We report `-E(x)` (negated) so that higher values indicate more OOD.

### 6.2 Mahalanobis distance

During training, for each class k, compute:
```
μₖ = (1/Nₖ) × Σᵢ:yᵢ=k  φ(xᵢ)          ← class mean feature
Σ  = (1/N) × Σₖ Σᵢ:yᵢ=k  (φ(xᵢ)-μₖ)(φ(xᵢ)-μₖ)ᵀ  ← shared covariance
```

For a test sample x with predicted class ŷ:
```
d_M(x) = √[ (φ(x) - μ_ŷ)ᵀ × Σ⁻¹ × (φ(x) - μ_ŷ) ]
```

This measures how many standard deviations x's features are from the nearest class center, accounting for feature correlations.

---

## Section 7 — Conformal Prediction Mathematics

### 7.1 Nonconformity scores (RAPS)

RAPS (Regularized Adaptive Prediction Sets) constructs scores as follows.

Sort classes in decreasing order of probability:
```
π(1), π(2), ..., π(K)   such that  P(π(1)) ≥ P(π(2)) ≥ ... ≥ P(π(K))
```

The nonconformity score for sample (x, y) is:
```
s(x, y) = Σ_{j : P(π(j)) ≥ P(y)}  P(π(j)) + u × P(y)
```
where u ~ Uniform(0,1) (adds randomness to break ties).

### 7.2 Calibration quantile

On the calibration split of size n:
```
q̂ = Quantile( {s(xᵢ, yᵢ)}ᵢ₌₁ⁿ, level = ⌈(n+1)(1-α)⌉/n )
```

### 7.3 Prediction set

For a new test input x:
```
C_α(x) = { class k :  s(x, k) ≤ q̂ }
```

**Guarantee:**
```
P( y ∈ C_α(x) ) ≥ 1 - α
```

This holds for any joint distribution on (x, y) as long as calibration and test samples are exchangeable.

### 7.4 Mixed-set rule

```
BLOCK_PROCEED(x) = True   if   C_α(x) ∩ Benign ≠ ∅   AND   C_α(x) ∩ Malignant ≠ ∅
                   False  otherwise
```

**What "mixed" means:** the conformal set contains at least one class from the benign group AND at least one from the malignant group. The model cannot commit to either malignancy category. PROCEED is blocked.

---

## Section 8 — Learn-then-Test (LTT)

### 8.1 Setup

Define the **controlled risk** as the malignancy miss rate on auto-proceeded cases:

```
R(λ) = P( g(y) = M  AND  g(ŷ) = B  AND  a*(λ) = PROCEED )
```

Or, conditionally on proceeding:
```
R_cond(λ) = P( g(y) = M AND g(ŷ) = B | a*(λ) = PROCEED )
```

We want to find λ such that R(λ) ≤ α with high probability.

### 8.2 Grid construction

Choose a grid of thresholds ordered from most conservative (small λ, few cases proceed) to most permissive (large λ, more cases proceed):

```
λ₁ < λ₂ < λ₃ < ... < λⱼ
```

For each threshold λⱼ, compute the empirical risk on the calibration split:
```
R̂(λⱼ) = (number of proceeded cases where malignancy was missed) / (total proceeded cases)
```

### 8.3 Statistical testing (Hoeffding-Bentkus bound)

For each λⱼ, compute a p-value for the null hypothesis `H_j : R(λⱼ) > α`:

```
p_j = P( Binomial(n_j, α) ≥ k_j )
```

where:
- `n_j` = number of cases that proceed under λⱼ
- `k_j` = number of malignancy misses among those cases
- The test asks: "if the true risk were α, how likely is it to see this many misses or more?"

### 8.4 Fixed-sequence testing

Walk through the grid from conservative to permissive:

```
for j = 1, 2, ..., J:
    if p_j ≤ δ:           ← null hypothesis rejected → risk ≤ α is plausible
        λ* = λⱼ           ← keep updating the selected threshold
    else:
        break              ← stop at the first non-rejection
```

The result λ* is the most permissive threshold where the statistical test was passed.

### 8.5 The LTT guarantee

**Theorem:** With probability ≥ 1 - δ over the calibration draw:
```
R(λ*) ≤ α
```

**What this means in plain language:**
- We ran calibration once on our calibration set
- There is a 10% chance (δ = 0.10) that we got unlucky and the guarantee is violated
- But with 90% probability, the selected threshold λ* genuinely achieves ≤ 5% malignancy miss rate on proceeded cases

**What this does NOT mean:**
- It does not guarantee this holds on shifted data (that's what DERMA-STRESS measures)
- It does not mean every individual patient has ≤ 5% risk — it's a population-level statement

### 8.6 The guarantee violation under shift

Under Family D (distribution shift) of DERMA-STRESS, we measure:

```
Guarantee violation = R̂_test - α
```

Where R̂_test is the empirical risk on the shifted test set.

- Negative → guarantee holds (empirical risk is actually lower than α)
- Positive → guarantee is broken (the shift pushed risk above α)

We plot this violation as a function of shift severity. This is a **core reported finding**.

---

## Section 9 — Project-Defined Metrics

### 9.1 Acquisition Efficiency AE(r)

At acquisition rate r (fraction of cases where we acquire dermoscopy):

```
AE(r) = [ G_policy(r) - G_random(r) ] / [ G_oracle(r) - G_random(r) ]
```

Where G is the total loss reduction from acquisitions across all N test cases.

- `G_oracle(r)` = if we could perfectly identify the rN cases with the highest benefit
- `G_random(r)` = if we randomly chose rN cases to acquire
- `G_policy(r)` = what our policy actually achieves

**Range:** AE ∈ [0, 1]. AE = 1 means oracle-level selection. AE = 0 means no better than random.

**Worked example:**
```
N = 100 cases, r = 0.2 (acquire for 20 cases)
Oracle selects the 20 cases with highest actual benefit: G_oracle = 8.5
Random acquires 20 random cases: G_random = 2.0
Our policy achieves: G_policy = 6.3

AE(0.2) = (6.3 - 2.0) / (8.5 - 2.0) = 4.3 / 6.5 = 0.66
```

This means our policy recovered 66% of the maximum possible benefit.

### 9.2 Negative Information Rate (NIR)

NIR measures how often adding a modality *hurts*:

```
NIR_m = (1/N) × Σᵢ  𝟙[ L(ŷᵢ^{S∪m}, yᵢ) > L(ŷᵢ^S, yᵢ) ]
```

In plain language: what fraction of the time does adding dermoscopy make the prediction *worse*?

**Why it matters:** dermoscopy is not always helpful. A model that blindly fuses both modalities may actually hurt performance on cases where the dermoscopic image is misleading or poorly acquired.

Also report the **mean magnitude** of the increase when it occurs, because a tiny loss increase and a catastrophic one are very different.

### 9.3 Minimum Sufficient Evidence Set (MSES)

For case i with available evidence set Eᵢ, the MSES is the smallest-cost subset S* ⊆ Eᵢ such that:

```
S* = argmin_{S ⊆ Eᵢ} cost(S)
     subject to: policy(S) = PROCEED  AND  malignancy(ŷ) = malignancy(y)  AND  R_proc(S) ≤ λ*
```

- If no subset qualifies: report "no sufficient subset"
- Compare against baseline: what fraction of cases are sufficient with clinical-only evidence?
- The MSES quantifies how much evidence is actually needed

### 9.4 AUGRC

The Area Under the Generalised Risk-Coverage Curve (Traub et al., NeurIPS 2024) fixes known flaws in AURC.

**Interpretation:** AUGRC equals the average risk of undetected failures — cases where the system proceeds but is wrong. Lower is better.

**Why not AURC?** AURC double-counts some regions of the risk-coverage curve and doesn't penalise confident errors appropriately. AUGRC corrects these issues.

---

## Section 10 — The Final Policy (Complete)

Putting all pieces together:

```
Algorithm EADE(S):
    Input: evidence S, all signal values, threshold λ*, calibration data

    1. Compute p_S(y | xₛ) via temperature-scaled ensemble
    2. Compute H[P̄], MI, JS-divergence, quality scores, E(x), d_M(x)
    3. Compute R_proc(S) = min over ŷ { E_y[L(y, ŷ)] }
    4. Compute conformal set C_α(xₛ)
    5. Check mixed-set rule: flag = BLOCK_PROCEED(xₛ)

    IF flag AND dermoscopy available AND rounds < T_max:
        RETURN ACQUIRE(d), reason=CONF_MIXED

    IF R_proc(S) ≤ λ* AND NOT flag:
        RETURN PROCEED, reason=EXPECTED_LOSS_OK

    FOR each available m in M_avail \ S:
        Compute R_acq(S, m) = R_proc(S) - h_m(xₛ) + c_acq(m)

    m* = argmin_m R_acq(S, m)

    IF R_acq(S, m*) < min(R_proc(S), c_def) AND rounds < T_max:
        RETURN ACQUIRE(m*), reason=EXPECTED_LOSS_GAIN

    RETURN DEFER, reason=INSUFFICIENT_EVIDENCE
```

### Decision tree summary

```
                    Input evidence S arrives
                           │
                    Compute all signals
                           │
                    ┌──────┴──────┐
                    │ Conformal   │
                    │ set mixed?  │
                    └──────┬──────┘
                     Yes   │   No
              ┌────────────┘   └────────────────┐
         ACQUIRE(d)                        R_proc ≤ λ*?
         (if avail.)                            │
         else DEFER                      Yes    │    No
                                    PROCEED     │
                                           Acquire
                                           beneficial?
                                                │
                                         Yes   │   No
                                       ACQUIRE   DEFER
```

---

## Summary of All Equations

| Equation | Formula | Purpose |
|----------|---------|---------|
| Temperature scaling | `P(k) = exp(fₖ/T) / Σ exp(fⱼ/T)` | Calibrate probabilities |
| Ensemble average | `P̄(y) = (1/M) Σⱼ Pⱼ(y)` | Aggregate predictions |
| Predictive entropy | `H[P̄] = -Σ_y P̄(y) log P̄(y)` | Total uncertainty |
| Mutual information | `MI = H[P̄] - (1/M) Σⱼ H[Pⱼ]` | Epistemic uncertainty |
| JS divergence | `JSD = ½KL(P\|\|m) + ½KL(Q\|\|m)` | Cross-modal consistency |
| Energy OOD | `E(x) = -T log Σₖ exp(fₖ/T)` | OOD detection |
| Mahalanobis OOD | `d_M = √[(φ-μ)ᵀΣ⁻¹(φ-μ)]` | OOD detection |
| Expected loss (PROCEED) | `R_proc = min_ŷ Σ_y p_S(y) L(y,ŷ)` | EADE decision |
| Expected loss (ACQUIRE) | `R_acq = R_proc - h_m(xₛ) + c_acq` | EADE decision |
| Expected loss (DEFER) | `R_def = c_def` | EADE decision |
| LTT controlled risk | `R(λ) = P(g(y)=M, g(ŷ)=B, a*=PROCEED)` | Guarantee objective |
| Acquisition efficiency | `AE(r) = (G_policy - G_random)/(G_oracle - G_random)` | Acquisition quality |
| NIR | `NIR_m = (1/N) Σᵢ 𝟙[L(ŷ^{S∪m}) > L(ŷ^S)]` | Modality harm |
| RAPS score | `s(x,y) = Σ_{j:P(π(j))≥P(y)} P(π(j)) + u P(y)` | Conformal score |

---

*Next: [`04_Data_Pipeline.md`](./04_Data_Pipeline.md) — Dataset structure, splits, and preprocessing →*
