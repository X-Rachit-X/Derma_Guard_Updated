# 05 — Architecture Deep Dive
### *Every System Component Explained — Inputs, Outputs, Responsibilities, and Data Flow*

> **Prerequisites:** [`04_Data_Pipeline.md`](./04_Data_Pipeline.md)  
> **Next:** [`06_DERMA_STRESS_Protocol.md`](./06_DERMA_STRESS_Protocol.md)

---

## Full System Data Flow

```
USER ACTION
│
│  "Upload clinical image for case #42"
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│  A. CASE INTAKE                                                      │
│     Create case ID → validate file → store asset → record provenance│
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                         Enqueue background job
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  B. VISION INTELLIGENCE (Member 1)                                   │
│     Run 3-seed ensemble → temperature scale → quality metrics → OOD │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  C. CONSISTENCY MODULE (Member 2)                                    │
│     JS divergence between modalities → contradiction flags           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  D. EVIDENCE MANAGER                                                  │
│     Aggregate all signals → versioned feature record                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                │
               ▼                ▼                │
┌──────────────────┐  ┌──────────────────┐       │
│  E. EADE         │  │  C2. RAG MODULE  │       │
│  (Decision)      │◄─┤  (Explanation)   │       │
└────────┬─────────┘  └──────────────────┘       │
         │                                        │
         ▼                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  F. PERSISTENCE & AUDIT LAYER                                        │
│     Append-only: assessment, model run, audit event                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FRONTEND UI                                                          │
│  Show: action + reason codes + expected losses + conformal set       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component A: Case Intake

### Responsibility
The entry point for all clinical cases. Handles user uploads, validation, and storage.

### Inputs
- Clinical photograph (JPG/PNG/TIFF, ≤ configured size limit)
- Dermoscopic image (optional)
- Structured metadata form (age, sex, site)
- Optional symptom text description

### Outputs
- Stable case ID (UUID)
- Stored asset with metadata record
- Provenance record (who uploaded, when, from what source)
- Missingness record (what is absent)

### Key decisions

**`MISSING` vs `PRESENT_BUT_LOW_QUALITY`**

These are explicitly different states:
```
MISSING                    → Dermoscopy was never taken for this patient
PRESENT_BUT_LOW_QUALITY    → Dermoscopy was taken but is blurry/dark/corrupted

Why it matters:
  MISSING → "Acquire" is possible if dermoscopy is feasible
  PRESENT_BUT_LOW_QUALITY → "Re-acquire" with better technique, or DEFER
```

**Validation checks:**
1. File type: only allow image types (JPEG, PNG, TIFF, DICOM subset)
2. File size: reject oversized uploads
3. Image decode check: try to open the image; reject corrupt files
4. Content hash: compute SHA-256 for duplicate detection

### Database entity: `EvidenceAsset`
```
id              UUID primary key
case_id         → PatientCase
asset_type      ENUM: clinical | dermoscopy | metadata | text
storage_key     path/filename in object storage
provenance      source, uploader_id, upload_method
uploaded_at     timestamp
quality_status  ENUM: UNKNOWN | PRESENT | PRESENT_BUT_LOW_QUALITY | MISSING
quality_scores  JSON: {blur: 0.73, exposure: 0.91, resolution: 0.88, ...}
content_hash    SHA-256 of file content
```

---

## Component B: Vision Intelligence

**Owner:** Member 1

### Responsibility
Run the image classifiers and produce all signal values needed by the Evidence Manager.

### Model architecture

```
Three parallel branches (one per modality configuration):

┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ Clinical Branch    │   │ Dermoscopy Branch  │   │ Fused Branch       │
│ ConvNeXt-Tiny +   │   │ ConvNeXt-Tiny +   │   │ Both encoders +   │
│ classification    │   │ classification    │   │ metadata MLP +    │
│ head              │   │ head              │   │ late fusion head  │
└────────────────────┘   └────────────────────┘   └────────────────────┘
         ↓                        ↓                        ↓
  logits_c [K]           logits_d [K]           logits_fused [K]
         ↓                        ↓                        ↓
  temperature_c          temperature_d          temperature_fused
         ↓                        ↓                        ↓
     P_c [K]                P_d [K]               P_fused [K]
```

### 3-seed ensemble

Each branch is trained 3 times with different random seeds:

```
For branch B (e.g., clinical-only):
  Seed 0 → P_B^0   calibrated posterior
  Seed 1 → P_B^1
  Seed 2 → P_B^2
  
  Ensemble: P̄_B = (P_B^0 + P_B^1 + P_B^2) / 3
  Entropy:  H[P̄_B]
  MI:       H[P̄_B] - (1/3)(H[P_B^0] + H[P_B^1] + H[P_B^2])
```

### Modality dropout (training trick)

During training, for each batch item, randomly set one or more modalities to a zero tensor:

```
p_drop_dermoscopy = 0.3   # 30% of training samples see no dermoscopy
p_drop_metadata   = 0.2   # 20% see no metadata
p_drop_clinical   = 0.1   # 10% see no clinical (rare; used to strengthen derm branch)
```

This forces the model to be robust to missing inputs. Without this, the fused model often collapses to ignoring one modality.

### Quality estimation

```python
# Blur score (Laplacian variance)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

# Exposure score
mean_brightness = np.mean(image)
exposure_score = 1 - abs(mean_brightness/255 - 0.5) * 2   # 0=extreme, 1=ideal

# Resolution score
h, w = image.shape[:2]
resolution_score = min(1.0, (h * w) / (224 * 224))
```

### OOD computation

Both energy score and Mahalanobis distance are computed using the penultimate layer of each branch. Class means and covariance are computed on the training set after final training.

### Outputs (per assessment)

```python
{
    "clinical_posterior": [p1, p2, ..., pK],        # per-seed and ensemble avg
    "dermoscopy_posterior": [...],
    "fused_posterior": [...],
    "entropy_clinical": 0.73,
    "entropy_dermoscopy": 0.81,
    "mutual_info_clinical": 0.12,
    "mutual_info_dermoscopy": 0.19,
    "energy_ood_clinical": -22.4,
    "mahalanobis_clinical": 3.7,
    "quality_blur": 0.82,
    "quality_exposure": 0.91,
    "quality_resolution": 1.0,
    "model_version": "v1.2.0",
    "ensemble_seed": [0, 1, 2],
    "temperature": {"clinical": 1.3, "dermoscopy": 1.1, "fused": 1.2},
    "runtime_ms": 340,
    "status": "SUCCESS"
}
```

### Database entity: `ModelRun`
```
id                  UUID
assessment_id       → Assessment
model_name          "convnext_tiny_clinical"
model_version       "v1.2.0"
ensemble_seed       0 | 1 | 2
preprocessing_version "v1.0"
temperature         1.3
output_scores       JSON: all posteriors
entropy             0.73
mutual_information  0.12
ood_score           3.7
runtime_ms          340
status              SUCCESS | FAILED | TIMEOUT
error_details       null | "timeout after 10s"
```

---

## Component C: Consistency Module

**Owner:** Member 2

### Two sub-modules

```
C1. Consistency module    → FEEDS into the decision (Evidence Manager → EADE)
C2. RAG explanation module → Does NOT feed the decision (context only)
```

### C1. Consistency module

**Purpose:** Quantify whether clinical image and metadata/text agree on the likely diagnosis.

**Inputs:**
- `P_c` = clinical model posterior (from Vision Intelligence)
- `P_z` = metadata-based posterior (if a metadata-only classifier exists, or approximated from clinical + metadata)
- Structured metadata fields
- Extracted symptom text

**Outputs:**
- `js_divergence`: JSD between P_c and P_z (Section 5.1 in Math doc)
- `contradiction_flags`: list of specific contradictions detected

**Contradiction detection examples:**

```python
flags = []

# Example 1: Site mismatch
if extracted_text_mentions("on the back") and metadata["site"] == "face":
    flags.append({"type": "site_mismatch", "severity": "high"})

# Example 2: Age mismatch  
if "elderly patient" in extracted_text and metadata["age_bin"] == "15-19":
    flags.append({"type": "age_mismatch", "severity": "medium"})

# Example 3: Colour description mismatch
if "pale lesion" in extracted_text and clinical_hue_analysis() == "dark":
    flags.append({"type": "color_mismatch", "severity": "low"})
```

**Symptom extraction:**

LLM with schema-constrained output (or rule-based fallback):

```python
# Schema for extraction
class SymptomExtraction(BaseModel):
    duration_weeks: Optional[int]
    growth_reported: Optional[bool]
    itch_reported: Optional[bool]
    bleeding_reported: Optional[bool]
    site_mentioned: Optional[str]
    color_descriptors: List[str]
```

The extractor must never read ground-truth labels from the text. Use only patient-reported symptoms.

### C2. RAG Explanation Module

**Purpose:** Provide source-grounded context for the model's assessment.

**Document corpus:**
- Curated public dermatology guideline summaries (e.g., AAD guidelines)
- ISIC educational materials
- Dermatoscopy atlas excerpts (verify licences for each source)

**Retrieval pipeline:**

```
Query: "dermoscopic features of {predicted_class}"
           │
           ▼
┌─────────────────────┐      ┌─────────────────────┐
│ BM25 lexical search │      │ pgvector semantic   │
│ (rank_bm25)        │      │ search              │
└──────────┬──────────┘      └──────────┬──────────┘
           │                             │
           └────────────┬───────────────┘
                        ▼
               ┌──────────────────┐
               │ Fusion + rerank  │
               │ (RRF or cross-  │
               │  encoder)        │
               └────────┬─────────┘
                        ▼
               Top 3-5 passages
               with source metadata
```

**Source attribution:**
- Every generated claim is linked to a specific passage
- Passages that are not retrieved: flagged as "uncited"
- Passages that don't support the claim: flagged as "unsupported"
- Never cite a source that was not retrieved

**Critical constraint:** The RAG module's output is labelled as "supporting context, not clinical validation." It never changes the PROCEED/ACQUIRE/DEFER decision.

### Database entity: `ConsistencyRecord`
```
id                  UUID
assessment_id       → Assessment
js_divergences      JSON: {"clinical_vs_metadata": 0.29, ...}
contradiction_flags JSON: [{"type": "site_mismatch", "severity": "high"}]
extractor_version   "v1.0"
```

### Database entity: `EvidenceSource`
```
id               UUID
assessment_id    → Assessment
document_id      reference to document in corpus
source_url       URL of the original document
passage_reference paragraph/section identifier
retrieval_score  BM25 score or vector similarity
index_version    "v1.2" (corpus version)
supported        true | false | null
```

---

## Component D: Evidence Manager

### Responsibility
Aggregates all signals from Vision Intelligence and the Consistency Module into a single, versioned feature record for each decision. Acts as the central hub.

### Inputs
- All Vision Intelligence outputs (posteriors, entropy, MI, OOD, quality)
- Consistency module outputs (JS divergence, contradiction flags)
- Missingness record (from Case Intake)
- All model and calibration version numbers

### Outputs
- Versioned feature record (a structured snapshot of all available evidence signals)
- Log of which signals were unavailable and why

### Key design principle: preserve raw values

The Evidence Manager does **not** collapse signals into a single "trust score." It preserves individual signal values so that:
1. Ablation studies can remove individual signals
2. The EADE can weight signals according to the policy
3. Audit records show exactly what was observed

### What gets aggregated

```python
FeatureRecord = {
    # Vision signals
    "clinical_posterior": [...],
    "dermoscopy_posterior": [...] if available else None,
    "fused_posterior": [...] if available else None,
    "entropy": {"clinical": 0.73, "dermoscopy": 0.81},
    "mutual_info": {"clinical": 0.12},
    "ood_energy": {"clinical": -22.4, "dermoscopy": -19.1},
    "ood_mahalanobis": {"clinical": 3.7},
    "quality": {"blur": 0.82, "exposure": 0.91},
    
    # Consistency signals
    "js_divergence": {"clinical_vs_metadata": 0.29},
    "contradiction_flags": [{"type": "site_mismatch"}],
    
    # Missingness
    "missingness": {
        "dermoscopy": "MISSING",
        "metadata.sex": "PRESENT",
        "metadata.age": "PRESENT"
    },
    
    # Versions
    "model_version": "v1.2.0",
    "calibration_version": "v1.0",
    "feature_record_version": "v1.0",
    
    # Unavailability log
    "unavailable_signals": [
        {"signal": "dermoscopy_posterior", "reason": "dermoscopy MISSING"}
    ]
}
```

---

## Component E: EADE (Evidence-Adaptive Decision Engine)

### Responsibility
Apply the pre-registered decision policy to produce an action.

### Inputs
- Feature record from Evidence Manager
- LTT-calibrated threshold λ* (loaded from calibration artefact)
- Conformal calibration data q̂ (loaded from calibration artefact)
- Cost matrix (loaded from pre-registration file)
- Acquisition-value estimator h_d (trained model)
- Available modalities for this case

### Algorithm (see also Algorithm 1 in the IEEE paper)

```
1. Load calibrated posteriors from feature record
2. Compute R_proc(S) using cost matrix
3. Construct conformal set C_α(xₛ)
4. Check mixed-set rule
5. Check if R_proc ≤ λ* (LTT threshold)
6. Compute R_acq for each available missing modality
7. Select action with lowest expected loss
8. Apply maximum-rounds check
9. Record action + reason codes + all expected losses
```

### Outputs (stored in Assessment record)

```python
{
    "action": "ACQUIRE",
    "action_target": "dermoscopy",
    "reason_codes": ["CONF_MIXED", "EXPECTED_LOSS_GAIN"],
    "conformal_set": [1, 4, 7],   # class indices
    "expected_losses": {
        "proceed": 3.2,
        "acquire_dermoscopy": 1.8,
        "defer": 1.5
    },
    "ltt_threshold": 0.73,
    "policy_version": "v1.0.0",
    "calibration_version": "v1.0",
    "cost_matrix_version": "v1.0"
}
```

### Version tracking

Every single version is recorded:
- `policy_version`: the EADE logic version (increment if decision rules change)
- `calibration_version`: the LTT threshold and conformal quantile version
- `cost_matrix_version`: the values of ℓ_miss, ℓ_fa, c_acq, c_def

If any of these change, old assessments remain valid (they are stamped with their own versions) but new assessments use the new versions.

---

## Component F: Persistence and Audit Layer

**Owner:** Member 3

### Design principle: append-only

**Never overwrite or delete** assessment records. If a human reviewer corrects the model output, the correction is stored as a separate `ReviewDecision` record, not by modifying the original `Assessment`.

This is essential for:
- Scientific reproducibility: you can always recover exactly what the model predicted and why
- Audit trails: regulators and researchers can see the sequence of events
- Trust: the system cannot hide its mistakes

### Core entities and their relationships

```
PatientCase ─────────────────────────────────────────────────────┐
    │                                                              │
    ├── EvidenceAsset (many)                                       │
    │       └── file, quality, hash                               │
    │                                                              │
    ├── EvidenceSnapshot (many)                                    │
    │       └── which assets were present at time of assessment   │
    │                                                              │
    ├── Assessment (many, append-only)                            │
    │       ├── linked to one EvidenceSnapshot                    │
    │       ├── policy_version, calibration_version               │
    │       ├── action, reason_codes                              │
    │       └── ModelRun (many, one per model branch per seed)    │
    │               └── posteriors, entropy, OOD, runtime         │
    │                                                              │
    ├── ConsistencyRecord (one per Assessment)                     │
    │                                                              │
    ├── EvidenceSource (many per Assessment)                       │
    │       └── RAG citations                                      │
    │                                                              │
    ├── ReviewDecision (many)                                      │
    │       └── reviewer + decision + rationale (never overwrites)│
    │                                                              │
    ├── WorkflowJob (many)                                         │
    │       └── job status, errors, retries                        │
    │                                                              │
    └── AuditEvent (many)                                         │
            └── every state transition, who triggered it          │
                                                                  │
ExperimentRun ────────────────────────────────────────────────────┘
    └── DERMA-STRESS experiment records
```

### AuditEvent types

```
CASE_CREATED
ASSET_UPLOADED
ASSET_VALIDATION_FAILED
ASSESSMENT_REQUESTED
JOB_ENQUEUED
JOB_STARTED
MODEL_RUN_COMPLETED
MODEL_RUN_FAILED
EVIDENCE_SNAPSHOT_CREATED
ASSESSMENT_COMPLETED
ACTION_PROCEED
ACTION_ACQUIRE
ACTION_DEFER
REVIEW_SUBMITTED
WORKER_RESTARTED
```

### The timeline view

The frontend shows a chronological list of all AuditEvents for a case. A user can:
- See exactly what evidence was available at each assessment
- See why the policy chose each action
- See when a reviewer made a correction
- See any worker failures and retries

---

## Component G: DERMA-STRESS Runner

**Owner:** Member 2

### Responsibility
Apply stress conditions from DERMA-STRESS families A-G and run all baselines and EADE on identical cases.

### Flow

```
1. Load frozen split (train/val/cal/test — already determined)
2. Load a stress configuration YAML:
   {family: A, type: gaussian_blur, severity: 2, seed: 42}
3. Apply the corruption to the specified split
4. Run ALL baselines (B1-B8) and EADE on the same corrupted cases
5. Save predictions and policy actions to disk
6. Compute all metrics and confidence intervals
7. Write JSON result file + Markdown report
8. Create ExperimentRun record in the database
```

### Critical: identical cases

All methods must process **the exact same set of cases** under the exact same corruption. If one method skips a case (e.g., due to missing-modality handling), this must be recorded and reported.

### YAML configuration format

```yaml
family: F
description: "Combined failure: degraded clinical + missing metadata"
components:
  - type: gaussian_blur
    target: clinical
    severity: 2
    sigma: 3.0
  - type: missing_metadata
    fields: [age, sex]
seed: 42
dataset: MILK10k
split_version: v1-lesion-60-15-10-15
model_versions:
  clinical: "v1.2.0"
  dermoscopy: "v1.2.0"
  fused: "v1.2.0"
policy_version: "v1.0.0"
```

### Per-run outputs

```
experiment_results/
├── config.yaml              # exact configuration used
├── config_hash.txt          # SHA-256 of config for reproducibility
├── predictions/
│   ├── B1_predictions.csv   # per-case predictions for each baseline
│   ├── B2_predictions.csv
│   ├── ...
│   └── EADE_predictions.csv
├── actions/
│   └── EADE_actions.csv     # PROCEED/ACQUIRE/DEFER per case
├── metrics/
│   ├── metrics.json         # all metrics with CIs
│   └── report.md            # human-readable summary
└── environment.json         # software versions, GPU, date
```

---

## Background Jobs (WorkflowJob)

### Why background jobs?

Model inference takes seconds to minutes. The user cannot wait for the API response. Instead:
1. API creates a case, returns case ID immediately
2. API enqueues a background job
3. Worker picks up the job and runs inference
4. UI polls for job completion (or uses WebSocket push)

### Job states

```
PENDING → RUNNING → COMPLETED
                 → FAILED → PENDING (retry)
                 → TIMEOUT → FAILED
```

### Retry logic

```
Max retries: 3
Retry delay: exponential backoff (1s, 4s, 16s)
On final failure: set status=FAILED, store error_details
                  case remains fully inspectable
```

**Key requirement:** a failed job must leave the case in a fully inspectable state. The UI must show the error clearly. The case must not be in a broken half-processed state.

---

## Frontend Pages Summary

| Page | Key Features |
|------|-------------|
| **Overview** | Case counts, job queue states, recent experiments |
| **Case List** | Filter by status, date, action type |
| **Case Creation** | Metadata form + file upload with quality preview |
| **Case Detail** | Evidence assets + quality badges + current action |
| **Assessment Detail** | Calibrated posteriors, conformal set visual, expected-loss bars for all three actions, reason codes |
| **Acquisition Screen** | Explains what is being requested and estimated benefit |
| **Case Timeline** | Chronological list of all events with exact evidence snapshots |
| **DERMA-STRESS Lab** | Config picker, risk-coverage plots, AUGRC table, guarantee-violation chart, AE chart |
| **Evidence Sources** | Retrieved passages, support status, document links |
| **Admin** | Model/policy/calibration versions, access control |

### Critical UI constraint

> The UI must **never** imply that a confident score, a heatmap, or a citation proves clinical correctness.

Every assessment is accompanied by explicit limitation statements and reason codes. The system banner always reads: "Research Prototype — Not a Medical Device."

---

## Roles and Authorization

| Role | Permissions |
|------|------------|
| **Researcher/Admin** | Configure models, datasets, experiments, all settings |
| **Case Operator** | Create cases, upload evidence, request assessments |
| **Reviewer** | View cases, record review decisions |
| **Viewer** | View authorized cases and experiment results only |

Authorization checks exist at the **backend API level** for every case operation. A frontend-only role check is not sufficient.

---

*Next: [`06_DERMA_STRESS_Protocol.md`](./06_DERMA_STRESS_Protocol.md) — The 7 stress families explained →*
