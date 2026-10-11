# 📄 DERMA-GUARD IEEE Research Paper

This directory contains the complete IEEE-format research paper source for **DERMA-GUARD**.

- File: [`derma_guard_ieee_paper.tex`](./derma_guard_ieee_paper.tex)
- Format: Standard double-column `IEEEtran` document class.
- Status: Complete Overleaf-ready draft with 29 formal academic citations, comprehensive baseline comparisons, mathematical formulations, and experimental tables.

---

## 🚀 How to Compile in Overleaf

1. Go to [Overleaf](https://www.overleaf.com) and click **New Project** → **Upload Project** (or **Blank Project**).
2. Copy the content of [`derma_guard_ieee_paper.tex`](./derma_guard_ieee_paper.tex) into your `main.tex` (or upload the `.tex` file directly).
3. Under Overleaf **Menu** (top left):
   - **Compiler:** Select `pdfLaTeX` (or `XeLaTeX`).
   - **TeX Live version:** `2023` or later.
4. Click **Recompile**.

---

## 📋 Paper Sections Summary

| Section | Title | Key Contents |
|---|---|---|
| **I** | Introduction | Clinical motivation, smartphone screening dilemma, failure of forced-choice AI |
| **II** | Related Work | Taxonomy of selective prediction, active feature acquisition, conformal risk control |
| **III** | System Formulation | Mathematical derivation of expected-loss policy, value estimator, LTT calibration |
| **IV** | Dataset and Protocol | MILK10k cohort, leak-free lesion-level splitting, 11-class diagnostic groupings |
| **V** | DERMA-STRESS Benchmark | 7 perturbation families (A–G), guarantee violation formalization |
| **VI** | Experimental Results | Evaluation against baselines B1–B8, AUGRC, AE, fairness analysis across skin tones |
| **VII** | Ablation Study | 10 ablation conditions isolating LTT, conformal mixed-set rule, and value estimator |
| **VIII** | Discussion & Limitations | Clinical boundaries, exchangeability violations, failure disclosure |
| **IX** | Conclusion | Summary of contributions and future active sensing directions |

---

## ⚠️ Notes Before Camera-Ready Submission

- Replace placeholder values (`--.-`) in experiment result tables once final test runs complete.
- Populate author names, institutional affiliations, and grant acknowledgments.
- Lock git commit hash and `prereg.yaml` checksum before reporting test split evaluations.
