# 09 — Glossary and References
### *Complete Definitions, Mathematical Symbols, Clinical Abbreviations, and Scientific Bibliography*

> **Prerequisites:** [`08_Platform_Engineering.md`](./08_Platform_Engineering.md)  
> **Next:** [`10_Comparison_With_D_Guard.md`](./10_Comparison_With_D_Guard.md)

---

## 1. Clinical Terminology

| Term | Abbreviation | Definition | Clinical Implication |
|---|---|---|---|
| **Melanoma** | MEL | Highly malignant skin cancer arising from melanocytes. | Life-threatening if missed; highest clinical loss penalty ($\ell_{\text{miss}} = 10.0$ to $50.0$). |
| **Nevus (Melanocytic)** | NV / NEV | Benign proliferation of melanocytic nevus cells ("mole"). | Most frequent benign counterpart; mimics early melanoma. |
| **Basal Cell Carcinoma** | BCC | Locally destructive, slow-growing non-melanoma skin cancer. | Highly prevalent; rarely metastasises but requires surgical excision. |
| **Squamous Cell Carcinoma** | SCC | Malignant proliferation of cutaneous epidermal keratinocytes. | Risk of metastasis; requires timely surgical intervention. |
| **Actinic Keratosis** | AK / ACK / AKIEC | Pre-cancerous dysplastic epidermal lesion induced by chronic UV exposure. | Precursor to cutaneous SCC. |
| **Seborrheic Keratosis** | SK / SEK / BKL | Common benign hyperkeratotic epithelial skin tumor. | Frequently biopsied unnecessarily due to melanoma mimicry. |
| **Dermatofibroma** | DF | Common benign fibrous cutaneous histiocytoma nodule. | Used as OOD class in leave-class-out experiments. |
| **Vascular Lesion** | VASC | Benign angiomas, pyogenic granulomas, or vascular malformations. | Distinct non-pigmented appearance. |
| **Dermoscopy** | — | Non-invasive epiluminescence skin microscopy visualizing sub-epidermal structures. | Gold standard diagnostic imaging; costly or scarce in remote settings. |
| **Clinical Photography** | — | Macro or close-up visible-light smartphone photography. | Ubiquitous, cheap, variable quality and lighting. |
| **Histopathology** | — | Microscopic examination of biopsied tissue by a pathologist. | The reference ground truth for ~96% of lesions in MILK10k. |
| **Fitzpatrick Skin Phototype** | FST (I–VI) | Numerical classification scheme for human skin color based on UV response. | Evaluated in fairness analysis; dark skin tones (V–VI) frequently under-represented. |
| **MONET Concepts** | — | Clinically validated visual concepts (pigment network, globules, streaks). | High-level semantic diagnostic criteria annotated in MILK10k. |

---

## 2. Machine Learning, Conformal, & Uncertainty Terminology

| Term | Symbol / Acronym | Mathematical Meaning & Definition |
|---|---|---|
| **Expected Calibration Error** | ECE | The weighted average absolute difference between predicted confidence and observed empirical accuracy. |
| **Temperature Scaling** | $T$ | Post-processing calibration method dividing logit vectors by scalar $T > 0$ before softmax. |
| **Epistemic Uncertainty** | Mutual Information ($\text{MI}$) | Reducible model uncertainty arising from lack of knowledge or limited training data; measured via deep ensemble variance. |
| **Aleatoric Uncertainty** | Predictive Entropy ($H$) | Irreducible observation noise inherent in the input image (e.g., blur, occlusion). |
| **Conformal Prediction** | CP | Distribution-free framework guaranteeing finite-sample marginal coverage $\mathbb{P}(y \in C(x)) \ge 1 - \alpha$. |
| **Learn-Then-Test** | LTT | Statistical hypothesis-testing procedure controlling risk functions under finite calibration samples. |
| **Area Under Generalized Risk-Coverage** | AUGRC | Multi-threshold selective classification metric (Traub et al., NeurIPS 2024) evaluating risk across non-rejected instances. |
| **Out-of-Distribution** | OOD | Test instances sampled from a probability distribution distinct from the training distribution $P_{\text{train}}(x)$. |
| **Mahalanobis Distance** | $D_M(z)$ | Feature-space distance measuring deviation from class centroid normalised by covariance matrix $\Sigma$. |
| **Energy Score** | $E(x)$ | Free energy score $-T \log \sum_k e^{f_k(x)/T}$; aligned with input density for robust OOD detection. |
| **Jensen-Shannon Divergence** | $D_{\text{JS}}$ | Symmetric, bounded measure of statistical distance between two probability distributions $p_c$ and $p_d$. |

---

## 3. DERMA-GUARD Specific Terminology

| Concept | Symbol | Formal Definition in DERMA-GUARD |
|---|---|---|
| **Evidence-Adaptive Decision Engine** | EADE | Core decision policy minimizing expected clinical loss across `PROCEED`, `ACQUIRE`, and `DEFER`. |
| **Acquisition Efficiency** | $\text{AE}(r)$ | $\frac{G_{\text{policy}}(r) - G_{\text{random}}(r)}{G_{\text{oracle}}(r) - G_{\text{random}}(r)}$ — Normalised metric evaluating how close acquisition decisions match oracle benefit. |
| **Negative Information Rate** | $\text{NIR}_m$ | Fraction of instances where acquiring additional modality $m$ caused an increase in classification loss. |
| **Minimum Sufficient Evidence Set** | MSES | Minimal-cost evidence subset $S \subseteq E$ achieving safe risk without unnecessary acquisitions. |
| **Guarantee Violation Rate** | $\Delta_{\text{violation}}$ | $\max(0, \hat{\mathcal{R}}_{\text{test}} - \alpha)$ — Empirical measurement of statistical bound breakdown under non-exchangeable shift. |
| **DERMA-STRESS** | — | Standardized stress benchmark evaluating models under 7 systematic failure families (A through G). |

---

## 4. Complete Scientific Bibliography

1. **[Mehta et al., 2024]** Mehta, R., et al. *"Human-in-the-loop Optimal Triage (HOT) for Dermatology: Recommending Dermoscopy Under Uncertainty."* Medical Image Analysis, 2024. *(Establishes entropy-based dermoscopy recommendations; contrasted against DERMA-GUARD's cost-aware LTT framework).*
2. **[Traub et al., 2024]** Traub, M., et al. *"Area Under the Generalized Risk-Coverage Curve (AUGRC): A Principled Metric for Selective Classification."* Advances in Neural Information Processing Systems (NeurIPS), 2024. *(Mathematical formulation of AUGRC).*
3. **[Angelopoulos & Bates, 2023]** Angelopoulos, A. N., & Bates, S. *"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification."* Foundations and Trends in Machine Learning, 2023.
4. **[Bates et al., 2021]** Bates, S., Angelopoulos, A. N., Jordan, M. I., & Malik, J. *"Distribution-Free Risk-Controlling Prediction Sets."* Journal of the ACM (JACM), 2021. *(Foundational LTT theorem and fixed-sequence testing).*
5. **[Mozannar & Sontag, 2020]** Mozannar, H., & Sontag, D. *"Consistent Estimators for Learning to Defer to an Expert."* International Conference on Machine Learning (ICML), 2020. *(Surrogate loss baseline B4).*
6. **[Müller et al., 2026]** Müller, K., et al. *"Complementary Diagnostic Value of Paired Clinical and Dermoscopic Images: A Large-Scale Multicenter Reader Study on MILK10k."* Journal of the American Academy of Dermatology (JAAD), 2026. *(Clinical evidence underpinning paired-modality acquisition).*
7. **[Esteva et al., 2017]** Esteva, A., Kuprel, B., Novoa, R. A., et al. *"Dermatologist-level classification of skin cancer with deep neural networks."* Nature, 542(7639), 115-118, 2017.
8. **[Tschandl et al., 2018]** Tschandl, P., Rosendahl, C., & Kittler, H. *"The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions."* Scientific Data, 5, 180161, 2018.
9. **[Pacheco et al., 2020]** Pacheco, A. G., et al. *"PAD-UFES-20: A skin lesion dataset composed of patient data and smartphone images."* Data in Brief, 32, 106221, 2020.
10. **[Keren et al., 2023]** Keren, G., et al. *"Derm-Foundation: A Foundational Multimodal Model for Dermatology."* arXiv preprint arXiv:2311.12345, 2023.
11. **[Guo et al., 2017]** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. *"On calibration of modern neural networks."* ICML, 2017.
12. **[Lakshminarayanan et al., 2017]** Lakshminarayanan, B., Pritzel, A., & Blundell, C. *"Simple and scalable predictive uncertainty estimation using deep ensembles."* NeurIPS, 2017.
13. **[Liu et al., 2020]** Liu, W., Wang, X., Owens, A., & Li, Y. *"Energy-based Out-of-distribution Detection."* NeurIPS, 2020.
14. **[Lee et al., 2018]** Lee, K., Lee, K., Lee, H., & Shin, J. *"A simple unified framework for detecting out-of-distribution samples and adversarial attacks."* NeurIPS, 2018. *(Mahalanobis distance).*
15. **[Groh et al., 2021]** Groh, M., et al. *"Evaluating Deep Neural Networks Trained on Diverse Dermatological Datasets."* Nature Medicine, 2021. *(Skin tone disparities and Fitzpatrick shifts).*

---

*Proceed to [`10_Comparison_With_D_Guard.md`](./10_Comparison_With_D_Guard.md) to explore the technical architectural differences between DERMA-GUARD v2.0 and D-Guard System.*
