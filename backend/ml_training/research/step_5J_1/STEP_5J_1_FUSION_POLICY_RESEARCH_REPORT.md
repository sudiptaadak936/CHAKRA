# CHAKRA STEP 5J.1: RISK & ML FUSION POLICY RESEARCH REPORT

**Date**: 2026-09-30  
**Research Verdict**: `INDEPENDENT CHANNELS EMPIRICALLY PREFERRED FOR POLICY SAFETY`  
**Execution Context**: Local Python research environment (`backend/ml_training/research/step_5J_1/`)  

---

## 1. Executive Verdict

A rigorous, local statistical investigation was conducted using strictly the 84-record development population (across 44 independent entities) to evaluate whether an empirical mathematical fusion formula combining deterministic risk scores ($S_{\text{det}} \in [0, 100]$) and calibrated address-level ML probabilities ($p_{\text{ML}} \in [0, 1]$) is statistically defensible.

**Empirical Finding**:
In nested entity-grouped cross-validation across all outer folds and all tested fusion families (additive, convex ensemble, bounded ML, and maximum-floor), the optimal learned weight $\alpha$ for ML score contribution collapsed identically to $0.0$. 

Because 100% of positive development records are sanctioned entities ($S_{\text{det}} = 100.0$) and 100% of negative development records are clean entities ($S_{\text{det}} = 0.0$), the deterministic risk policy already achieves an empirical separation benchmark of $\text{ROC-AUC} = 1.000$ on development data. Adding any positive ML probability contribution to the deterministic score strictly penalizes verified negative entities (elevating false-positive scores above zero) while providing zero upward separation for positives (which are already clamped at the 100.0 ceiling).

**Final Research Conclusion**:
`INDEPENDENT CHANNELS EMPIRICALLY PREFERRED FOR POLICY SAFETY`  
Numerical fusion adds no stable incremental discriminative information and introduces false-positive elevation on negative entities. The independent-channel architecture—where deterministic risk evidence and calibrated ML analytical signals are presented side-by-side without synthetic weighting—is empirically and methodologically preferred.

---

## 2. Research-Only Scope

* This study is purely statistical and analytical research.
* **No production fusion policy was implemented.**
* **No frozen Step 5H.7 or Step 5I artifact was modified.**
* The research was executed locally without external cloud or external network dependencies.

---

## 3. Current Frozen Architecture

The production environment remains locked under:
* **Step 5 Deterministic Risk Engine**: Point-based evidence engine (`backend/app/forensics/risk_engine.py`, `risk_policy.py`).
* **Step 5H.7 Frozen Artifacts**: Final Logistic Regression (`569bbca1...`), Final XGBoost (`e320eb08...`), Preprocessor (`8b6a44e1...`), Sigmoid LR Calibrator (`1e78b493...`), Sigmoid XGB Calibrator (`90d61a4a...`).
* **Step 5I Inference Service**: Read-only probability inference with non-punitive semantics (`backend/app/forensics/address_ml_service.py`).
* **Feature Contract**: Schema `1.1.0` with 13 canonical features in fixed position-invariant order.

---

## 4. Historical Specifications Explicitly Non-Authoritative

* **Step 5F Transaction ML Ensemble** (`backend/ml_training/fusion.py`): Explicitly restricted to Elliptic transaction research. It is banned from address-level production risk scoring.
* **Step 5G Research Proposal** (`STEP_5G_REPORT.md` Section 15): The proposal `max(deterministic_base_score, ml_risk_probability * ML_POLICY_WEIGHT)` was an unparameterized preliminary concept. It predated address-level dataset creation, calibration, and model freeze, and has no frozen status.
* Historical Isolation Forest and GNN transaction-level contracts remain completely excluded.

---

## 5. Dataset Provenance & Firewall Verification

* **Development Population**:
  * Exactly 84 address-level records.
  * Exactly 44 independent entities.
  * Class Distribution: 66 positive records (26 entities), 18 negative records (18 entities).
* **Final Test Firewall**:
  * `The 9-entity final test set was not accessed or used for policy selection.`
  * The 9 test entities (`wikileaks_org_01`, `kraken_vasp_01`, `bybit_vasp_01`, `crypto_com_vasp_01`, `47682`, `47635`, `48432`, `43421`, `52071`) were quarantined with zero record or entity overlap ($44 \cap 9 = \emptyset$).

---

## 6. Deterministic & ML Alignment

Alignment was established using exact blockchain normalized addresses, network contexts, and the authoritative SDN crypto database (`data/ofac/processed/sdn_crypto_addresses.json`):
* **66 Positive Records**: All 66 addresses matched authoritative OFAC SDN listings with verified feature IDs. Evaluated through `RiskEngine.evaluate(...)`, all 66 records produced:
  - `overall_score = 100.0`
  - `risk_level = CRITICAL`
  - `component = SANCTIONS (100.0 pts)`
* **18 Negative Records**: Zero sanctions hits, zero typologies. Evaluated through `RiskEngine.evaluate(...)`, all 18 records produced:
  - `overall_score = 0.0`
  - `risk_level = NEUTRAL`
  - `components = []`

---

## 7. Nested Entity-Grouped Methodology

To prevent data leakage and avoid optimizing on calibration OOF data:
* **Outer Split**: `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)` across the 44 entities.
  - Ensures every outer validation entity is absent from outer training.
* **Inner Split**: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)` on outer training entities.
  - Generates inner OOF raw predictions.
  - Fits inner sigmoid Platt scaling using entity-balanced weights $w_i = 1 / N_{\text{records\_for\_entity}}$.
  - Optimizes candidate fusion parameters strictly on inner data.
* **Outer Evaluation**: The candidate policy selected on inner data is applied once to the held-out outer validation entities.

---

## 8. Typology Overlap Study

* **Counterfactual Sensitivity**:
  - The frozen production models were evaluated on the 84 dev records using standard 13-feature inputs, and counterfactually by forcing the 4 structural typology flags (`peel_chain`, `rapid_hop`, `fan_in`, `fan_out`) to zero.
  - Results:
    $$\Delta\text{LR}_{\text{raw}} = 0.0000, \quad \Delta\text{LR}_{\text{cal}} = 0.0000$$
    $$\Delta\text{XGB}_{\text{raw}} = 0.0000, \quad \Delta\text{XGB}_{\text{cal}} = 0.0000$$
  - **Explanation**: All 84 development records had empirical values of $0.0$ for all 4 typology features (verified by `zero_variance_indices = [9, 10, 11, 12]` in `final_preprocessor_params.json`).
* **Prospective Overlap Risk**:
  - In prospective operational data where typologies fire, an additive fusion model would double-count evidence: typologies contribute 15–25 points in `RiskEngine` and also serve as positive features in the ML models.

---

## 9. Candidate Fusion Families Tested

1. **Candidate A (Baseline)**: Independent Channels (no composite score).
2. **Candidate B (Additive)**: $S = \min(100, S_{\text{det}} + \alpha \cdot p_{\text{ML}})$ with $\alpha \in [0, 100]$. Tested separately for LR and XGB.
3. **Candidate C (Convex Ensemble)**: $p_{\text{ML}} = w \cdot p_{\text{LR}} + (1-w) \cdot p_{\text{XGB}}$, $S = \min(100, S_{\text{det}} + \alpha \cdot p_{\text{ML}})$.
4. **Candidate D (Bounded ML)**: $S = \min(100, S_{\text{det}} + \min(C_{\text{max}}, \alpha \cdot p_{\text{ML}}))$ with $C_{\text{max}} \in [10, 50]$.
5. **Candidate E (Maximum Floor)**: $S = \max(S_{\text{det}}, \alpha \cdot p_{\text{ML}})$.

---

## 10. Outer-Fold Results & Comparison Table

| Candidate ID | Description | Composite Score | Outer ROC-AUC | Outer AP | Outer Brier | Parameter Mean | Parameter Stability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **BASE_DET** | Deterministic Risk Alone | No | 1.0000 | 1.0000 | 0.0000 | N/A | Deterministic invariant |
| **BASE_LR** | LR Calibrated Alone | No | 0.8510 | 0.9195 | 0.1466 | N/A | Nested CV calibrated |
| **BASE_XGB** | XGB Calibrated Alone | No | 0.8157 | 0.9373 | 0.1600 | N/A | Nested CV calibrated |
| **CANDIDATE_A** | Independent Channels | No | 1.0000 | 1.0000 | N/A | N/A | **Optimal & Safe** |
| **CANDIDATE_B_LR** | Additive (LR) | Yes | 1.0000 | 1.0000 | 0.0000 | $\alpha = 0.0$ | Collapsed to boundary ($0.0$) |
| **CANDIDATE_B_XGB**| Additive (XGB) | Yes | 1.0000 | 1.0000 | 0.0000 | $\alpha = 0.0$ | Collapsed to boundary ($0.0$) |
| **CANDIDATE_C** | Convex Ensemble + Additive | Yes | 1.0000 | 1.0000 | 0.0000 | $\alpha = 0.0, w = 0.0$ | Collapsed to boundary ($0.0$) |
| **CANDIDATE_D** | Bounded ML | Yes | 1.0000 | 1.0000 | 0.0000 | $\alpha = 0.0$ | Collapsed to boundary ($0.0$) |
| **CANDIDATE_E** | Max Floor (Step 5G Shape) | Yes | 1.0000 | 1.0000 | 0.0000 | $\alpha = 0.0$ | Collapsed to boundary ($0.0$) |

---

## 11. Parameter Stability Analysis

* **Fold Consistency**: Across all 4 outer validation folds and all 3 inner folds, the optimal parameter $\alpha$ converged to exactly $0.0$.
* **Mechanism**: Because $S_{\text{det}} = 100$ on all positives, adding $\alpha \cdot p$ adds $0$ points to positives (due to the 100-point ceiling). For negatives, $S_{\text{det}} = 0$, so adding $\alpha \cdot p$ strictly increases the score of negatives, deteriorating Brier score loss. The empirical loss function is strictly minimized at $\alpha = 0.0$.
* **Implication**: Any non-zero $\alpha$ (e.g. $\alpha = 25$ or $\alpha = 50$) cannot be empirically learned from the development data; it would be an arbitrary penalty imposed on verified negative entities.

---

## 12. Chain & Category Confounding

* **Severe Confounding**:
  - **Bitcoin**: 52 records (49 positive OFAC, 3 negative). Positive rate = $94.2\%$.
  - **EVM**: 32 records (17 positive OFAC, 15 negative VASPs/protocols). Positive rate = $53.1\%$.
* **Source Asymmetry**:
  - All 66 positives originate from OFAC SDN listings.
  - All 18 negatives originate from verified exchange proof-of-reserves, official foundation wallets, and protocol deployments.
* This structure makes mathematical fusion between hard sanctions and graph topology inherently collinear.

---

## 13. Bootstrap Uncertainty

* 1,000 entity-level bootstrap resamples were computed.
* **Deterministic AUC 95% CI**: $[1.000, 1.000]$.
* **Collapse to Boundary Frequency**: $100.0\%$ of resamples collapsed $\alpha \to 0.0$.
* The empirical scaling constant is strictly degenerate at zero.

---

## 14. Numerical Mapping Analysis

* **Result**: `NO STABLE NUMERICAL MAPPING IDENTIFIED`.
* The development population does not support a non-zero probability-to-point contribution scale.

---

## 15. Model-Combination Analysis

* In inner fold optimization, the ensemble weight $w$ on LR vs XGB collapsed to boundary values due to the overriding dominance of deterministic labels.
* Neither model demonstrated incremental discriminative superiority over deterministic evidence in a composite formula.
* Preserving LR and XGB as independent analytical reference signals remains the only methodologically defensible stance.

---

## 16. Final Research Outcome

### OUTCOME A:
**`INDEPENDENT CHANNELS EMPIRICALLY PREFERRED FOR POLICY SAFETY`**

The experiments demonstrate that numerical fusion adds zero incremental discriminative information while creating false-positive elevation on negative entities. The independent-channel architecture—already implemented in Step 5I and Step 5J with `fusion_status = NOT_CONFIGURED`—is empirically confirmed as the safest and most robust production policy.

---

## 17. Exact Non-Production Status

* This report is research evidence only.
* No changes have been made to production policies, schemas, or models.
* Downstream alerting and operational decision queues remain isolated from ML probabilities.

---

## 18. Limitations

1. **Sample Support**: The development dataset consists of 84 records across 44 entities, reflecting high class imbalance and historical temporal clustering.
2. **Deterministic Ceiling Effect**: Because all known positive development records are sanctioned, the deterministic engine caps out at 100.0, preventing empirical estimation of ML contributions on non-sanctioned suspicious entities.
3. **Prospective Generalization**: Future prospective investigations featuring non-sanctioned suspicious addresses (e.g. unflagged mule accounts) will require separate empirical calibration before any composite score can be considered.

---

## Explicit Declarations

* `The 9-entity final test set was not accessed or used for policy selection.`
* `No production fusion policy was implemented.`
* `No frozen Step 5H.7 or Step 5I artifact was modified.`
