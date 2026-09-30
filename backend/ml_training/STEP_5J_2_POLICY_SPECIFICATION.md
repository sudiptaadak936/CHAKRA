# CHAKRA STEP 5J.2: FORMAL INDEPENDENT-CHANNEL POLICY SPECIFICATION

**Policy Name**: `STEP_5J_2_INDEPENDENT_CHANNEL_POLICY`  
**Contract Version**: `1.0.0`  
**Policy Status**: `FROZEN`  
**Effective Date**: 2026-09-30  
**Machine-Readable Contract**: [`backend/ml_training/policy/step_5J_2_INDEPENDENT_CHANNEL_POLICY.json`](file:///c:/Users/sudip/Documents/antigravity/Chakra/backend/ml_training/policy/step_5J_2_INDEPENDENT_CHANNEL_POLICY.json)  

---

## 1. Executive Policy Verdict

`NUMERICAL ML/RISK FUSION DISABLED — INDEPENDENT CHANNEL ARCHITECTURE FROZEN`

The CHAKRA system formally establishes an **Independent-Channel Architecture** governing on-chain risk evaluation and forensic machine learning. 

* The **Deterministic Risk Engine** (Step 5) remains the sole authoritative source of numerical risk scores ($0.0 \text{ to } 100.0$) and operational risk tiering.
* The **Address ML Inference Service** (Step 5I) remains an independent, non-interfering analytical layer exposing unmerged, calibrated predictive probabilities.
* The **Fusion Layer** (Step 5J) operates in a governed `NOT_CONFIGURED` state. No numerical composite score is computed, authorized, or emitted.

> **Foundational Governance Declarations**:
> * `This policy does not claim that ML is ineffective.`
> * `This policy states that the current evidence does not identify a defensible numerical mapping between deterministic risk and ML signals.`

---

## 2. Why Numerical Fusion is Disabled

Forensic audits in Step 5J.1 and Step 5J.1A established that the authoritative 84-record development population (44 independent entities) possesses complete collinearity between ground-truth labels and deterministic sanctions designations:
* Exactly 66 of 66 positive records are OFAC-sanctioned ($S_{\text{det}} = 100.0$).
* Exactly 18 of 18 negative records have zero sanctions hits and zero typology triggers ($S_{\text{det}} = 0.0$).
* Positive records with $S_{\text{det}} < 100$: **0**.
* Positive records without sanctions: **0**.

Under the evaluated bounded additive fusion formulation and objective, the optimization forces $\alpha^* = 0.0$ because positives cannot rise above 100, while adding risk points to verified negatives increases the evaluated loss. The finding $\alpha = 0$ is a **mathematical artifact of deterministic ceiling saturation ($S_{\text{det}} = 100$) and target-policy coupling**.

Because the incremental value of ML for sanctions-free positive addresses is mathematically **non-identifiable** on the available data, adopting an arbitrary or heuristic composite formula would violate scientific integrity and create regulatory risk. Disabling numerical fusion is a governed safety state resulting from insufficient empirical support, not a model failure.

---

## 3. Deterministic Channel Contract

The existing frozen deterministic Step 5 risk engine remains the authoritative numerical risk index.

### Operational Guarantees:
1. **Output Range**: Discrete point index $S_{\text{det}} \in [0.0, 100.0]$.
2. **Operational Bands**:
   * `CRITICAL`: $\ge 80.0$
   * `HIGH`: $\ge 60.0$
   * `MEDIUM`: $\ge 35.0$
   * `LOW`: $\ge 10.0$
3. **Scoring Semantics**: Immutable. Policy weights defined in `risk_policy.py` are strictly preserved:
   * Sanctions hit: $100.0$ points.
   * `PEEL_CHAIN`: $25.0$ points.
   * `RAPID_HOPPING`: $20.0$ points.
   * `FAN_IN` / `FAN_OUT`: $15.0$ points each.
4. **Immutability Invariants**:
   * `ml_affects_deterministic_score = false`
   * `ml_affects_deterministic_level = false`
   * `ml_attenuates_deterministic_evidence = false`
   * `sanctions_override_by_ml = false`

ML predictions cannot add points, deduct points, reclassify risk bands, or suppress deterministic evidence.

---

## 4. ML Channel Contract

Step 5I remains the independent address-level ML analytical layer.

### Exposed Signals:
* `lr_raw_probability` $\in [0.0, 1.0]$
* `lr_calibrated_probability` $\in [0.0, 1.0]$
* `xgb_raw_probability` $\in [0.0, 1.0]$
* `xgb_calibrated_probability` $\in [0.0, 1.0]$

### Semantic Nature:
These signals are **statistical analytical reference signals only**.
* They are **NOT** probabilities of criminality.
* They are **NOT** probabilities of intent.
* They are **NOT** ownership or attribution probabilities.
* They are **NOT** legal culpability probabilities.
* They are **NOT** deterministic evidence.
* They are **NOT** operational risk levels.

They serve as an independent quantitative intelligence channel for human investigators and downstream compliance systems.

---

## 5. Fusion-Disabled Contract

The fusion contract explicitly codifies the following machine-readable states:

* `numerical_fusion_enabled = false`
* `fusion_status = "NOT_CONFIGURED"`
* `fusion_score = null`
* `fusion_policy_version = null`
* `probability_to_points_mapping = null`

Any request for a unified or fused risk score must return `fusion_status="NOT_CONFIGURED"` alongside the independent deterministic and ML payloads. The system refuses to silently synthesize an ad-hoc or average composite score.

---

## 6. Model-Combination Policy

* `lr_xgb_ensemble_enabled = false`
* `ensemble_weights = null`
* `selection_status = "NO_WINNER_SELECTED"`

Under $\alpha = 0$, the ensemble weight $w$ in $p_{\text{ens}} = w \cdot p_{\text{LR}} + (1 - w) \cdot p_{\text{XGB}}$ is mathematically non-identifiable. The policy formally refuses to select Logistic Regression over XGBoost or vice versa. Both models are presented side-by-side as distinct algorithmic perspectives (linear/calibrated vs. non-linear tree ensemble).

---

## 7. Threshold Policy

* `operational_ml_threshold = null`
* `risk_level_mapping_allowed = false`

No operational decision threshold ($\tau$) is authorized for the ML channel.
* The diagnostic $0.5$ threshold used in Step 5H.6.3 and Step 5H.7 remains strictly **historical evaluation-only**.
* ML probabilities must **NEVER** be mapped to `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW` risk bands.
* Deterministic risk bands remain exclusively the product of the deterministic risk policy.

---

## 8. Sanctions Precedence

* `sanctions_override_by_ml = false`
* `sanctions_precedence = "STRICT_DETERMINISTIC_OVERRIDE"`

Within CHAKRA's deterministic risk policy, a verified sanctions match has strict deterministic precedence and cannot be overridden, attenuated, or dismissed by ML. A sanctions hit permanently assigns $100.0$ deterministic points. Statistical ML inferences, regardless of their magnitude, can never override, attenuate, delay, or dismiss a deterministic sanctions match.

---

## 9. Provenance Policy

The policy binds execution to exact, cryptographically frozen repository assets:

* **Deterministic Policy Version**: `STEP_5_DETERMINISTIC_RISK_POLICY_v1.0.0`
* **ML Freeze Identity**: `STEP_5H_7_FINAL_FREEZE`
* **Feature Schema**: `1.1.0` (13 canonical features)
* **Model Artifacts**:
  * Logistic Regression: `logistic_regression_final.joblib`  
    `SHA-256: 569bbca10c392f06b262a5c0cb7ae5209a34ecc92a852ff93ed2e576b2f86bac`
  * XGBoost: `xgboost_final.json`  
    `SHA-256: e320eb085b2d091bb202d03261cbc15683ede51070c152313cd211ba13575397`
* **Calibrator Artifacts**:
  * LR Calibrator: `lr_calibrator.joblib`  
    `SHA-256: 1e78b493577a03884a4ddcaaf4488cf507f6448aaf16c9eb865f337f513a98df`
  * XGB Calibrator: `xgb_calibrator.joblib`  
    `SHA-256: 90d61a4a7cc91442e58104bb9e7fa9c57c0554df13a55b034fef4fe7ab00aad2`
* **Preprocessor Artifact**:
  * Preprocessor Params: `final_preprocessor_params.json`  
    `SHA-256: 8b6a44e10e0c3b47f7c887f95cc12022aaa206db2c0d6a643cc0f72d5c60e3d5`

---

## 10. Fail-Closed / Degraded State Behavior

1. **Deterministic Risk Unavailable**:
   * When deterministic risk computation is unavailable, the system must not emit a deterministic score or deterministic risk level. It must return an explicit uncalculated/error state and must never synthesize a fallback score.
2. **ML Unavailable / Degraded**:
   * Deterministic channel remains fully operational and authoritative.
   * ML channel explicitly reports `status="UNAVAILABLE"` or degraded.
   * No fallback or estimated ML probabilities may be injected.
3. **Numerical Fusion Requested While Disabled**:
   * The system returns explicit `fusion_status="NOT_CONFIGURED"`.
   * Under no circumstances will a composite score be ad-hoc synthesized.
4. **Unsupported Policy Version**:
   * If a consumer requests an unrecognized or unauthorized policy version, the system fails closed and raises an exception.
5. **Incompatible Provenance**:
   * If any model, calibrator, or preprocessor artifact fails SHA-256 checksum verification, the system raises an `ArtifactIntegrityError`; the affected inference request fails closed and no ML prediction is emitted.

---

## 11. Future Re-Evaluation Requirements

Numerical fusion cannot be authorized or revisited on the existing development cohort. A future fusion validation cycle requires a newly acquired and independently audited dataset satisfying:

1. **Sanctions-Free Positive Entities**: An adequately sized cohort of independent sanctions-free positive entities must be established through a separate prospective statistical support/power analysis before numerical fusion is reconsidered (verified hacks, smart contract exploits, phishing syndicates, ransomware) with no OFAC hit and $S_{\text{det}} < 100$.
2. **Intermediate Deterministic Range**: Entities where observed typologies produce intermediate risk scores ($10 \le S_{\text{det}} \le 75$).
3. **Independent Negative Reference Entities**: Active retail and DeFi traders to empirically measure false-positive elevation under non-zero $\alpha$.
4. **Point-in-Time Behavioral Histories**: Clean multi-transaction histories captured during active malicious operation.
5. **Cross-Chain / Category Strata**: Adequate independent support across relevant chain and source strata must be demonstrated before fusion validation.
6. **Strict Test Isolation**: The frozen 9-entity final test set (`FROZEN_5H_6_FINAL_TEST`) cannot be reused for fusion tuning and remains permanently evaluation-only.

---

## 12. Relationship to Step 5H.7 and Step 5I

* **Step 5H.7 (Artifact Freeze)**: Step 5J.2 depends directly on Step 5H.7. All model binaries, calibrator objects, and feature schemas are consumed strictly as frozen, read-only dependencies.
* **Step 5I (ML Inference Service)**: Step 5J.2 establishes the governance wrapper around Step 5I. Step 5I continues to execute address feature extraction and probability calibration exactly as designed, feeding its results directly into the independent ML channel.

---

## 13. Explicit Limitations

1. **No Universal Inferiority Claim**: This policy does not imply that machine learning cannot generalize or that fusion is intrinsically invalid. It reflects the empirical limits of the current development sample.
2. **Descriptive Nature of Test Set**: The 9-entity held-out test evaluation from Step 5H.6.3 provides separate descriptive evidence and does not establish population-level generalization or justify ad-hoc fusion rules.
3. **No Automated Enforcement Actions**: Neither deterministic risk scores nor ML probabilities constitute autonomous account freezes or asset seizures without mandatory human forensic review.
