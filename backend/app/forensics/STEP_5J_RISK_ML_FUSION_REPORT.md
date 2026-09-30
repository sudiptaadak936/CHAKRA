# CHAKRA STEP 5J: RISK & ML FUSION AUDIT REPORT

**Date**: 2026-09-30  
**Status**: `BLOCKED` (Mathematical fusion policy not specified in repository contracts)  
**Execution Milestone**: `STEP_5J_BLOCKED_FUSION_POLICY_NOT_SPECIFIED`  

---

## 1. Architecture Audit

An exhaustive forensic architecture audit of the CHAKRA repository was conducted to inspect the deterministic risk engine, the newly implemented Step 5I address-level ML inference layer, and all potential integration boundaries:

1. **Deterministic Risk Boundary**:
   - Location: [`backend/app/forensics/risk_engine.py`](file:///c:/Users/sudip/Documents/antigravity/Chakra/backend/app/forensics/risk_engine.py), [`backend/app/forensics/risk_policy.py`](file:///c:/Users/sudip/Documents/antigravity/Chakra/backend/app/forensics/risk_policy.py).
   - Domain Level: `ADDRESS`.
   - Inputs: `target_address`, `chain`, `network`, `sanctions_hits`, `typologies`, `attributions`, `change_inferences`, `cluster_contagions`.
   - Scoring Logic: Point-based additive policy (`SANCTIONS_HIT = 100.0`, typologies like `PEEL_CHAIN = 25.0`, `RAPID_HOPPING = 20.0`, `FAN_IN = 15.0`, `FAN_OUT = 15.0`), clamped to `[0.0, 100.0]`.
   - Threshold Bands: `CRITICAL >= 80.0`, `HIGH >= 60.0`, `MEDIUM >= 35.0`, `LOW >= 10.0`, `NEUTRAL < 10.0`.
   - Extension Points: The deterministic engine produces a frozen `RiskScoreRecord` with a deterministic hash. It currently does not contain an internal extension point for ML probability injection.

2. **Address-Level ML Inference Boundary**:
   - Location: [`backend/app/forensics/address_ml_service.py`](file:///c:/Users/sudip/Documents/antigravity/Chakra/backend/app/forensics/address_ml_service.py), [`backend/app/forensics/address_ml_loader.py`](file:///c:/Users/sudip/Documents/antigravity/Chakra/backend/app/forensics/address_ml_loader.py), [`backend/app/forensics/address_ml_preprocessor.py`](file:///c:/Users/sudip/Documents/antigravity/Chakra/backend/app/forensics/address_ml_preprocessor.py).
   - Domain Level: `ADDRESS`.
   - Inputs: 13 canonical features (Schema `1.1.0`).
   - Models: Final Logistic Regression and Final XGBoost models trained on 84 development records.
   - Calibrators: Frozen Platt sigmoid calibrators from Step 5H.6.2.2.
   - Outputs: `lr_raw_probability`, `lr_calibrated_probability`, `xgb_raw_probability`, `xgb_calibrated_probability`, artifact provenance.

3. **Existing Fusion Abstractions**:
   - `backend/ml_training/fusion.py` is explicitly restricted to transaction-level Elliptic research ML (Step 5F). Its docstring specifies: *"This class is restricted to the Elliptic transaction-level research domain. It MUST NOT be integrated directly into CHAKRA production address-level risk scoring."*
   - In `backend/ml_training/STEP_5G_REPORT.md` Section 15, a provisional proposal was noted: `final_score = max(deterministic_base_score, ml_risk_probability * ML_POLICY_WEIGHT)`. However, no `ML_POLICY_WEIGHT` constant was ever specified, no model combination rule between LR and XGB was authorized, and no threshold interaction was codified into frozen policy contracts.

---

## 2. Deterministic Risk Contract

The Step 5 deterministic risk engine remains 100% frozen and byte-for-byte immutable:
- Operates on verified blockchain evidence and sanctions lists.
- Completely reproducible with input-canonicalized SHA-256 hashes (`deterministic_hash`).
- Strictly non-probabilistic: points are assigned based on evidentiary categories, not learned parameters.
- No modifications have been made to scoring weights, typology points, threshold bands, or sanctions rules.

---

## 3. ML Inference Contract

The Step 5I address-level ML inference layer remains 100% frozen and byte-for-byte immutable:
- Accepts exact 13-feature vectors under Schema 1.1.0.
- Imputes missing velocity with frozen development median $24253.384615$ (LR) and maps sentinel to `NaN` (XGB).
- Generates unadulterated raw probabilities and applies frozen sigmoid recalibration.
- Retains cryptographic provenance hashes of all 5 frozen artifacts.
- Outputs are strictly analytical reference probabilities, never factual proof of wrongdoing.

---

## 4. Exact Fusion Policy Source

**Findings**:
- **No Authorized Frozen Fusion Policy Exists**:
  - The repository does not contain a frozen specification defining how to mathematically fuse deterministic risk scores with address-level ML probabilities.
  - While Step 5G proposed a template `max(base, p * weight)`, the exact value of `ML_POLICY_WEIGHT` was intentionally omitted, as Step 5G predated address dataset sourcing, baseline training, OOF calibration, and final model evaluation.
  - In Step 5H.6.3, the report explicitly deferred ensemble winner selection and thresholding: *"No production threshold has been selected... LR and XGBoost are preserved as independent models."*
- **Policy Injunction**:
  - Under Step 5J instructions: *"A policy decision must not be fabricated by the implementation agent. If no explicit fusion specification exists: STOP before inventing a production fusion formula... report STEP 5J BLOCKED — FUSION POLICY NOT YET SPECIFIED."*

---

## 5. Mathematical Formula

* **Configured Formula**: **NONE** (`None`).
* **Ad-Hoc Formulas Rejected**:
  - `0.5 * LR + 0.5 * XGB` (Rejected: Unprincipled model combination).
  - `0.7 * deterministic + 0.3 * ML` (Rejected: Incompatible dimensionalities and fabricated weights).
  - `max(deterministic, 100 * p)` (Rejected: Unspecified policy scaling).
* **Execution State**:
  - In accordance with Section 5J.7, when both deterministic and ML outputs are independently available but no fusion policy is defined:
    $$\text{fusion\_status} = \text{"NOT\_CONFIGURED"}$$
    $$\text{fusion\_score} = \text{None}$$
    $$\text{fusion\_policy\_version} = \text{None}$$

---

## 6. Model-Component Handling

Because Logistic Regression and XGBoost exhibit different calibration bounds, extrapolation characteristics, and calibration slope sensitivities:
- The two models are **NOT averaged** or ensembled.
- Both model outputs (`lr_raw_probability`, `lr_calibrated_probability`, `xgb_raw_probability`, `xgb_calibrated_probability`) are preserved independently within `ml_layer`.
- Model provenance hashes are carried alongside the outputs.

---

## 7. Threshold Handling

* **Zero Operational Thresholds**: No operational decision threshold exists.
* The diagnostic $0.5$ classification threshold utilized during offline Step 5H.6.3 testing is **NOT** a production threshold and is strictly forbidden from operational alerting.
* The system does **NOT** convert ML probabilities into `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW` bands.
* The deterministic risk level (`risk_level`) is derived strictly from deterministic evidence points (e.g. sanctions, typologies).

---

## 8. Provenance Handling

Every composite result generated by `RiskMLFusionService` carries:
1. `deterministic_layer.deterministic_hash`: Cryptographic digest of deterministic inputs.
2. `ml_layer.provenance`: Cryptographic digests of all 5 frozen Step 5H.7 ML artifacts:
   - LR Model: `569bbca10c392f06b262a5c0cb7ae5209a34ecc92a852ff93ed2e576b2f86bac`
   - XGB Model: `e320eb085b2d091bb202d03261cbc15683ede51070c152313cd211ba13575397`
   - Preprocessor: `8b6a44e10e0c3b47f7c887f95cc12022aaa206db2c0d6a643cc0f72d5c60e3d5`
   - LR Calibrator: `1e78b493577a03884a4ddcaaf4488cf507f6448aaf16c9eb865f337f513a98df`
   - XGB Calibrator: `90d61a4a7cc91442e58104bb9e7fa9c57c0554df13a55b034fef4fe7ab00aad2`
3. `fusion_layer.fusion_status`: Explicitly indicates `NOT_CONFIGURED`.

---

## 9. Fail-Closed Behavior

The composite fusion service enforces fail-closed operations across all boundaries:
- If target address, chain, or network are missing/empty: fails closed (`ValueError`).
- If deterministic risk evaluation fails: fails closed (raises exception).
- If ML feature schema $\ne \text{"1.1.0"}$, feature count $\ne 13$, or non-finite features: fails closed (`FeatureContractError`).
- If ML artifact integrity is breached: fails closed (`ArtifactIntegrityError`).
- If an unauthorized fusion policy version is supplied: returns `fusion_status=FAILED` with explicit error reason.
- If combining pre-computed results where target address, chain, or network do not match: fails closed (`ValueError`).

---

## 10. Side-Effect Audit

The integration service is strictly read-only and purely analytical:
- **PostgreSQL**: Zero insert, update, or delete operations.
- **Neo4j**: Zero write transactions.
- **Redis / Alerts**: Zero alert records constructed, zero stream messages published.
- **ML Artifacts**: Zero files modified or retrained.
- **External Calls**: Zero HTTP or network calls dispatched.

---

## 11. Test Coverage

Comprehensive test suite [`backend/tests/test_risk_ml_fusion.py`](file:///c:/Users/sudip/Documents/antigravity/Chakra/backend/tests/test_risk_ml_fusion.py) tests:
1. Deterministic risk preservation (exact output, evidence, and deterministic hash identical to standalone Step 5).
2. ML output preservation (exact raw/calibrated probabilities and artifact provenance identical to standalone Step 5I).
3. Explicit `NOT_CONFIGURED` fusion status when mathematical policy is unspecified.
4. Refusal to fabricate weights or implicitly average LR and XGB.
5. Strict address/chain/network consistency checking in composite combinations.
6. Fail-closed handling for bad schema, bad dimensions, and non-finite features.
7. Read-only audit confirming zero DB, alert, or disk mutations.
8. Threshold protection verifying no operational risk bands are applied to ML probabilities.

---

## 12. Known Limitations & Required Specifications to Unblock Step 5J

To unblock formal mathematical fusion in future milestones, the governance / ML risk committee must formally authorize and codify:
1. **Model Selection / Weighting**: A contract specifying whether LR, XGB, or a formal ensemble represents the authoritative address ML signal.
2. **Mathematical Fusion Operator**: A formally approved fusion formula (e.g. bounded Bayesian update, weighted maximum, or tiered override).
3. **Dimensional Scaling**: Explicit scaling factors converting calibrated probabilities `[0.0, 1.0]` into risk engine points `[0.0, 100.0]`.
4. **Precedence Contract**: Precise precedence rules between deterministic sanctions, typology detections, and ML probabilities.
5. **Operational Threshold Policy**: Formally validated operational decision thresholds with approved false-positive / false-negative trade-offs.

---

## 13. Final Gate Conclusion

Because no explicit mathematical fusion policy has been specified in frozen contracts, the integration layer preserves independent presentation of deterministic evidence and ML signals with `fusion_status = NOT_CONFIGURED`, and halts:

`STEP 5J BLOCKED — FUSION POLICY NOT SPECIFIED`
