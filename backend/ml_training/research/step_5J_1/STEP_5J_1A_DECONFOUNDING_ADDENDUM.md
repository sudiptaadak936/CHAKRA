# CHAKRA STEP 5J.1A: STATISTICAL DECONFOUNDING & IDENTIFIABILITY AUDIT ADDENDUM

**Date**: 2026-09-30  
**Research Verdict**: `NO DEFENSIBLE NUMERICAL FUSION IDENTIFIED`  
**Execution Context**: Local Python research environment (`backend/ml_training/research/step_5J_1/`)  

---

## 1. Verified Class / Sanctions Contingency Table

Using strictly the 84 development records across 44 independent entities, the exact cross-tabulation between ground-truth label $Y \in \{0, 1\}$ and deterministic sanctions status was computed:

| Category | $Y = 1$ (Positive) | $Y = 0$ (Negative) | Total |
| :--- | :---: | :---: | :---: |
| **Sanctions-Hit Present ($\text{sanctions} = 1$)** | 66 | 0 | 66 |
| **Sanctions-Hit Absent ($\text{sanctions} = 0$)** | 0 | 18 | 18 |
| **Total** | **66** | **18** | **84** |

**Forensic Finding**:
The contingency matrix is strictly diagonal:
* 100% of positive development records ($66/66$) possess an authoritative OFAC SDN sanctions hit.
* 100% of negative development records ($18/18$) have zero sanctions hits.
* There is exact collinearity between target label $Y$ and deterministic sanctions status on this cohort.

---

## 2. Deterministic Score Decomposition

The deterministic risk score $S_{\text{full}} \in [0.0, 100.0]$ was decomposed into its constituent evidentiary contributions:
* Sanctions Component: $100.0$ points if non-synthetic hit present, else $0.0$.
* Typologies Component: Sum of observed typology weights (`PEEL_CHAIN: 25.0`, `RAPID_HOPPING: 20.0`, `FAN_IN: 15.0`, `FAN_OUT: 15.0`).
* Other Components: Attributions, change inferences, cluster contagion.

### Score Distributions by Class:

| Metric | Positive Cohort ($Y = 1$, $N = 66$) | Negative Cohort ($Y = 0$, $N = 18$) |
| :--- | :---: | :---: |
| **$S_{\text{full}}$ (Full Deterministic Score)** | | |
| - Mean $\pm$ Std | $100.0 \pm 0.0$ | $0.0 \pm 0.0$ |
| - Min / Median / Max | $100.0 / 100.0 / 100.0$ | $0.0 / 0.0 / 0.0$ |
| **$S_{\text{without\_sanctions}}$ (Decomposed Score)** | | |
| - Mean $\pm$ Std | $0.0 \pm 0.0$ | $0.0 \pm 0.0$ |
| - Min / Median / Max | $0.0 / 0.0 / 0.0$ | $0.0 / 0.0 / 0.0$ |

**Forensic Finding**:
When the sanctions component is removed counterfactually, $S_{\text{without\_sanctions}}$ collapses to $0.0$ for all 66 positive records and all 18 negative records. The development cohort contains **zero positive records distinguishable by non-sanctions deterministic evidence**.

---

## 3. Identifiability of ML Incremental Value

To determine whether the data can empirically evaluate if ML improves risk assessment on addresses without sanctions:
* Positive records with $S_{\text{full}} < 100$: **0**
* Positive entities with $S_{\text{full}} < 100$: **0**
* Positive records with $\text{sanctions} = 0$: **0**
* Positive entities with $\text{sanctions} = 0$: **0**

### Formal Determination:
**`ML incremental value for the sanctions-free positive population is NOT IDENTIFIABLE from the current development cohort.`**

This is an empirical support limitation of the development dataset. It does **not** establish that ML is ineffective; rather, the required empirical support (suspicious or illicit addresses that lack hard sanctions listings) is entirely absent from the historical development cohort.

---

## 4. Mathematical Reinterpretation of $\alpha = 0$

In the previous research run, all additive and bounded fusion models converged to $\alpha = 0.0$. This result must be understood mathematically rather than as an empirical rejection of ML:

Consider any additive or bounded composite score:
$$S(\alpha) = \min(100, S_{\text{det}} + \alpha \cdot p_{\text{ML}}), \quad \alpha \ge 0, \quad p_{\text{ML}} \in (0, 1)$$

1. **For Positive Records ($Y = 1$)**:
   $$S_{\text{det}} = 100 \implies S(\alpha) = \min(100, 100 + \alpha \cdot p_{\text{ML}}) = 100.0 \quad \forall \alpha \ge 0$$
   Positive scores are permanently saturated at the 100.0 ceiling. The derivative with respect to $\alpha$ is zero: $\frac{\partial S}{\partial \alpha} = 0$.
2. **For Negative Records ($Y = 0$)**:
   $$S_{\text{det}} = 0 \implies S(\alpha) = \min(100, 0 + \alpha \cdot p_{\text{ML}}) = \alpha \cdot p_{\text{ML}}$$
   Negative scores increase strictly monotonically with $\alpha$.
3. **Loss Surface Behavior**:
   Any loss function that penalizes assigning positive risk score to verified negative records while the positive records are already saturated at 100 will be minimized at $\alpha = 0$ on this cohort:
   $$\mathcal{L}(\alpha) = \sum_{i \in \text{pos}} \ell(1, 100) + \sum_{j \in \text{neg}} \ell(0, \alpha \cdot p_{j})$$
   Because positive scores remain at 100 for all $\alpha \ge 0$, positive $\alpha$ cannot improve positive separation while increasing negative scores. The positive term is constant, so $\mathcal{L}(\alpha)$ strictly increases with $\alpha$:
   $$\frac{\partial \mathcal{L}}{\partial \alpha} = \sum_{j \in \text{neg}} \frac{\partial \ell}{\partial S_j} p_j > 0 \quad \forall \alpha > 0$$
   Therefore, $\alpha = 0.0$ is the inevitable mathematical global minimum on this cohort.

**Conclusion**:
$\alpha = 0$ is a mathematical artifact of **deterministic saturation ($S_{\text{det}} = 100$) and target-policy coupling** in this cohort. It must **not** be cited as evidence that ML has no predictive utility.

---

## 5. Probability-Metric Validity Audit

* **Deterministic Score Nature**: $S_{\text{det}} \in [0, 100]$ is a discrete, point-based heuristic risk index resulting from policy weights ($100$ for sanctions, $25$ for peeling chain, etc.). It is **not** a calibrated probability.
* **Methodological Invalidation**: The preliminary conversion $p = S_{\text{det}} / 100$ used in Brier score and log loss calculations in earlier drafts is:
  **`INVALID / NOT PROBABILISTICALLY INTERPRETABLE`** for the deterministic risk index.
* **Appropriate Metrics**: Rank-based metrics (ROC-AUC, Average Precision / PR-AUC) evaluate ordering without assuming probabilistic calibration and remain the only mathematically defensible discriminative measures for composite indices.

---

## 6. Model-Weight Identifiability Audit

In Candidate C (Convex Ensemble):
$$p_{\text{ML}} = w \cdot p_{\text{LR}} + (1 - w) \cdot p_{\text{XGB}}, \quad S = \min(100, S_{\text{det}} + \alpha \cdot p_{\text{ML}})$$
* Under the dominant boundary solution $\alpha = 0$:
  $$S = S_{\text{det}} \quad \forall w \in [0, 1]$$
* The composite score and loss function are completely independent of $w$:
  $$\frac{\partial \mathcal{L}}{\partial w} \equiv 0 \quad \forall w \in [0, 1]$$
* **Formal Finding**: The ensemble weight $w$ is **completely non-identifiable / irrelevant** under the boundary solution.
* Likewise, the bounding cap $C_{\text{max}}$ in Candidate D is non-identifiable under $\alpha = 0$.
* These boundary values ($w = 0.0, C_{\text{max}} = 10.0$) must **never** be cited as evidence for model selection or capacity limits.

---

## 7. Typology Support Interpretation

* All 4 structural typology features (`peel_chain`, `rapid_hop`, `fan_in`, `fan_out`) are identically $0.0$ across the 84 development records.
* The counterfactual masking experiment produced $\Delta = 0.0000$ solely because there were zero typology events to mask.
* **Interpretation**: This finding does **not** prove that typologies are irrelevant prospectively; it merely confirms that the development cohort contains zero empirical typology variance.

---

## 8. Chain & Category Support Asymmetry

The exact distribution of the 84 development records is:
* **Bitcoin**: 52 records (49 positive, 3 negative).
* **EVM**: 32 records (17 positive, 15 negative).
* **Source Asymmetry**:
  - Positives: Exclusively OFAC SDN legal listings.
  - Negatives: Exclusively reference-negative infrastructure (exchange proof-of-reserves, official foundation wallets, protocol pause proxies).
* These proportions reflect the historical sourcing constraints of Steps 5H.1–5H.6.1E and must not be interpreted as representative of general blockchain activity.

---

## 9. Correct Research Conclusion

### **`NO DEFENSIBLE NUMERICAL FUSION IDENTIFIED`**

**Precise Analytical Meaning**:
1. The development cohort does not possess the empirical support required to identify a non-zero probability-to-point contribution scale $\alpha$.
2. The finding $\alpha = 0$ is a consequence of deterministic saturation ($S_{\text{det}} = 100$) and target coupling, not an empirical proof that ML is ineffective.
3. The incremental value of ML for sanctions-free suspicious addresses is non-identifiable on this cohort.
4. No empirical evidence justifies selecting LR over XGB or vice versa.
5. No production fusion policy can be statistically authorized or frozen from this data.
6. The production architecture must continue to expose independent channels with:
   $$\text{fusion\_status} = \text{"NOT\_CONFIGURED"}$$
   This represents an explicit safety and governance state, not an empirical model ranking.

---

## 10. Future Data Requirements to Unblock Numerical Fusion

Before any numerical fusion policy can be empirically calibrated and reviewed for production authorization, the following data requirements must be satisfied:
1. **Sanctions-Free Positive Support**: Authoritative positive addresses (e.g. verified theft, exploit, scam, or ransomware addresses) that do **not** trigger an OFAC sanctions hit and have $S_{\text{det}} < 100$.
2. **Intermediate Deterministic Range**: Examples where deterministic risk produces intermediate scores ($10 \le S_{\text{det}} \le 75$) from observed graph typologies or attribution context.
3. **Independent Negative Stratification**: Non-VASP, ordinary active user addresses to measure false-positive elevation under non-zero $\alpha$.
4. **Balanced Cross-Chain Coverage**: Adequate sample size across both Bitcoin and EVM families for both classes.
5. **Point-in-Time Behavioral Histories**: Unadulterated pre-cutoff observation windows for prospective evaluation.

---

## 11. Historical Metric Reconciliation Note

* In Step 5H.6.3 post-calibration final-test evaluation, the verified discriminative metrics were:
  - **Logistic Regression**: $\text{ROC-AUC} = 0.9500, \quad \text{PR-AUC} = 0.9667$
  - **XGBoost**: $\text{ROC-AUC} = 0.8500, \quad \text{PR-AUC} = 0.8767$
* An informal draft summary previously referenced $0.90 / 0.60$. The official, frozen Step 5H.6.3 report (`STEP_5H_6_3_POST_CALIBRATION_TEST_EVALUATION_REPORT.md` Section 10) and `STEP_5H_6_3_RESULTS.json` record $0.9500$ and $0.8500$ as the authoritative final-test metrics.
* The 9-entity test results are separate descriptive evaluation evidence and do not establish population-level generalization or justify fusion-policy selection.

---

## Explicit Research Declarations

* `The final test set was not accessed.`
* `No production artifact was modified.`
* `No production fusion policy was implemented.`
* `No threshold was selected.`
* `No LR/XGB winner was selected.`
