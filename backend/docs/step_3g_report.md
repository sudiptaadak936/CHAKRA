# CHAKRA Step 3G — Implementation & Semantic Correction Report

## Status: PASS

| Field | Value |
|---|---|
| Step | 3G — Typology Detection |
| Implementation file | `backend/app/forensics/typology_detector.py` |
| Test file | `backend/tests/test_typology_detector.py` |
| Step 3G tests | **26 / 26 passed** |
| Full regression | **304 / 304 passed** |
| Step 3F baseline | 278 tests |
| Step 3G tests added | 26 |
| Total reconciled count | **304** (278 + 26) ✓ |
| Steps 3A–3F modified | **NO** (0 earlier files modified) |
| DB writes issued | **NONE** (strictly read-only) |
| Frozen steps broken | **NONE** |

---

## 1. Architecture & Semantic Principles

`TypologyDetector` is an analytical, read-only classification layer built directly on top of CHAKRA's existing traversal, clustering, and path-scoring infrastructure. It consumes a `TraversalPath` and queries historical transaction graphs without altering any database tables or state.

> **CRITICAL FORENSIC BOUNDARY**:
> Typologies identify transaction-flow structural patterns. They **do NOT establish criminality, illicit intent, common ownership, attribution, or guilt**. Thresholds are engineering calibration defaults, not legal or probabilistic certainty metrics.

---

## 2. Configurable Engineering Defaults

| Parameter | Default Value | Description |
|---|---|---|
| `MIN_PEEL_CHAIN_HOPS` | `3` | Minimum consecutive hops satisfying the peel/remainder rule |
| `MIN_FAN_IN_SOURCES` | `3` | Minimum distinct upstream senders within the temporal window |
| `MIN_FAN_OUT_DESTINATIONS` | `3` | Minimum distinct downstream recipients within the temporal window |
| `MAX_RAPID_HOP_INTERVAL_MINUTES` | `60.0` | Maximum temporal duration across consecutive rapid hops |
| `MIN_RAPID_HOPS` | `3` | Minimum consecutive hops within the rapid interval |
| `DEFAULT_FAN_WINDOW_HOURS` | `24.0` | Configurable temporal aggregation window for Fan-In and Fan-Out |

---

## 3. Implemented Typology Rules

### A. PEEL_CHAIN (Strengthened Rule)
- **Applicability**: Bitcoin UTXO paths (`path.hops >= MIN_PEEL_CHAIN_HOPS`).
- **Rule**:
  1. Each hop in the sequence must have its continuing output classified as `CHANGE_CANDIDATE` by Step 3D.
  2. The transaction must contain at least 2 outputs (single-output transactions, self-sweeps, and simple consolidations are strictly rejected).
  3. A separate non-change (peeled) output must exist with `value_sat > 0`.
  4. The continuing output must represent a meaningful remainder (`retention_ratio >= 0.50` and `continuing_sat >= peeled_sat`).
  5. Amounts must be present; missing amounts or incompatible data result in conservative non-detection rather than fabricated values.
- **Recorded Evidence**:
  - `txid`, `continuing_address`, `continuing_amount_sat`, `peeled_addresses`, `peeled_amount_sat`, `upstream_amount_sat`, `retention_ratio`, `peel_ratio`, timestamps, Step 3D classification and reason.

### B. FAN_IN (Aggregation)
- **Applicability**: All nodes traversed by the path.
- **Rule**:
  1. Identifies distinct upstream senders to the aggregation node.
  2. Enforces the configurable temporal window (`max_fan_window_hours = 24.0h`).
  3. At least `MIN_FAN_IN_SOURCES = 3` distinct sources must transfer to the node within a single window of duration `<= max_fan_window_hours`. Historical sources dispersed across days/months outside this window are rejected.
  4. Missing timestamps prevent detection rather than using fabricated times.
- **Recorded Evidence**:
  - `addresses` (aggregation address), `counterparty_addresses` (distinct upstream sources, sorted), `transaction_ids` (sorted), `amounts` (with unit), `timestamps`, `temporal_window` (`window_start`, `window_end`, `duration_hours`, `max_window_hours`).

### C. FAN_OUT (Distribution)
- **Applicability**: All nodes traversed by the path.
- **Rule**:
  1. Identifies distinct downstream destinations from the source node.
  2. Enforces the configurable temporal window (`max_fan_window_hours = 24.0h`).
  3. At least `MIN_FAN_OUT_DESTINATIONS = 3` distinct destinations must receive transfers from the node within a single window of duration `<= max_fan_window_hours`.
  4. Ordinary service/exchange distribution is noted as structural topology only, never classified as suspicious.
- **Recorded Evidence**:
  - `addresses` (source address), `counterparty_addresses` (distinct downstream destinations, sorted), `transaction_ids` (sorted), `amounts` (with unit), `timestamps`, `temporal_window`.

### D. RAPID_HOPPING
- **Rule**: Evaluates consecutive path hop timestamps. If `MIN_RAPID_HOPS = 3` hops occur within `<= MAX_RAPID_HOP_INTERVAL_MINUTES = 60.0` minutes, triggers `observed` rapid hopping. Missing timestamps prevent classification.

### E. MIXER_INTERACTION (Safety Stub)
- **Rule**: Strictly returns `confidence_level = "insufficient_evidence"` with explanation: `"Mixer detection unavailable: no authoritative mixer identification source currently configured."` Never equates high degree or multi-party transactions with a mixer.

### F. CROSS_CHAIN / BRIDGE (Safety Stub)
- **Rule**: Strictly returns `confidence_level = "insufficient_evidence"` with explanation: `"No bridge detection. Explicit cross-chain bridge data is not currently configured."` Never infers cross-chain bridges from timestamps or multi-chain addresses alone.

---

## 4. Determinism & Provenance

1. **Deterministic SHA-256 ID Generation**:
   `_make_detection_id` canonicalizes inputs as:
   `f"{typ}:{chain}:" + "|".join(sorted(set(elements)))`
   ensuring that identical evidence always yields identical IDs regardless of database row order or retrieval timing.
2. **Deterministic Output Ordering**:
   Result detections are ordered by `(d.typology_type.value, d.detection_id)`.
3. **Lexicographical Element Sorting**:
   All `addresses`, `counterparty_addresses`, and `transaction_ids` lists are deduplicated and sorted lexicographically.
4. **Zero State Mutation / Read-Only**:
   Only `SELECT` queries are executed; no DML/DDL is emitted.

---

## 5. Test Suite Verification

### Targeted Test Suite (`backend/tests/test_typology_detector.py`)
**26 / 26 PASSED** (0.43s)

1. `test_01_peel_chain_genuine_3_hops_positive`: PASSED
2. `test_02_peel_chain_change_without_peel_remainder_negative`: PASSED
3. `test_03_peel_chain_insufficient_hops_negative`: PASSED
4. `test_04_peel_chain_missing_incompatible_amounts_conservative`: PASSED
5. `test_05_peel_chain_evidence_provenance`: PASSED
6. `test_06_peel_chain_broken_by_non_change_hop`: PASSED
7. `test_07_fan_in_exact_3_sources_within_window_positive`: PASSED
8. `test_08_fan_in_3_historical_sources_outside_window_negative`: PASSED
9. `test_09_fan_in_source_addresses_and_txids_preserved`: PASSED
10. `test_10_fan_in_below_source_threshold_negative`: PASSED
11. `test_11_fan_in_missing_timestamps_negative`: PASSED
12. `test_12_fan_out_exact_3_destinations_within_window_positive`: PASSED
13. `test_13_fan_out_3_historical_destinations_outside_window_negative`: PASSED
14. `test_14_fan_out_destination_addresses_and_txids_preserved`: PASSED
15. `test_15_fan_out_below_destination_threshold_negative`: PASSED
16. `test_16_rapid_hopping_exact_boundary`: PASSED
17. `test_17_rapid_hopping_exceeds_threshold`: PASSED
18. `test_18_rapid_hopping_missing_timestamps`: PASSED
19. `test_19_mixer_always_insufficient_evidence`: PASSED
20. `test_20_cross_chain_always_insufficient_evidence`: PASSED
21. `test_21_overlapping_typologies_all_returned`: PASSED
22. `test_22_deterministic_sorting_by_type`: PASSED
23. `test_23_read_only_no_db_writes`: PASSED
24. `test_24_deterministic_id_generation`: PASSED
25. `test_25_repeatability`: PASSED
26. `test_26_empty_path_returns_empty`: PASSED

### Full Regression Suite (`pytest backend/tests -v`)
**304 / 304 PASSED** (0 failures, 0 skips, 1m 26s)
- Step 3F baseline: 278 passed
- Step 3G suite: 26 passed
- Total: 304 passed ✓

---

## 6. Gate Evaluation

| Gate Requirement | Status |
|---|---|
| Peel chain strengthened (requires peel output + remainder value) | PASS |
| Single-output sweeps excluded from peel chains | PASS |
| Fan-in enriched with upstream sources and temporal window | PASS |
| Fan-out enriched with downstream destinations and temporal window | PASS |
| Temporal aggregation window parameter (`max_fan_window_hours = 24.0`) | PASS |
| Mixer & Cross-Chain safety stubs (`insufficient_evidence`) | PASS |
| Deterministic IDs and lexicographical list sorting | PASS |
| Read-only behavior enforced | PASS |
| Targeted tests: 26/26 passed | PASS |
| Full regression: 304/304 passed | PASS |
| Steps 3A–3F integrity preserved | PASS |

**STEP 3G GATE: PASS**
