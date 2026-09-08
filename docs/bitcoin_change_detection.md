# CHAKRA Bitcoin Change-Address Detection (Step 3D)

## 1. What Bitcoin Change-Address Detection Means in CHAKRA
In Bitcoin's UTXO model, spending an unspent transaction output consumes the full amount. When the value of the inputs exceeds the intended transfer amount plus transaction fees, standard wallet software generates a change output to return the difference to the sender. Detecting this change output allows forensic analysts to distinguish between funds transferred to a counterparty and funds retained by the spending entity.

## 2. Deterministic Signals and Audited Rules
CHAKRA implements a calibrated deterministic change detection engine based on on-chain signals:

1. **Self-Change (Direct Cryptographic Evidence)**:
   - *Condition*: Output address matches an input address of the same transaction.
   - *Classification*: `CHANGE_CANDIDATE`, `HIGH` confidence.
   - *Rationale*: Direct on-chain evidence that funds returned to a key that funded the transaction.

2. **Observed Address Reuse (Heuristic Recipient Evidence)**:
   - *Condition*: Output address has been observed receiving funds across multiple distinct transactions (`tx_count > 1`).
   - *Classification*: `EXTERNAL_RECIPIENT`, `LOW` confidence.
   - *Rationale*: Modern HD wallets generate fresh change addresses per transaction (BIP-32/BIP-44). Reused addresses typically function as static deposit addresses, merchant payment points, or donation addresses. However, because users may occasionally reuse addresses or sweep to static cold-storage addresses, address reuse is **evidence rather than proof**. It does not prove distinct ownership or non-change status, so confidence is deliberately calibrated to `LOW`.

3. **One-Time Change via Reuse Elimination (Derived Heuristic Inference)**:
   - *Condition*: In a multi-output transaction where other outputs exhibit observed address reuse, exactly one output is newly observed / single-use.
   - *Classification*: `CHANGE_CANDIDATE`, `LOW` confidence.
   - *Rationale*: The single fresh address is inferred to be change by eliminating the reused counterparty outputs. Because the elimination depends on the heuristic assumption that reused addresses are external recipients, the change inference is also heuristic and is appropriately calibrated to `LOW` confidence.

4. **Single-Output Transactions (Ambiguous Context)**:
   - *Condition*: Transaction has exactly one output.
   - *Classification*: If output matches an input address, `CHANGE_CANDIDATE` (`HIGH` confidence, self-sweep / consolidation). Otherwise, `UNKNOWN` (`UNKNOWN` confidence).
   - *Rationale*: A single output cannot be contrasted against a companion change output. It may represent a complete balance sweep to an external party or an internal wallet consolidation/cold-storage move. Forcing an external recipient classification without contrasting evidence is forensically unsound.

## 3. Evidence vs. Ownership Boundary
CHAKRA maintains a strict boundary between observable blockchain facts, heuristic inferences, and real-world entity claims:

- **Observed Address Reuse**: An observable on-chain pattern where an address receives funds repeatedly.
- **Inferred Change Candidate**: An analytical classification indicating an output likely returned to the sender. It does **not** assert identity or prove common ownership.
- **Inferred External-Recipient Behavior**: An analytical classification indicating an output behaves like an external destination based on reuse.
- **Unknown / Ambiguous**: When available evidence is insufficient to justify an inference, the detector safely returns `UNKNOWN` with an explicit reason rather than guessing.

Under no circumstances does Step 3D assert that two addresses belong to the same legal person or real-world entity.

## 4. Graph & Persistence Representation
- **PostgreSQL Derived Table**: Inferences are persisted in `bitcoin_change_inferences` (`txid`, `output_index`, `address`, `classification`, `confidence`, `evidence_reason`). Canonical transaction and UTXO tables are never mutated.
- **Neo4j Graph Model**: Step 3A/3B graph topology remains strictly UTXO-based (`Address → SPENT_INPUT → Transaction → CREATED_OUTPUT → Address`). Inferred change candidate status attaches as metadata; no synthetic `Address → TRANSFERRED → Address` edges are fabricated.

## 5. Deliberately Ambiguous / Inconclusive Cases
The detector strictly outputs `UNKNOWN` without forcing a guess in the following situations:
- Multiple fresh outputs without prior transaction history.
- Equal-value outputs where neither matches an input and neither has prior history.
- Isolated transactions with zero historical context across the database.
- A fresh output in a transaction that already has an explicit self-change output.
- Single-output transactions to addresses not appearing in the spending inputs.
