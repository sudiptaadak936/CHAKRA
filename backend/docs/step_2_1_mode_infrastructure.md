# CHAKRA — Step 2.1: Live / Demo Mode Infrastructure

## 1. Overview
Step 2.1 implements the centralized operational mode infrastructure for CHAKRA. It establishes a strongly typed, deterministic mechanism for configuring and resolving whether the application runs in `LIVE` or `DEMO` mode.

## 2. Architectural Principle
> **DEMO MODE CHANGES THE DATA SOURCE, NOT THE FORENSIC SEMANTICS.**

CHAKRA enforces a single canonical forensic and ingestion pipeline. Demo mode does NOT duplicate traversal, clustering, scoring, typology detection, graph projection, or persistence logic. Instead, future synthetic scenario sources (Steps 2.2–2.4) feed transactions directly through the canonical pipeline:

```text
Synthetic Scenario Source
    ↓
Canonical Transaction
    ↓
Existing Ingestion/Persistence Pipeline
    ↓
PostgreSQL
    ↓
Neo4j Projection
    ↓
Money-Flow Traversal
    ↓
Clustering
    ↓
Path Scoring
    ↓
Typology Detection
```

## 3. Available Modes
CHAKRA defines two explicit operational modes via the `AppMode` enum (`Mode = AppMode`):

1. **`LIVE` (Default)**
   - Normal production/investigative operation.
   - Consumes live external blockchain data providers (TronGrid, Etherscan, Mempool/Esplora, Solscan/Helius).
   - Fresh installations and unconfigured environments strictly default to `LIVE`.

2. **`DEMO`**
   - Demonstration and evaluation mode.
   - Allows future synthetic scenario generators (Steps 2.2–2.3) to be selected as the transaction input source.
   - **Crucial note**: Step 2.1 establishes ONLY the mode infrastructure. Step 2.1 does NOT generate synthetic scenario data, nor does it alter ingestion pipeline execution.

## 4. Configuration Mechanism
The mode is centralized within the canonical Pydantic settings (`app.core.config.Settings`):

- **Setting Name**: `MODE`
- **Supported Aliases**: `CHAKRA_MODE`, `mode`, `chakra_mode`
- **Default Value**: `AppMode.LIVE`
- **Validation**:
  - Case-insensitive parsing with whitespace stripping (`demo`, `Demo`, `DEMO` → `AppMode.DEMO`; `live`, `Live`, `LIVE` → `AppMode.LIVE`).
  - Strict validation: unknown or invalid tokens (e.g. `banana`, `staging`) raise an explicit `pydantic.ValidationError`.
  - Under no circumstances does an invalid value silently fall back to `DEMO`.

## 5. Central Resolver & Service
Components query the mode via `app.services.mode_service.ModeService`:

```python
from app.services.mode_service import get_mode_service

mode_service = get_mode_service()
current_mode = mode_service.get_mode()  # AppMode.LIVE or AppMode.DEMO
is_live = mode_service.is_live()        # bool
is_demo = mode_service.is_demo()        # bool
```

`ModeService` supports dependency injection and test overrides with zero mutable global state.

## 6. Scope Boundaries
Step 2.1 is strictly internal infrastructure. It does NOT include:
- Scenario definitions or generators (Step 2.2 / 2.3)
- Real pipeline scenario feeding (Step 2.4)
- Chronological replay mode (Step 2.5)
- Public dashboard/frontend UI
