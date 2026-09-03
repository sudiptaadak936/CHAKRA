# CHAKRA — Cryptocurrency Fraud Investigation Platform

CHAKRA is an autonomous blockchain investigation platform designed to trace illicit funds, cluster multi-chain wallet infrastructure, identify exchange deposit endpoints (VASPs), and generate court-admissible forensic audit trails.

---

## Current Status: Step 0 Foundation Scaffold

Step 0 establishes the core containerized multi-database infrastructure, FastAPI backend, React + TypeScript frontend, unified asynchronous database lifecycle manager, dependency health probes, and live status dashboard.

> [!NOTE]
> Blockchain ingestion, RPC providers, graph algorithms, entity clustering, and machine learning components are scheduled for subsequent roadmap milestones and are not part of Step 0.

---

## Prerequisites

- **[Docker Engine](https://docs.docker.com/engine/install/)** (v24.0+) & **Docker Compose** (v2.20+)
- *(Optional for direct local development)*:
  - **Python 3.11+**
  - **Node.js 20+** & **npm 10+**

---

## Quickstart (Docker Compose)

### 1. Environment Setup (Optional)
Copy the example environment configuration to customize ports or credentials:
```bash
cp .env.example .env
```
*(Docker Compose provides safe local development defaults even if `.env` is omitted.)*

### 2. Build and Start the Stack
Start all services (PostgreSQL, Neo4j Community, Redis, FastAPI Backend, React Frontend) in detached mode:
```bash
docker compose up --build -d
```

### 3. Verify Container Status
Check that all containers are healthy and running:
```bash
docker compose ps
```

---

## Service Endpoints

| Service | Endpoint / URL | Description |
|---|---|---|
| **Frontend UI** | [http://localhost:3000](http://localhost:3000) | Live Infrastructure Status & Health Dashboard |
| **Backend Health** | [http://localhost:8000/health](http://localhost:8000/health) | Core API liveness probe (`{"status":"ok"}`) |
| **Dependency Health** | [http://localhost:8000/health/dependencies](http://localhost:8000/health/dependencies) | PostgreSQL, Neo4j, and Redis connectivity probes |
| **API Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive OpenAPI / Swagger UI |
| **Neo4j Browser** | [http://localhost:7474](http://localhost:7474) | Neo4j Community Graph Database Web UI |

---

## Health Check CLI

To run the automated health check script against the running backend:
```bash
python scripts/provider_health_check.py
```

---

## Running Backend Tests

### Unit Tests
Run backend unit tests (which do not require external databases running):
```bash
cd backend
python -m pytest tests/test_config.py tests/test_health.py
```

### Integration Tests
Run integration tests against live running containers:
```bash
cd backend
python -m pytest tests/integration/test_dependencies_integration.py
```

---

## Logs & Lifecycle Management

### View Container Logs
```bash
# View logs from all services
docker compose logs -f

# View logs for a specific service
docker compose logs -f backend
docker compose logs -f postgres
docker compose logs -f neo4j
docker compose logs -f redis
```

### Stop the Stack
```bash
docker compose down
```

### Reset Databases (Volume Teardown)
To completely reset PostgreSQL, Neo4j, and Redis persistent volumes:
```bash
docker compose down -v
```
