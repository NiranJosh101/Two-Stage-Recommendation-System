# Data Service (Service 1)

## Overview

This repository contains **Service 1: the Data Service** for a two-stage recommendation system (retrieval + ranking).

The purpose of this service is **not model training**. Its sole responsibility is to:

> Reliably produce clean entity features and training datasets, and hand them off as durable artifacts for downstream ML services.

This service is orchestrated with **Apache Airflow**, uses **ephemeral staging storage**, and persists final outputs into a **feature store**.



## What This Service Produces

A successful run of the Data Service produces exactly four artifacts:

1. **User features** (offline)
2. **Job features** (offline)
3. **Two-tower training dataset** (retrieval)
4. **Ranking training dataset** (cross-feature based)

Once these artifacts exist, the Data Service is complete.

Downstream services (model training, ANN indexing, ranking) never touch raw data or Airflow internals.



## System Design Principles

This service was intentionally designed to mirror real production ML systems:

* **Clear data ownership** — this service owns feature and dataset creation
* **Strict separation of concerns** — orchestration vs data logic
* **Ephemeral vs durable storage** — raw data is transient; features are permanent
* **Reproducibility** — deterministic pipelines, rerunnable DAGs
* **Simplicity over overengineering** — no unnecessary tooling



## High-Level Data Flow

1. Raw jobs and users are ingested
2. Synthetic interactions are generated after entities exist
3. Raw datasets are validated (fail fast)
4. Clean user and job features are built
5. Interactions are labeled and sampled
6. Retrieval and ranking training datasets are assembled
7. Final artifacts are published to the feature store

Airflow enforces ordering, retries, and failure boundaries.



## Orchestration

### Airflow DAG

The entire service is orchestrated by a **single DAG**:

```
airflow/dags/data_service_dag.py
```

Design choices:

* One DAG = one service
* Flat task structure (no TaskGroups)
* `BashOperator + python -m` for clean execution boundaries
* Manual trigger by default (`schedule_interval=None`)

Airflow is used **only for orchestration** — never for data processing or storage.



## Feature Store

The feature store is the **first durable layer** in the system.

Stored artifacts:

* User features
* Job features
* Two-tower training pairs
* Ranking training dataset

The feature store acts as a **contract** between data and modeling services.



## Why This Design

This project intentionally prioritizes:

* System design clarity over model accuracy
* Production-aligned workflows over experimentation
* Explainability and reasoning over complexity

Synthetic data and proxy features are used deliberately to focus on **architecture, data flow, and ownership**.



## How to Run

### Prerequisites

* Python 3.10+
* Apache Airflow

Ensure `PYTHONPATH` includes the service source directory:

```bash
export PYTHONPATH=$PWD/data-service/src
```

Trigger the DAG manually from the Airflow UI or CLI.



## What Comes Next

This Data Service feeds directly into:

* Retrieval model training
* ANN index construction
* Ranking model training and evaluation

Those concerns intentionally live outside this service.



## Final Note

This service represents how production ML systems are **simplified for clarity without losing correctness**.

It is designed to be readable, defensible, and extensible — not flashy.
