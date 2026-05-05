

# production-Grade 2-Stage Recommender System

**A Self-Evolving, Distributed ML Platform on Kubernetes**

## Overview

This repository contains a full-stack, production-ready  2-stage recommendation engine. This project implements a **7-service microservice architecture** that handles everything from real-time inference to automated continuous training (CT).

### Key Highlights:

* **Airflow-Orchestrated Data Pipeline:** A robust ETL engine managing ingestion from fragmented APIs, feature engineering, automated feature supervision, and strict data validation to ensure pipeline integrity [Read the deep dive on Medium](https://medium.com/@niranjosh011/the-model-is-just-a-consumer-why-ml-success-starts-at-the-data-layer-8c4b071d06f6).
* **Kubeflow ML Training Pipelines:** End-to-end orchestration of model training for both Retrieval and Ranking layers, facilitating scalable hyperparameter tuning and model evaluation.
* **Two-Stage Retrieval:** Candidate generation via a high-dimensional **Two-Tower model** followed by precise ranking using **XGBoost** [Read the deep dive on Medium](https://medium.com/@niranjosh011/scaling-discovery-building-an-automated-two-stage-recommendation-engine-b172161ae35d).
* **Full CI/CD/CT Lifecycle:** * **CI:** Path-based Docker builds for microservice efficiency.
    * **CD:** Automated deployment and rolling updates on Kubernetes.
    * **CT:** Continuous Training triggered by a custom **Sentinel Service** based on real-time data volume and drift.


* **Infrastructure-as-Code:** Fully containerized environment managed on **Kubernetes** using `Deployments`, `Jobs`, `CronJobs`, and `StatefulSets` [Read the deep dive on Medium](https://medium.com/@niranjosh011/architecting-a-two-stage-recommendation-system-as-kubernetes-microservices-821966791715).
* **Production Observability:** Full-stack visibility into system health and performance using **Prometheus**, **Grafana**, and distributed request tracing with **Jaeger**.



## System Architecture

The system is divided into two logical groups:

### 1. The Always-On Backbone (Inference)

[![](./assets/image_22lm1d22lm1d22lm.png)]

The live request path optimized for low latency and high availability.

* **Gateway Service:** The entry point and orchestrator.
* **User-Embedding Service:** Converts raw user features into high-dimensional vectors.
* **Retrieval Service:** Queries **Pinecone** for Top-K job candidates.
* **Ranking Service:** Re-scores candidates for the final Top-N recommendation.

### 2. The Transient Workers (Pipeline)

[![](./assets/image_v8l1kjv8l1kjv8l1.png)]

Ephemeral services that manage the lifecycle of data and models.

* **Data Service:** Cleans and pushes features to the **Feast Feature Store**.
* **Model-Training Job:** K8s Jobs for distributed training and hyperparameter tuning (Optuna).
* **Job-Embedding CronJob:** Synchronizes the vector database with new job postings.


## Tech Stack

| Category | Tools |
| --- | --- |
| **Orchestration** | Kubernetes (EKS/GKE), Helm |
| **ML Frameworks** | PyTorch (Two-Tower), XGBoost (Ranking), kubeflow |
| **Data & Features** | Airflow, Feast (Feature Store), PostgreSQL, Pinecone (Vector DB) |
| **MLOps** | MLflow (Model Registry), GitHub Actions (CI/CD/CT) |
| **Observability** | Prometheus, Grafana, Jaeger (Tracing) |



## The MLOps Loop (CI/CT/CD)

This project implements a "Closed-Loop" automation strategy:

1. **CI:** Path-based Docker builds via GitHub Actions.
2. **CT:** The **Sentinel Service** monitors Postgres; once a ~1k row threshold is met, it triggers a K8s Training Job.
3. **CD:** Automated "Production" tagging in MLflow and rolling updates in K8s ensure zero-downtime model swaps.



## Getting Started

### Prerequisites

* Kubernetes Cluster (Minikube/Kind for local)
* Python 3.9+

### Installation

1. **Clone the Repo:**
```bash
git clone https://github.com/NiranJosh101/Two-Stage-Recommendation-System.git

```


2. **Deploy Infrastructure:**
```bash
helm install ml-infra ./charts/infrastructure

```


3. **Deploy Services:**
```bash
kubectl apply -f ./k8s/manifests/

```


## Monitoring & Tracing

Access the dashboards to view system health:

* **Grafana:** `http://localhost:3000` (Inference Latency & Error Rates)
* **Jaeger:** `http://localhost:16686` (Distributed Tracing across the 7 services)




