# 🛡️ Fraud Detect Env | Technical Documentation

> This directory contains the implementation of the **Bank Fraud Detection Environment**, following the OpenEnv protocol for evaluating AI agents.

## 🏗️ Architecture

The system is split into an **Environment Server** and an **Agent Script**. This separation ensures the agent has no access to ground truth labels, making evaluation more rigorous and production-ready.

---

## 📂 Core Package Structure

| Path | Description |
| :--- | :--- |
| `server/app.py` | **Environment API.** Uses FastAPI to serve episodes. Handles state transitions (`/reset`, `/step`) and reward scoring. Implements the Web Dashboard at `/`. |
| `inference.py` | **Stateless AI Agent.** Communicates only via HTTP to orchestrate LLM calls and return verdicts to the environment server. |
| `models.py` | Defines the structured data used by both the agent and the server, such as `Transaction`, `FraudType`, and `ActionPlan`. |
| `environment.py` | Centralizes system and task prompts, ensuring consistency across different evaluation runs. |
| `rewards_logger.py` | Utility to capture and serialize agent performance data (JSON/CSV) to the `logs/` directory. |
| `openenv.yaml` | The formal environment specification (inputs, outputs, and condition logic). |

---

## 🛠️ API Surface (Port 5050)

The environment server exposes the following OpenEnv-compliant endpoints (serving on Port **7860**):

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/reset` | Starts a new episode with a fresh transaction. Returns an `observation`. |
| `POST` | `/step` | Submits the agent's action for the current task. Returns a `reward` and a `done` flag. |
| `GET` | `/state` | Returns the detailed current episode state (for dashboard polling). |
| `GET` | `/system` | Diagnostic check for environment stability and API connectivity. |
| `GET` | `/results` | Lists all processed transaction summaries and scores. |
| `GET` | `/` | Serves the **MR.INSPECTOR Surveillance Dashboard**. |

---

## 🧠 Evaluation Logic

The server evaluates the agent based on **ground truth labels** stored in `server/data/transactions.json`.

1.  **Classify Task**: 1.0 reward if correctly identified as `FRAUD` or `LEGIT`.
2.  **Identify Type**: 1.0 reward if the fraud category matches exactly.
3.  **Action Plan**: Multi-point scoring based on the presence of critical remediation keywords (Risk Level, Recommended Action, etc.).

---

## ⚙️ Environment Configuration

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | HuggingFace Inference API URL. |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | The LLM used for fraud detection. |
| `HF_TOKEN` | *Required* | Your HuggingFace API key. |
| `ENV_BASE_URL` | `http://localhost:7860` | The environment server location. |

---

## 🐋 Containerization

A `Dockerfile` is provided for isolated deployment of the environment server and agent. 

```bash
docker build -t fraud-detect-env .
docker run -p 7860:7860 --env-file ../.env fraud-detect-env
```
