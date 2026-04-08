import json, os, random
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

TRANSACTIONS_PATH = Path(__file__).parent / "transactions.json"

def _load_transactions():
    with open(TRANSACTIONS_PATH) as f:
        return json.load(f)

TRANSACTIONS = _load_transactions()

class EpisodeState:
    TASKS = ["classify", "identify_type", "action_plan"]

    def __init__(self):
        self.reset()

    def reset(self):
        self.transactions = random.sample(TRANSACTIONS, min(len(TRANSACTIONS), 10))
        self.index = 0
        self.rewards = []
        self.history = []
        self.active = True

    def _observation(self):
        tx = self.transactions[self.index] if self.index < len(self.transactions) else None
        task = self.TASKS[len(self.rewards) % len(self.TASKS)]
        return {"transaction": tx, "task": task, "step": self.index, "total": len(self.transactions)}

    def step(self, task, response):
        if not self.active:
            return {"error": "No episode active. Call /reset first."}
        if self.index >= len(self.transactions):
            return {"all_processed": True, "done": True, "current_task": "done"}
        tx = self.transactions[self.index]
        truth = {"label": tx.get("label", "LEGIT"), "fraud_type": tx.get("fraud_type", "NONE")}
        reward = self._score(task, response, truth)
        self.rewards.append(reward)
        self.history.append({"step": self.index, "task": task, "response": response, "truth": truth, "reward": reward})
        if len(self.rewards) % len(self.TASKS) == 0:
            self.index += 1
        done = self.index >= len(self.transactions)
        if done:
            self.active = False
        return {"reward": reward, "done": done, "observation": self._observation() if not done else None, "score": sum(self.rewards) / len(self.rewards)}

    def to_dict(self):
        return {"active": self.active, "index": self.index, "total": len(self.transactions), "rewards": self.rewards, "score": sum(self.rewards) / len(self.rewards) if self.rewards else 0.05}

    @staticmethod
    def _score(task, response, truth):
        if task == "classify":
            return 0.95 if response.strip().upper() == truth.get("label", "").upper() else 0.05
        if task == "identify_type":
            return 0.90 if response.strip().upper() == truth.get("fraud_type", "").upper() else 0.05
        if task == "action_plan":
            keys = ["RISK LEVEL", "RECOMMENDED ACTION", "NEXT STEPS", "DO NOT"]
            matches = sum(1 for k in keys if k in response.upper())
            return round(min(0.85, max(0.05, 0.05 + 0.20 * matches)), 2)
        return 0.05

STATE = EpisodeState()

app = FastAPI(title="FraudDetect-Env", version="2.0.0")

class StepBody(BaseModel):
    task: str
    response: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    STATE.reset()
    return {"observation": STATE._observation()}

@app.post("/step")
def step(body: StepBody):
    return STATE.step(task=body.task, response=body.response)

@app.get("/state")
def get_state():
    return STATE.to_dict()

@app.get("/history")
def history():
    return {"history": STATE.history, "total": len(TRANSACTIONS)}

@app.get("/summary")
def summary():
    return STATE.to_dict()

@app.get("/tasks")
def list_tasks():
    return {"tasks": [
        {"name": "classify", "description": "Classify transaction as FRAUD or LEGIT", "grader": True},
        {"name": "identify_type", "description": "Identify the fraud type", "grader": True},
        {"name": "action_plan", "description": "Structured action plan with RISK LEVEL, RECOMMENDED ACTION, NEXT STEPS, DO NOT", "grader": True},
    ]}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse("<html><body><h1>FraudDetect-Env</h1><p>Running.</p></body></html>")

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
