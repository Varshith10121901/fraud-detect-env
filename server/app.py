import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

TRANSACTIONS_PATH = Path(__file__).parent.parent / "transactions.json"


def _load_transactions():
    """Load transaction data from JSON file"""
    with open(TRANSACTIONS_PATH) as f:
        return json.load(f)


TRANSACTIONS = _load_transactions()


class EpisodeState:
    """Manages the state of a fraud detection episode"""
    
    TASKS = ["classify", "identify_type", "action_plan"]

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset episode state with new random sample of transactions"""
        self.transactions = random.sample(TRANSACTIONS, min(len(TRANSACTIONS), 10))
        self.index = 0
        self.rewards = []
        self.history = []
        self.active = True

    def _observation(self):
        """Get current observation for the agent"""
        tx = self.transactions[self.index] if self.index < len(self.transactions) else None
        task = self.TASKS[len(self.rewards) % len(self.TASKS)]
        return {
            "transaction": tx,
            "task": task,
            "step": self.index,
            "total": len(self.transactions)
        }

    def step(self, task: str, response: str):
        """Process agent action and return reward"""
        if not self.active:
            return {"error": "No episode active. Call /reset first."}
        
        if self.index >= len(self.transactions):
            return {
                "all_processed": True,
                "done": True,
                "current_task": "done"
            }
        
        tx = self.transactions[self.index]
        truth = {
            "label": tx.get("label", "LEGIT"),
            "fraud_type": tx.get("fraud_type", "NONE")
        }
        
        reward = self._score(task, response, truth)
        self.rewards.append(reward)
        self.history.append({
            "step": self.index,
            "task": task,
            "response": response,
            "truth": truth,
            "reward": reward
        })
        
        # Move to next transaction after completing all 3 tasks
        if len(self.rewards) % len(self.TASKS) == 0:
            self.index += 1
        
        done = self.index >= len(self.transactions)
        if done:
            self.active = False
        
        return {
            "reward": reward,
            "done": done,
            "observation": self._observation() if not done else None,
            "score": sum(self.rewards) / len(self.rewards) if self.rewards else 0.05
        }

    def to_dict(self):
        """Return state as dictionary"""
        return {
            "active": self.active,
            "index": self.index,
            "total": len(self.transactions),
            "rewards": self.rewards,
            "score": sum(self.rewards) / len(self.rewards) if self.rewards else 0.05
        }

    @staticmethod
    def _score(task: str, response: str, truth: Dict[str, str]) -> float:
        """Score agent response based on task type"""
        if task == "classify":
            return 0.95 if response.strip().upper() == truth.get("label", "").upper() else 0.05
        
        if task == "identify_type":
            return 0.90 if response.strip().upper() == truth.get("fraud_type", "").upper() else 0.05
        
        if task == "action_plan":
            keys = ["RISK LEVEL", "RECOMMENDED ACTION", "NEXT STEPS", "DO NOT"]
            matches = sum(1 for k in keys if k in response.upper())
            return round(min(0.85, max(0.05, 0.05 + 0.20 * matches)), 2)
        
        return 0.05


# Global state instance
STATE = EpisodeState()

# FastAPI app
app = FastAPI(
    title="FraudDetect-Env",
    version="2.0.0",
    description="Bank Fraud Detection RL Environment - OpenEnv Hackathon"
)


class StepBody(BaseModel):
    """Request body for /step endpoint"""
    task: str
    response: str


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/reset")
def reset():
    """Reset the environment to start a new episode"""
    STATE.reset()
    return {"observation": STATE._observation()}


@app.post("/step")
def step(body: StepBody):
    """Take a step in the environment with agent's response"""
    return STATE.step(task=body.task, response=body.response)


@app.get("/state")
def get_state():
    """Get current state of the episode"""
    return STATE.to_dict()


@app.get("/history")
def history():
    """Get full episode history"""
    return {
        "history": STATE.history,
        "total": len(TRANSACTIONS)
    }


@app.get("/summary")
def summary():
    """Get episode summary"""
    return STATE.to_dict()


@app.get("/tasks")
def list_tasks():
    """List all tasks with grader configuration (OpenEnv compliant)"""
    return {
        "tasks": [
            {
                "name": "classify",
                "description": "Classify transaction as FRAUD or LEGIT",
                "grader": {
                    "type": "reward_based",
                    "reward_range": [0.01, 0.99]
                }
            },
            {
                "name": "identify_type",
                "description": "Identify the type of fraud",
                "grader": {
                    "type": "reward_based",
                    "reward_range": [0.01, 0.99]
                }
            },
            {
                "name": "action_plan",
                "description": "Generate structured fraud mitigation action plan with RISK LEVEL, RECOMMENDED ACTION, NEXT STEPS, and DO NOT sections",
                "grader": {
                    "type": "reward_based",
                    "reward_range": [0.05, 0.85]
                }
            }
        ]
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Root endpoint with HTML dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FraudDetect-Env</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            h1 { margin: 0 0 10px 0; }
            .badge { 
                background: #4CAF50;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 14px;
                display: inline-block;
                margin: 10px 0;
            }
            .endpoints {
                background: rgba(0, 0, 0, 0.2);
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            }
            .endpoint { margin: 10px 0; font-family: monospace; }
            a { color: #FFD700; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ FraudDetect-Env</h1>
            <div class="badge">✓ Running</div>
            <p>Bank Fraud Detection RL Environment for OpenEnv Hackathon</p>
            
            <div class="endpoints">
                <h3>📡 Endpoints:</h3>
                <div class="endpoint">POST /reset - Start new episode</div>
                <div class="endpoint">POST /step - Submit agent response</div>
                <div class="endpoint">GET /tasks - List graded tasks</div>
                <div class="endpoint">GET /state - Current state</div>
                <div class="endpoint">GET /history - Episode history</div>
                <div class="endpoint">GET /health - Health check</div>
            </div>
            
            <p style="margin-top: 20px;">
                <a href="/docs">📖 API Documentation</a> | 
                <a href="/redoc">📚 ReDoc</a>
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(html_content)


def main():
    """Main entry point for running the server"""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )


if __name__ == "__main__":
    main()