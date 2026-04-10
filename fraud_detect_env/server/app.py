from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="FraudDetect-Env")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def list_tasks():
    """✅ Hackathon Phase 2 compliant - 3 tasks with inline graders"""
    return {
        "tasks": [
            {
                "id": "classify",
                "name": "Classify Transaction",
                "description": "Classify a bank transaction as FRAUD or LEGIT",
                "difficulty": "easy",
                "max_steps": 1,
                "reward_range": [0.01, 0.99],
                "grader": {
                    "type": "exact_match",
                    "key": "classification.verdict"
                }
            },
            {
                "id": "identify_type",
                "name": "Identify Fraud Type",
                "description": "Identify the specific type of fraud",
                "difficulty": "medium",
                "max_steps": 1,
                "reward_range": [0.01, 0.99],
                "grader": {
                    "type": "contains",
                    "key": "fraud_type"
                }
            },
            {
                "id": "action_plan",
                "name": "Generate Action Plan",
                "description": "Generate a structured fraud mitigation action plan",
                "difficulty": "hard",
                "max_steps": 1,
                "reward_range": [0.05, 0.85],
                "grader": {
                    "type": "keyword",
                    "keywords": [
                        "RISK LEVEL",
                        "RECOMMENDED ACTION",
                        "NEXT STEPS",
                        "DO NOT"
                    ]
                }
            }
        ]
    }

# Optional: keep other endpoints if they exist
@app.get("/")
def root():
    return {"message": "Fraud Detection Environment - Running"}
