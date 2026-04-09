def grade_classify(response, truth):
    if not response:
        return 0.05
    return 0.95 if str(response).strip().upper() == str(truth.get("label", "")).upper() else 0.05

def grade_identify(response, truth):
    if not response:
        return 0.05
    return 0.90 if str(response).strip().upper() == str(truth.get("fraud_type", "")).upper() else 0.05

def grade_action_plan(response, truth):
    if not response:
        return 0.05
    keys = ["RISK LEVEL", "RECOMMENDED ACTION", "NEXT STEPS", "DO NOT"]
    matches = sum(1 for k in keys if k in str(response).upper())
    return round(min(0.85, max(0.05, 0.05 + 0.20 * matches)), 2)
