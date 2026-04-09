from __future__ import annotations
import json
import os
import urllib.request
import urllib.error

# Scaler / LiteLLM proxy config: DO NOT change these lines
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]

MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

SYSTEM_PROMPT = "You are a senior bank fraud analyst AI. Be precise and aggressive at catching fraud."

CLASSIFY_PROMPT = """Transaction:\n{txn_json}\n\nRespond with EXACTLY one word: FRAUD or LEGIT."""

IDENTIFY_TYPE_PROMPT = """Transaction flagged as FRAUD:\n{txn_json}\n\nChoose one:\n- CARD_NOT_PRESENT\n- ACCOUNT_TAKEOVER\n- MONEY_LAUNDERING\n- IDENTITY_THEFT\n- PHISHING\n\nRespond with EXACTLY one label."""

ACTION_PLAN_PROMPT = """Confirmed FRAUD. Type: {fraud_type}\n\nTransaction:\n{txn_json}\n\nWrite:\nRISK LEVEL: HIGH\nRECOMMENDED ACTION: <action>\nNEXT STEPS: 1. <step> 2. <step> 3. <step>\nDO NOT: <avoid>"""

VALID_FRAUD_TYPES = {"CARD_NOT_PRESENT","ACCOUNT_TAKEOVER","MONEY_LAUNDERING","IDENTITY_THEFT","PHISHING"}

def _post_json(url, data=None):
    body = json.dumps(data).encode("utf-8") if data else b""
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def env_reset(): return _post_json(f"{ENV_BASE_URL}/reset")
def env_step(task, response): return _post_json(f"{ENV_BASE_URL}/step", {"task": task, "response": response})

def llm_call(prompt, max_tokens=256):
    url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    body = json.dumps(data).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp_json = json.loads(resp.read().decode("utf-8"))
        return resp_json["choices"][0]["message"]["content"].strip()

def extract_label(raw):
    u = raw.upper()
    return "FRAUD" if "FRAUD" in u else "LEGIT" if "LEGIT" in u else "FRAUD"

def extract_fraud_type(raw):
    u = raw.upper()
    for ft in VALID_FRAUD_TYPES:
        if ft in u: return ft
    return "CARD_NOT_PRESENT"

def main():
    step, rewards = 0, []
    print(f"[START] task=fraud_detection env=fraud_detect_env model={MODEL_NAME}")
    try:
        obs = env_reset().get("observation", {})
        if obs.get("all_processed") or obs.get("done"): return
        txn_json = json.dumps(obs.get("transaction", {}), indent=2)
        detected_type = "CARD_NOT_PRESENT"
        done = False
        while not done:
            current_task = obs.get("task", obs.get("current_task", "done"))
            if current_task == "done": break
            error_str, action = "null", ""
            try:
                if current_task == "classify":
                    action = extract_label(llm_call(CLASSIFY_PROMPT.format(txn_json=txn_json), 10))
                elif current_task == "identify_type":
                    action = extract_fraud_type(llm_call(IDENTIFY_TYPE_PROMPT.format(txn_json=txn_json), 20))
                    detected_type = action
                elif current_task == "action_plan":
                    action = llm_call(ACTION_PLAN_PROMPT.format(txn_json=txn_json, fraud_type=detected_type), 512)
                else:
                    action = "UNKNOWN"
            except Exception as exc:
                action, error_str = "", str(exc)[:120]
            result = env_step(task=current_task, response=action)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            obs = result.get("observation", {})
            rewards.append(reward)
            step += 1
            print(f"[STEP] step={step} action={action.replace(chr(10),' ')[:80]} reward={reward:.2f} done={'true' if done else 'false'} error={error_str}")
            if step >= 10: break
    except Exception as exc:
        print(f"[STEP] step={step+1} action= reward=0.00 done=true error={str(exc)[:120]}")
    finally:
        score = sum(rewards)/len(rewards) if rewards else 0.0
        print(f"[END] success={'true' if score>0.5 else 'false'} steps={step} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards) if rewards else '0.00'}")

if __name__ == "__main__":
    main()