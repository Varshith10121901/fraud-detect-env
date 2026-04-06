"""
rewards_logger.py — Persistent reward storage and analysis

Logs episode results to:
  - JSON files (structured, queryable)
  - CSV file (easy analysis in Excel/Jupyter)
  - Console (real-time feedback)
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Create logs directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

EPISODES_JSON = LOGS_DIR / "episodes.json"
EPISODES_CSV = LOGS_DIR / "episodes.csv"
STATS_FILE = LOGS_DIR / "summary_stats.json"


def save_episode(
    transaction_id: str,
    rewards: List[float],
    steps: int,
    score: float,
    model_name: str = "Unknown",
) -> Dict[str, Any]:
    """Save a single episode result to JSON and CSV."""
    
    timestamp = datetime.now().isoformat()
    episode = {
        "timestamp": timestamp,
        "transaction_id": transaction_id,
        "model": model_name,
        "steps": steps,
        "rewards": rewards,
        "score": score,
        "success": score > 0.5,
    }
    
    # ── Save to JSON ────────────────────────────────────────────
    episodes = []
    if EPISODES_JSON.exists():
        with open(EPISODES_JSON, "r") as f:
            episodes = json.load(f)
    
    episodes.append(episode)
    
    with open(EPISODES_JSON, "w") as f:
        json.dump(episodes, f, indent=2)
    
    print(f"\n✅ [LOG] Episode saved to {EPISODES_JSON.relative_to(LOGS_DIR.parent)}")
    
    # ── Save to CSV ─────────────────────────────────────────────
    csv_exists = EPISODES_CSV.exists()
    
    with open(EPISODES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "transaction_id",
                "model",
                "steps",
                "score",
                "success",
                "rewards",
            ],
        )
        
        if not csv_exists:
            writer.writeheader()
        
        writer.writerow({
            "timestamp": episode["timestamp"],
            "transaction_id": episode["transaction_id"],
            "model": episode["model"],
            "steps": episode["steps"],
            "score": episode["score"],
            "success": episode["success"],
            "rewards": ",".join(f"{r:.2f}" for r in episode["rewards"]),
        })
    
    print(f"✅ [LOG] Episode saved to {EPISODES_CSV.relative_to(LOGS_DIR.parent)}")
    
    # ── Update summary stats ────────────────────────────────────
    _update_stats(episodes)
    
    return episode


def _update_stats(episodes: List[Dict[str, Any]]) -> None:
    """Calculate and save aggregate statistics."""
    
    if not episodes:
        return
    
    scores = [e["score"] for e in episodes]
    successes = sum(1 for e in episodes if e["success"])
    
    stats = {
        "total_episodes": len(episodes),
        "successful_episodes": successes,
        "success_rate": round(successes / len(episodes) * 100, 2),
        "avg_score": round(sum(scores) / len(scores), 3),
        "best_score": round(max(scores), 3),
        "worst_score": round(min(scores), 3),
        "total_steps": sum(e["steps"] for e in episodes),
        "last_updated": datetime.now().isoformat(),
    }
    
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ [LOG] Stats updated: {stats['success_rate']}% success rate "
          f"({stats['successful_episodes']}/{stats['total_episodes']})")


def get_summary() -> Dict[str, Any]:
    """Retrieve current summary statistics."""
    if STATS_FILE.exists():
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return {}


def list_episodes(limit: int = 10) -> List[Dict[str, Any]]:
    """List recent episodes."""
    if not EPISODES_JSON.exists():
        return []
    
    with open(EPISODES_JSON, "r") as f:
        episodes = json.load(f)
    
    return episodes[-limit:] if episodes else []
