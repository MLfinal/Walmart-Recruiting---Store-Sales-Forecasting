"""Inventory W&B runs and metric schemas without modifying the project."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import wandb

ENTITY = "kende23-n-a"
PROJECT = "Walmart-Recruiting---Store-Sales-Forecasting"
OUTPUT = Path("/tmp/walmart_wandb_inventory.json")


def json_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return str(value)


def scalar_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): json_value(value)
        for key, value in mapping.items()
        if isinstance(value, (str, bool, int, float)) or value is None
    }


def main() -> None:
    api = wandb.Api(timeout=60)
    records = []
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        summary = scalar_mapping(dict(run.summary))
        config = scalar_mapping(
            {key: value for key, value in run.config.items() if not key.startswith("_")}
        )
        records.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "job_type": run.job_type,
                "group": run.group,
                "tags": list(run.tags or []),
                "created_at": str(run.created_at),
                "url": run.url,
                "summary": json_value(summary),
                "config": json_value(config),
            }
        )
    OUTPUT.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(json.dumps({"runs": len(records), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
