"""Verify a running PathSnap server over HTTP and save observed responses.

Run this client on Laptop 2 against Laptop 1. This script does not capture
screenshots or claim that a Streamlit page was rendered. Existing drawn figures
are historical illustrations, not deployment evidence.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests


def verify_deployment(server_url, trip_id=None, partition=None, output_dir=None):
    server_url = server_url.rstrip("/")
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    health_response = requests.get(f"{server_url}/health", timeout=30)
    health_response.raise_for_status()
    health = health_response.json()
    if health.get("status") != "online":
        raise ValueError("Server did not report online status.")
    trips_response = requests.get(f"{server_url}/trips", timeout=30)
    trips_response.raise_for_status()
    trips = trips_response.json()["trips"]
    choices = [t for t in trips if (trip_id is None or t["trip_id"] == trip_id)
               and (partition is None or t["partition"] == partition)]
    if not choices:
        raise ValueError("No matching evaluation trip was advertised by the server.")
    target = choices[0]
    payload = {"trip_id": target["trip_id"], "partition": target["partition"], "include_map_html": False}
    evaluation_response = requests.post(f"{server_url}/evaluate", json=payload, timeout=120)
    evaluation_response.raise_for_status()
    result = evaluation_response.json()
    if result.get("status") != "success" or any(result.get(k) != target[k] for k in ("trip_id", "partition")):
        raise ValueError("Unexpected evaluation response.")
    evidence = {
        "kind": "http_client_verification",
        "observed_at_utc": started,
        "server_url": server_url,
        "total_client_ms": (time.perf_counter() - t0) * 1000,
        "health": health,
        "advertised_trip_count": len(trips),
        "evaluation": result,
        "scope": "HTTP endpoints only; does not verify Streamlit rendering or a second physical laptop.",
    }
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "http_verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--trip-id")
    parser.add_argument("--partition", choices=["student_test", "evaluator_test"])
    parser.add_argument("--output-dir", default="outputs/deployment")
    args = parser.parse_args()
    evidence = verify_deployment(args.server_url, args.trip_id, args.partition, args.output_dir)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
