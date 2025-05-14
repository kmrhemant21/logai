import requests
import pandas as pd

LOKI_URL = "https://loki.example.com"
LOKI_QUERY = '{namespace="dev"}'
LIMIT = 10000

def fetch_logs():
    url = f"{LOKI_URL}/loki/api/v1/query_range"
    params = {
        "query": LOKI_QUERY,
        "limit": LIMIT,
        "direction": "BACKWARD"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    log_lines = []
    for stream in data.get("data", {}).get("result", []):
        for entry in stream.get("values", []):
            log_lines.append({
                "timestamp": entry[0],
                "log": entry[1]
            })
    df = pd.DataFrame(log_lines)
    df.to_csv("data/logs.csv", index=False)
    print(f"Fetched {len(df)} logs.")

if __name__ == "__main__":
    fetch_logs()