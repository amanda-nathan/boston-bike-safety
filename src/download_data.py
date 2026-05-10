import requests
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "traffic_inventory_2024.csv": (
        "https://geo-massdot.opendata.arcgis.com/api/download/v1/items/"
        "7d1b621999a94265b9a7d6516e68ca63/csv?layers=1"
    ),
    "bike_stress.csv": (
        "https://geo-massdot.opendata.arcgis.com/api/download/v1/items/"
        "71564cab48b34e0e988287345191e2aa/csv?layers=0"
    ),
    "crash_records.csv": (
        "https://data.boston.gov/dataset/7b29c1b2-7ec2-4023-8292-c24f5d8f0905/"
        "resource/e4bfe397-6bfc-49c5-9367-c879fac7401d/download/tmp6uic_r9e.csv"
    ),
}


def download(name, url):
    path = DATA_DIR / name
    if path.exists():
        return
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)


if __name__ == "__main__":
    for name, url in DATASETS.items():
        download(name, url)
