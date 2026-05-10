import csv
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

STREETS = ["HYDE PARK", "WASHINGTON", "CUMMINS", "CENTRE"]
BOSTON_CITY_CODE = "35"


def analyze_bike_stress():
    path = DATA_DIR / "bike_stress.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))

    for st in STREETS:
        segs = [r for r in rows if st in r.get("St_Name", "").upper()
                and r.get("City", "") == BOSTON_CITY_CODE]
        lts = Counter(r["LTS_define"] for r in segs if r.get("LTS_define"))
        aadt_vals = [int(r["AADT"]) for r in segs
                     if r.get("AADT", "").isdigit() and int(r.get("AADT", 0)) > 0]

        mid = sorted(aadt_vals)[len(aadt_vals) // 2] if aadt_vals else 0
        print(f"{st}: {len(segs)} segs, LTS {dict(sorted(lts.items()))}, "
              f"AADT {min(aadt_vals, default=0):,}-{max(aadt_vals, default=0):,} (med {mid:,})")


def analyze_crashes():
    path = DATA_DIR / "crash_records.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))

    for st in STREETS:
        all_c = [r for r in rows if st in r.get("street", "").upper()]
        bike_c = [r for r in all_c if r["mode_type"] == "bike"]
        print(f"{st}: {len(all_c)} total crashes, {len(bike_c)} bike")


if __name__ == "__main__":
    analyze_bike_stress()
    print()
    analyze_crashes()
