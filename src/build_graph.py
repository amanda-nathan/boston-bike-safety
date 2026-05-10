import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import osmnx as ox

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

STUDY_CENTER = (42.296, -71.112)  # Hyde Park Ave / Cummins Hwy area
STUDY_RADIUS = 4000  # meters, covers HP -> Forest Hills -> JP -> West Rox

STUDY_STREETS = ["HYDE PARK", "WASHINGTON", "CUMMINS", "CENTRE"]
BOSTON_CITY = "35"


def load_road_graph():
    cache = PROCESSED / "road_graph.graphml"
    if cache.exists():
        return ox.load_graphml(cache)
    G = ox.graph_from_point(STUDY_CENTER, dist=STUDY_RADIUS, network_type="bike")
    ox.save_graphml(G, cache)
    return G


def load_traffic():
    candidates = sorted(RAW.glob("traffic_inventory_*.csv"), reverse=True)
    df = pd.read_csv(candidates[0], low_memory=False)
    df["City"] = df["City"].astype(str)
    df = df[df["City"] == BOSTON_CITY]
    df["AADT"] = pd.to_numeric(df["AADT"], errors="coerce").fillna(0).astype(int)
    return df


def load_bike_stress():
    df = pd.read_csv(RAW / "bike_stress.csv", low_memory=False)
    df["City"] = df["City"].astype(str)
    df = df[df["City"] == BOSTON_CITY]
    df["LTS_define"] = pd.to_numeric(df["LTS_define"], errors="coerce").fillna(2).astype(int)
    df["AADT"] = pd.to_numeric(df["AADT"], errors="coerce").fillna(0).astype(int)
    df["Speed_Lim"] = pd.to_numeric(df["Speed_Lim"], errors="coerce").fillna(25).astype(int)
    df["Num_Lanes"] = pd.to_numeric(df["Num_Lanes"], errors="coerce").fillna(2).astype(int)
    df["Surface_Wd"] = pd.to_numeric(df["Surface_Wd"], errors="coerce").fillna(24).astype(int)
    return df


def load_crashes():
    df = pd.read_csv(RAW / "crash_records.csv", low_memory=False)
    df = df.dropna(subset=["lat", "long"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df = df.dropna(subset=["lat", "long"])
    return df


def match_street_name(edge_name, study_streets=STUDY_STREETS):
    if not edge_name:
        return None
    if isinstance(edge_name, list):
        edge_name = edge_name[0]
    upper = edge_name.upper()
    for st in study_streets:
        if st in upper:
            return st
    return None


def assign_crashes_to_edges(G, crashes):
    bike_counts = defaultdict(int)
    mv_counts = defaultdict(int)
    ped_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for _, row in crashes.iterrows():
        nearest = ox.nearest_nodes(G, row["long"], row["lat"])
        total_counts[nearest] += 1
        mode = row.get("mode_type", "")
        if mode == "bike":
            bike_counts[nearest] += 1
        elif mode == "mv":
            mv_counts[nearest] += 1
        elif mode == "ped":
            ped_counts[nearest] += 1

    return dict(bike_counts), dict(mv_counts), dict(ped_counts), dict(total_counts)


def neighbor_crash_sum(G, counts, node, hops=2):
    visited = {node}
    frontier = {node}
    total = counts.get(node, 0)
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            for neighbor in G.neighbors(n):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    total += counts.get(neighbor, 0)
        frontier = next_frontier
    return total


def build_node_features(G, stress_df, bike_counts, total_counts):
    nodes = list(G.nodes())
    features = []

    street_aadt = {}
    street_lts = {}
    street_speed = {}
    street_lanes = {}
    street_width = {}
    for _, row in stress_df.iterrows():
        name = row.get("St_Name", "")
        if not name:
            continue
        upper = name.upper()
        street_aadt[upper] = max(street_aadt.get(upper, 0), row["AADT"])
        street_lts[upper] = max(street_lts.get(upper, 0), row["LTS_define"])
        street_speed[upper] = max(street_speed.get(upper, 0), row["Speed_Lim"])
        street_lanes[upper] = max(street_lanes.get(upper, 0), row["Num_Lanes"])
        street_width[upper] = max(street_width.get(upper, 0), row["Surface_Wd"])

    node_street_names = {}

    for node in nodes:
        edges = G.edges(node, data=True)
        max_aadt = 0
        max_lts = 1
        max_speed = 25
        max_lanes = 1
        max_width = 24
        total_length = 0
        n_edges = 0
        has_oneway = 0
        names = set()

        for u, v, data in edges:
            name = data.get("name", "")
            if isinstance(name, list):
                for n in name:
                    if n:
                        names.add(n)
                name = name[0] if name else ""
            elif name:
                names.add(name)
            upper = name.upper() if name else ""

            if upper in street_aadt:
                max_aadt = max(max_aadt, street_aadt[upper])
                max_lts = max(max_lts, street_lts.get(upper, 1))
                max_speed = max(max_speed, street_speed.get(upper, 25))
                max_lanes = max(max_lanes, street_lanes.get(upper, 1))
                max_width = max(max_width, street_width.get(upper, 24))

            if data.get("oneway", False):
                has_oneway = 1

            total_length += data.get("length", 0)
            n_edges += 1

        avg_length = total_length / max(n_edges, 1)
        neighbor_bike = neighbor_crash_sum(G, bike_counts, node, hops=2)
        neighbor_total = neighbor_crash_sum(G, total_counts, node, hops=2)
        node_street_names[node] = " / ".join(sorted(names)[:3]) if names else ""

        features.append([
            max_aadt,
            max_lts,
            max_speed,
            max_lanes,
            max_width,
            avg_length,
            n_edges,
            has_oneway,
            neighbor_bike,
            neighbor_total,
            bike_counts.get(node, 0),
            total_counts.get(node, 0),
        ])

    return np.array(features, dtype=np.float32), node_street_names


def build_edge_index(G):
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    sources = []
    targets = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            sources.append(node_to_idx[u])
            targets.append(node_to_idx[v])
    return np.array([sources, targets], dtype=np.int64)


def main():
    G = load_road_graph()
    stress = load_bike_stress()
    crashes = load_crashes()

    bike_counts, mv_counts, ped_counts, total_counts = assign_crashes_to_edges(G, crashes)
    node_features, node_street_names = build_node_features(G, stress, bike_counts, total_counts)
    edge_index = build_edge_index(G)

    nodes = list(G.nodes())
    mv_per_node = np.array([mv_counts.get(n, 0) for n in nodes], dtype=np.float32)
    ped_per_node = np.array([ped_counts.get(n, 0) for n in nodes], dtype=np.float32)

    target = node_features[:, 10]  # bike crash count per node
    features = node_features[:, :10]  # everything except crash counts

    graph_data = {
        "features": features,
        "target": target,
        "total_crashes": node_features[:, 11],
        "mv_crashes": mv_per_node,
        "ped_crashes": ped_per_node,
        "edge_index": edge_index,
        "nodes": nodes,
        "node_coords": {n: (G.nodes[n]["y"], G.nodes[n]["x"]) for n in G.nodes()},
        "node_streets": node_street_names,
        "feature_names": [
            "aadt", "lts", "speed_limit", "lanes", "surface_width",
            "avg_edge_length", "degree", "has_oneway",
            "neighbor_bike_crashes", "neighbor_total_crashes",
        ],
    }

    with open(PROCESSED / "graph_data.pkl", "wb") as f:
        pickle.dump(graph_data, f)


if __name__ == "__main__":
    main()
