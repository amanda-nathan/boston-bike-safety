import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import plotly.graph_objects as go

from train_gnn import BikeSafetyGNN

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_LOG = MODELS_DIR / "metrics_history.jsonl"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions():
    with open(DATA_DIR / "graph_data.pkl", "rb") as f:
        graph_data = pickle.load(f)

    state = torch.load(MODELS_DIR / "gnn_model.pt", weights_only=False)
    gnn_risk = state["risk_scores"]
    gnn_crash = state["crash_pred"]
    gnn_metrics = state["metrics"]

    with open(MODELS_DIR / "logistic.pkl", "rb") as f:
        lr_data = pickle.load(f)
    lr_risk = lr_data["risk_scores"]

    return graph_data, gnn_risk, gnn_crash, gnn_metrics, lr_risk


def load_crash_stats():
    import csv
    crash_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "crash_records.csv"
    with open(crash_path) as f:
        rows = list(csv.DictReader(f))

    years = sorted(set(r["dispatch_ts"][:4] for r in rows if r.get("dispatch_ts")))
    date_range = f"{years[0]}-{years[-1]}"
    bike = [r for r in rows if r["mode_type"] == "bike"]

    street_counts = {}
    for st in ["HYDE PARK", "WASHINGTON", "CUMMINS", "CENTRE"]:
        street_counts[st] = len([r for r in bike if st in r.get("street", "").upper()])

    return date_range, len(rows), len(bike), street_counts


def build_risk_map(graph_data, risk_score, crash_pred, metrics, label=""):
    coords = graph_data["node_coords"]
    nodes = graph_data["nodes"]
    features = graph_data["features"]
    actual_bike = graph_data["target"]
    actual_mv = graph_data["mv_crashes"]
    actual_ped = graph_data["ped_crashes"]
    node_streets = graph_data.get("node_streets", {})

    lats = [coords[n][0] for n in nodes]
    lons = [coords[n][1] for n in nodes]

    hover_text = []
    for i, n in enumerate(nodes):
        street = node_streets.get(n, "")
        street_line = f"<b>{street}</b><br>" if street else ""
        text = (
            f"{street_line}"
            f"Risk Score: {risk_score[i]:.2f}<br>"
            f"Predicted Bike Crashes: {crash_pred[i]:.1f}<br>"
            f"<br>"
            f"<b>Actual Crashes</b><br>"
            f"Bike: {actual_bike[i]:.0f}<br>"
            f"Motor Vehicle: {actual_mv[i]:.0f}<br>"
            f"Pedestrian: {actual_ped[i]:.0f}<br>"
            f"<br>"
            f"<b>Road</b><br>"
            f"AADT (vehicles/day): {features[i][0]:,.0f}<br>"
            f"LTS (1=low stress, 4=high): {features[i][1]:.0f}<br>"
            f"Speed Limit: {features[i][2]:.0f} mph<br>"
            f"Lanes: {features[i][3]:.0f}"
        )
        hover_text.append(text)

    fig = go.Figure()

    ei = graph_data["edge_index"]
    edge_lats = []
    edge_lons = []
    for i in range(ei.shape[1]):
        src, dst = ei[0, i], ei[1, i]
        src_node = nodes[src]
        dst_node = nodes[dst]
        edge_lats.extend([coords[src_node][0], coords[dst_node][0], None])
        edge_lons.extend([coords[src_node][1], coords[dst_node][1], None])

    fig.add_trace(go.Scattermap(
        lat=edge_lats, lon=edge_lons,
        mode="lines",
        line=dict(width=0.5, color="#888"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Scattermap(
        lat=lats, lon=lons,
        mode="markers",
        marker=dict(
            size=6,
            color=risk_score,
            colorscale="RdYlGn_r",
            cmin=0, cmax=1,
            colorbar=dict(title="Risk Score"),
            opacity=0.8,
        ),
        text=hover_text,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=f"{label} - Bike Crash Risk: Hyde Park to Forest Hills (updated {datetime.now().strftime('%Y-%m-%d')})",
        map=dict(
            style="carto-positron",
            center=dict(lat=42.296, lon=-71.112),
            zoom=13,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        width=1000,
        height=700,
    )

    return fig


def load_metrics_history():
    if not METRICS_LOG.exists():
        return []
    entries = []
    for line in METRICS_LOG.read_text().strip().split("\n"):
        if line:
            entries.append(json.loads(line))
    return entries


def build_history_table(history):
    if not history:
        return ""
    rows = ""
    for entry in history[-30:]:
        rows += (
            f"<tr>"
            f"<td>{entry['date']}</td>"
            f"<td>{entry.get('auc', 'n/a')}</td>"
            f"<td>{entry.get('rmse', 'n/a')}</td>"
            f"<td>{entry.get('r2', 'n/a')}</td>"
            f"<td>{entry.get('nodes', 'n/a')}</td>"
            f"</tr>\n"
        )
    return f"""
<div class="streets">
<h3>Model Performance History</h3>
<table>
<tr><th>Date</th><th>AUC</th><th>RMSE</th><th>R2</th><th>Nodes</th></tr>
{rows}
</table>
</div>"""


def build_html(gnn_fig, lr_fig, gnn_metrics, lr_auc):
    gnn_html = gnn_fig.to_html(full_html=False, include_plotlyjs="cdn")
    lr_html = lr_fig.to_html(full_html=False, include_plotlyjs=False)
    history = load_metrics_history()
    history_table = build_history_table(history)
    date_range, total_crashes, total_bike, street_counts = load_crash_stats()

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Bike Crash Risk: Hyde Park to Forest Hills</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
.container {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ color: #333; }}
.metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
.metric {{ background: white; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.metric .value {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
.metric .label {{ font-size: 13px; color: #666; }}
.map-container {{ background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 10px; margin: 20px 0; }}
.streets {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 20px 0; }}
.streets table {{ width: 100%; border-collapse: collapse; }}
.streets th, .streets td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
.streets th {{ color: #666; font-weight: 600; }}
.toggle {{ margin: 20px 0; }}
.toggle button {{ padding: 10px 20px; border: 2px solid #2563eb; background: white; color: #2563eb; cursor: pointer; font-size: 14px; font-weight: 600; }}
.toggle button.active {{ background: #2563eb; color: white; }}
.toggle button:first-child {{ border-radius: 6px 0 0 6px; }}
.toggle button:last-child {{ border-radius: 0 6px 6px 0; }}
footer {{ color: #999; font-size: 12px; margin-top: 40px; }}
</style>
</head>
<body>
<div class="container">
<h1>Bike Crash Risk: Hyde Park to Forest Hills Corridor</h1>
<p>Predicted crash risk for intersections in the Hyde Park / Roslindale / Jamaica Plain / West Roxbury corridor.
Red = higher risk. Hover for details. Trained on {total_crashes:,} crash dispatch records ({total_bike:,} bike) from {date_range}.</p>
<p style="font-size: 13px; color: #666;">Crash data from Boston Vision Zero are dispatch records (when public safety responded to a crash location).
Severity and fatality information is not included in this dataset.</p>

<div class="toggle">
<button class="active" onclick="showModel('gnn')">GNN (GraphSAGE)</button><button onclick="showModel('lr')">Logistic Regression</button>
</div>

<div class="metrics">
<div class="metric"><div class="value" id="auc-val">{gnn_metrics['auc']:.3f}</div><div class="label">AUC (classification)</div></div>
<div class="metric"><div class="value" id="rmse-val">{gnn_metrics['rmse']:.3f}</div><div class="label">RMSE (crash count)</div></div>
<div class="metric"><div class="value" id="r2-val">{gnn_metrics['r2']:.3f}</div><div class="label">R-squared</div></div>
</div>

<div class="map-container" id="map-gnn">
{gnn_html}
</div>
<div class="map-container" id="map-lr" style="display:none;">
{lr_html}
</div>

<script>
var gnnMetrics = {{auc: "{gnn_metrics['auc']:.3f}", rmse: "{gnn_metrics['rmse']:.3f}", r2: "{gnn_metrics['r2']:.3f}"}};
var lrMetrics = {{auc: "{lr_auc:.3f}", rmse: "n/a", r2: "n/a"}};
function showModel(m) {{
    document.getElementById('map-gnn').style.display = m === 'gnn' ? 'block' : 'none';
    document.getElementById('map-lr').style.display = m === 'lr' ? 'block' : 'none';
    var met = m === 'gnn' ? gnnMetrics : lrMetrics;
    document.getElementById('auc-val').textContent = met.auc;
    document.getElementById('rmse-val').textContent = met.rmse;
    document.getElementById('r2-val').textContent = met.r2;
    document.querySelectorAll('.toggle button').forEach(function(b) {{ b.classList.remove('active'); }});
    event.target.classList.add('active');
}}
</script>

<div class="streets">
<h3>Study Corridor</h3>
<table>
<tr><th>Street</th><th>Bike Crashes ({date_range})</th><th>Notes</th></tr>
<tr><td>Hyde Park Ave</td><td>{street_counts.get('HYDE PARK', 0)}</td><td>Primary commute route, Hyde Park to Forest Hills</td></tr>
<tr><td>Washington St</td><td>{street_counts.get('WASHINGTON', 0)}</td><td>Major arterial through Roslindale to JP</td></tr>
<tr><td>Cummins Hwy</td><td>{street_counts.get('CUMMINS', 0)}</td><td>27K cars/day on 2 lanes</td></tr>
<tr><td>Centre St</td><td>{street_counts.get('CENTRE', 0)}</td><td>West Roxbury corridor</td></tr>
</table>
</div>

<div class="streets">
<h3>Key Terms</h3>
<table>
<tr><th>Term</th><th>Definition</th></tr>
<tr><td>AADT</td><td>Annual Average Daily Traffic. Number of vehicles passing a road segment per day, averaged over the year. [1]</td></tr>
<tr><td>LTS</td><td>Level of Traffic Stress. A 1-5 rating of cycling comfort based on traffic volume, speed, lanes, and bike infrastructure.
1 = safe for all ages, 2 = most adults, 3 = experienced cyclists, 4 = high stress, 5 = no bike access. [2]</td></tr>
</table>
</div>

{history_table}

<div class="streets">
<h3>References</h3>
<ol style="font-size: 13px; color: #555; line-height: 1.8;">
<li><a href="https://geo-massdot.opendata.arcgis.com/datasets/MassDOT::traffic-inventory-2024">MassDOT Traffic Inventory 2024</a>. Massachusetts Department of Transportation.</li>
<li>Mekuria, M., Furth, P. G., &amp; Nixon, H. (2012). <a href="https://transweb.sjsu.edu/research/low-stress-bicycling-and-network-connectivity">Low-Stress Bicycling and Network Connectivity</a>. Mineta Transportation Institute Report 11-19, Northeastern University.</li>
<li><a href="https://data.boston.gov/dataset/vision-zero-crash-records">Boston Vision Zero Crash Records</a>. City of Boston. Updated monthly.</li>
<li>Hamilton, W. L., Ying, R., &amp; Leskovec, J. (2017). <a href="https://arxiv.org/abs/1706.02216">Inductive Representation Learning on Large Graphs</a>. NeurIPS 2017.</li>
<li>Boeing, G. (2017). <a href="https://doi.org/10.1016/j.compenvurbsys.2017.05.004">OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks</a>. Computers, Environment and Urban Systems, 65, 126-139.</li>
<li><a href="https://geo-massdot.opendata.arcgis.com/datasets/MassDOT::bike-level-of-traffic-stress">MassDOT Bike Level of Traffic Stress</a>. Massachusetts Department of Transportation.</li>
</ol>
</div>

<footer>
Model: GraphSAGE [4] on road network from OpenStreetMap [5]. Updated monthly via GitHub Actions.
<a href="https://github.com/amanda-nathan/boston-bike-safety">Source code on GitHub</a>.
</footer>
</div>
</body>
</html>"""

    with open(DOCS_DIR / "index.html", "w") as f:
        f.write(html)


def main():
    graph_data, gnn_risk, gnn_crash, gnn_metrics, lr_risk = load_predictions()
    gnn_fig = build_risk_map(graph_data, gnn_risk, gnn_crash, gnn_metrics, label="GNN (GraphSAGE)")
    lr_fig = build_risk_map(graph_data, lr_risk, gnn_crash, gnn_metrics, label="Logistic Regression")

    with open(MODELS_DIR / "logistic.pkl", "rb") as f:
        lr_data = pickle.load(f)

    metrics_log = load_metrics_history()
    lr_auc = metrics_log[-1]["lr_auc"] if metrics_log else 0.0

    build_html(gnn_fig, lr_fig, gnn_metrics, lr_auc)


if __name__ == "__main__":
    main()
