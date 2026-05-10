# Bike Crash Risk: Hyde Park to Forest Hills Corridor

Crash risk prediction for cyclists in the Hyde Park, Roslindale, Jamaica Plain, and West Roxbury neighborhoods of Boston. Two models are provided: a logistic regression baseline and a Graph Neural Network. The live map lets you toggle between them.

**[View the live risk map](https://amanda-nathan.github.io/boston-bike-safety/)** (updated monthly)

Data as of 2026-05-10. 42,684 crash records (3,896 bike) from 2015-2025. Current metrics are on the [live site](https://amanda-nathan.github.io/boston-bike-safety/).

## What this does

Predicts which intersections are most dangerous for cyclists along the commute corridor from Hyde Park to Forest Hills, using road infrastructure features (traffic volume, stress rating, speed, lanes, surface width) and historical crash locations.

## Study corridor

| Street | AADT | Bike LTS | Bike crashes (2015-2025) |
|--------|------|----------|--------------------------|
| Hyde Park Ave | 1,154 - 28,032 | 1-4 | 19 |
| Washington St | 1,154 - 41,570 | 1-5 | 87 |
| Cummins Hwy | 16,769 - 28,032 | 2-4 | 5 |
| Centre St | 484 - 33,898 | 1-4 | 48 |

## Acronyms

- **AADT**: Annual Average Daily Traffic. Vehicles per day averaged over the year. Source: [MassDOT Traffic Inventory](https://geo-massdot.opendata.arcgis.com/datasets/MassDOT::traffic-inventory-2024).
- **LTS**: Level of Traffic Stress. A 1-5 rating of cycling comfort based on volume, speed, lanes, and bike infrastructure. Developed by [Mekuria, Furth & Nixon (2012)](https://transweb.sjsu.edu/research/low-stress-bicycling-and-network-connectivity) at Northeastern University.
- **GNN**: Graph Neural Network. A neural network that operates on graph-structured data (nodes = intersections, edges = road segments).
- **GraphSAGE**: Graph Sample and Aggregate. A GNN architecture that learns node representations by aggregating features from neighbors ([Hamilton, Ying & Leskovec, 2017](https://arxiv.org/abs/1706.02216)).
- **AUC**: Area Under the ROC Curve. Probability that the model ranks a crash intersection higher than a non-crash intersection.
- **RMSE**: Root Mean Squared Error. Average prediction error in crash count units.

## Data sources

1. **[MassDOT Traffic Inventory](https://geo-massdot.opendata.arcgis.com/datasets/MassDOT::traffic-inventory-2024)**: Road segments with AADT, lane count, road classification. Updated yearly.

2. **[MassDOT Bike Level of Traffic Stress](https://geo-massdot.opendata.arcgis.com/datasets/MassDOT::bike-level-of-traffic-stress)**: Road segments with LTS score, speed limit, surface width, shoulder type. Methodology: [Mekuria, Furth & Nixon (2012)](https://transweb.sjsu.edu/research/low-stress-bicycling-and-network-connectivity).

3. **[Boston Vision Zero Crash Records](https://data.boston.gov/dataset/vision-zero-crash-records)**: Crash dispatch records with timestamp, lat/lon, and mode (bike, motor vehicle, pedestrian). Updated monthly.

4. **[OpenStreetMap](https://www.openstreetmap.org/)** via [OSMnx](https://osmnx.readthedocs.io/) ([Boeing, 2017](https://doi.org/10.1016/j.compenvurbsys.2017.05.004)): Road network graph for the study area.

## Method

### Features

Each intersection $v$ in the road network has a feature vector $\mathbf{x}_v \in \mathbb{R}^8$:

$$\mathbf{x}_v = \begin{bmatrix} \text{AADT}_v \\ \text{LTS}_v \\ \text{SpeedLimit}_v \\ \text{Lanes}_v \\ \text{SurfaceWidth}_v \\ \text{AvgEdgeLength}_v \\ \text{Degree}_v \\ \text{HasOneway}_v \end{bmatrix}$$

These are road infrastructure features only. Crash history is used as the target, not as an input, so the model scores risk based on what the road looks like rather than whether cyclists have already crashed there. A dangerous road that cyclists avoid (few recorded crashes) still gets a high risk score.

### Model 1: Logistic Regression (baseline)

Each intersection is classified independently:

$$P(\text{crash}_v = 1 \mid \mathbf{x}_v) = \sigma(\mathbf{w}^\top \mathbf{x}_v + b)$$

where $\sigma$ is the sigmoid function. Class weights are balanced inversely by frequency to handle the sparsity of crash events. This model is simple, interpretable, and treats every intersection as if it exists in isolation.

### Model 2: Graph Neural Network (GraphSAGE)

The road network is modeled as a directed graph $G = (V, E)$ where nodes are intersections and edges are road segments. The GNN learns that an intersection's risk depends not just on its own features but on its neighbors in the road network.

Three [GraphSAGE](https://arxiv.org/abs/1706.02216) convolution layers aggregate neighbor information:

$$\mathbf{h}_v^{(l+1)} = \text{ReLU}\left(\mathbf{W}^{(l)} \cdot \left[\mathbf{h}_v^{(l)} \| \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)} \right]\right)$$

The node's own embedding $\mathbf{h}_v^{(l)}$ is concatenated ($\|$) with the mean of its neighbors' embeddings, then transformed by a learned weight matrix. The layer dimensions are $8 \to 64 \to 32 \to 16$ with dropout between layers.

The final embedding feeds two heads: one predicts crash probability, the other predicts crash count:

$$\hat{p}_v = \sigma(\mathbf{w}_c^\top \mathbf{h}_v^{(3)} + b_c) \qquad \hat{y}_v = \mathbf{w}_r^\top \mathbf{h}_v^{(3)} + b_r$$

### Why two models

Logistic regression is a standard baseline in risk modeling. It's interpretable and fast. The GNN adds the ability to propagate risk through the road network: a low-traffic intersection surrounded by high-traffic arterials gets a higher risk score than the same intersection in a quiet neighborhood. This is the graph analog of diffusion, where danger spreads along connected road segments.

The live site lets you toggle between the two to see where they agree and where they differ. Where they disagree is often the most interesting: it reveals intersections where network context changes the risk picture.

## Project map

```
boston-bike-safety/
├── src/
│   ├── download_data.py     Pulls latest data from MassDOT and Boston open data
│   ├── build_graph.py       Constructs road graph, matches features and crashes to nodes
│   ├── train_gnn.py         Trains GraphSAGE model with dual regression/classification heads
│   ├── build_site.py        Generates interactive Plotly risk map for GitHub Pages
│   ├── build_readme.py      Regenerates this README with current data stats
│   ├── explore_streets.py   Street-level data exploration
│   └── pipeline.py          Runs the full pipeline end to end
├── .github/workflows/
│   └── nightly.yml          GitHub Actions: re-runs pipeline on the 1st of each month
├── docs/
│   └── index.html           Live risk map (auto-generated, do not edit)
└── pyproject.toml            Dependencies managed by uv
```

## Scheduled updates

The pipeline runs automatically on the 1st of each month via GitHub Actions. Crash data from Boston updates monthly and traffic inventory from MassDOT updates yearly, so monthly runs catch new data without wasting compute on unchanged inputs. Each run downloads the latest data, rebuilds the graph, retrains the model, publishes the updated risk map, and regenerates this README. Metrics are logged to `models/metrics_history.jsonl` so model performance can be tracked over time.

## Run locally

```bash
uv sync
uv run python src/pipeline.py
open docs/index.html
```

## Future work

- **Radar speed sign data**: Several streets in the study area (including Hyde Park Ave) have radar speed feedback signs that record actual vehicle speeds. This data is not currently published by the City of Boston. [Cambridge, MA](https://data.cambridgema.gov/Traffic-Parking-and-Transportation/Traffic-Speed-Studies-Historical-/k7qr-u489/data) and [Edmonton, AB](https://data.edmonton.ca/stories/s/Driver-Feedback-Signs-DFS-/byrh-7jdx/) publish comparable data. Actual speeds (vs posted speed limits) would improve risk prediction, especially on streets where speeding is common.
- **Temporal modeling**: Current model is static. Adding time-of-day crash patterns would enable route recommendations by departure time.
- **Bicycle counter data**: Boston and MassDOT are expanding bike counter installations. Exposure-adjusted crash rates (crashes per cyclist-mile) would better distinguish "dangerous road" from "road nobody rides."

## References

1. Mekuria, M., Furth, P. G., & Nixon, H. (2012). [Low-Stress Bicycling and Network Connectivity](https://transweb.sjsu.edu/research/low-stress-bicycling-and-network-connectivity). Mineta Transportation Institute Report 11-19.

2. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216). NeurIPS 2017.

3. Boeing, G. (2017). [OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks](https://doi.org/10.1016/j.compenvurbsys.2017.05.004). Computers, Environment and Urban Systems, 65, 126-139.

4. Kipf, T. N., & Welling, M. (2017). [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907). ICLR 2017.
