import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_LOG = MODELS_DIR / "metrics_history.jsonl"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class BikeSafetyGNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 64)
        self.conv2 = SAGEConv(64, 32)
        self.conv3 = SAGEConv(32, 16)
        self.head_regression = nn.Linear(16, 1)
        self.head_classification = nn.Linear(16, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv3(h, edge_index))
        reg = self.head_regression(h).squeeze(-1)
        cls = self.head_classification(h).squeeze(-1)
        return reg, cls


def load_graph():
    with open(DATA_DIR / "graph_data.pkl", "rb") as f:
        return pickle.load(f)


def prepare_data(graph_data):
    features = graph_data["features"]
    target = graph_data["target"]
    edge_index = graph_data["edge_index"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features).astype(np.float32)

    binary_target = (target > 0).astype(np.float32)

    n = features.shape[0]
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    return features_scaled, target, binary_target, edge_index, scaler, train_mask, test_mask


def evaluate_binary(y_true, y_prob):
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = float("nan")
    return auc


def train_logistic(features, binary_target, train_mask, test_mask):
    X_train = features[train_mask]
    X_test = features[test_mask]
    y_train = binary_target[train_mask]
    y_test = binary_target[test_mask]

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    all_prob = model.predict_proba(features)[:, 1]

    auc = evaluate_binary(y_test, test_prob)

    return model, all_prob, {"model": "logistic", "auc": auc}


def train_gnn(features, target, binary_target, edge_index, train_mask, test_mask, epochs=200):
    x = torch.FloatTensor(features)
    y_reg = torch.FloatTensor(target)
    y_cls = torch.FloatTensor(binary_target)
    ei = torch.LongTensor(edge_index)
    t_mask = torch.BoolTensor(train_mask)
    te_mask = torch.BoolTensor(test_mask)

    data = Data(x=x, edge_index=ei, y_reg=y_reg, y_cls=y_cls)
    data.train_mask = t_mask
    data.test_mask = te_mask

    pos_weight = torch.tensor([(y_cls == 0).sum() / max((y_cls == 1).sum(), 1)])

    model = BikeSafetyGNN(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        reg_pred, cls_pred = model(data.x, data.edge_index)
        reg_loss = F.mse_loss(reg_pred[data.train_mask], data.y_reg[data.train_mask])
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_pred[data.train_mask], data.y_cls[data.train_mask], pos_weight=pos_weight
        )
        (reg_loss + cls_loss).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        reg_pred, cls_pred = model(data.x, data.edge_index)
        all_risk = torch.sigmoid(cls_pred).numpy()
        all_crash = reg_pred.numpy()

    test_risk = all_risk[test_mask]
    y_test = binary_target[test_mask]
    y_test_reg = target[test_mask]
    test_crash = all_crash[test_mask]

    auc = evaluate_binary(y_test, test_risk)
    rmse = np.sqrt(mean_squared_error(y_test_reg, test_crash))
    r2 = r2_score(y_test_reg, test_crash)

    return model, all_risk, all_crash, {"model": "gnn", "auc": auc, "rmse": rmse, "r2": r2}


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    graph_data = load_graph()
    features, target, binary_target, edge_index, scaler, train_mask, test_mask = prepare_data(graph_data)

    lr_model, lr_risk, lr_metrics = train_logistic(features, binary_target, train_mask, test_mask)
    gnn_model, gnn_risk, gnn_crash, gnn_metrics = train_gnn(
        features, target, binary_target, edge_index, train_mask, test_mask
    )

    print(f"Logistic Regression AUC: {lr_metrics['auc']:.4f}")
    print(f"GNN AUC: {gnn_metrics['auc']:.4f}")
    print(f"GNN RMSE: {gnn_metrics['rmse']:.4f}")
    print(f"GNN R2: {gnn_metrics['r2']:.4f}")

    entry = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "nodes": int(features.shape[0]),
        "features": int(features.shape[1]),
        "lr_auc": round(lr_metrics["auc"], 4),
        "gnn_auc": round(gnn_metrics["auc"], 4),
        "gnn_rmse": round(gnn_metrics["rmse"], 4),
        "gnn_r2": round(gnn_metrics["r2"], 4),
    }
    with open(METRICS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    with open(MODELS_DIR / "logistic.pkl", "wb") as f:
        pickle.dump({"model": lr_model, "risk_scores": lr_risk}, f)

    torch.save({
        "model_state": gnn_model.state_dict(),
        "in_channels": features.shape[1],
        "risk_scores": gnn_risk,
        "crash_pred": gnn_crash,
        "metrics": gnn_metrics,
    }, MODELS_DIR / "gnn_model.pt")

    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
