import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
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

    x = torch.FloatTensor(features_scaled)
    y_reg = torch.FloatTensor(target)
    y_cls = torch.FloatTensor(binary_target)
    ei = torch.LongTensor(edge_index)

    data = Data(x=x, edge_index=ei, y_reg=y_reg, y_cls=y_cls)

    n = x.shape[0]
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    data.train_mask = train_mask
    data.test_mask = test_mask

    return data, scaler


def train(data, epochs=200):
    model = BikeSafetyGNN(data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    pos_weight = torch.tensor([(data.y_cls == 0).sum() / max((data.y_cls == 1).sum(), 1)])

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reg_pred, cls_pred = model(data.x, data.edge_index)

        reg_loss = F.mse_loss(reg_pred[data.train_mask], data.y_reg[data.train_mask])
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_pred[data.train_mask], data.y_cls[data.train_mask], pos_weight=pos_weight
        )
        loss = reg_loss + cls_loss

        loss.backward()
        optimizer.step()

    return model


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        reg_pred, cls_pred = model(data.x, data.edge_index)

    mask = data.test_mask
    y_true_reg = data.y_reg[mask].numpy()
    y_pred_reg = reg_pred[mask].numpy()
    y_true_cls = data.y_cls[mask].numpy()
    y_pred_cls = torch.sigmoid(cls_pred[mask]).numpy()

    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    r2 = r2_score(y_true_reg, y_pred_reg)

    if len(np.unique(y_true_cls)) > 1:
        auc = roc_auc_score(y_true_cls, y_pred_cls)
    else:
        auc = float("nan")

    return {"rmse": rmse, "r2": r2, "auc": auc}


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    graph_data = load_graph()
    data, scaler = prepare_data(graph_data)

    model = train(data, epochs=200)
    metrics = evaluate(model, data)

    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")

    entry = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "nodes": int(data.x.shape[0]),
        "features": int(data.x.shape[1]),
        "train_size": int(data.train_mask.sum()),
        "test_size": int(data.test_mask.sum()),
        **{k: round(v, 4) for k, v in metrics.items()},
    }
    with open(METRICS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    torch.save({
        "model_state": model.state_dict(),
        "in_channels": data.x.shape[1],
        "metrics": metrics,
    }, MODELS_DIR / "gnn_model.pt")

    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
