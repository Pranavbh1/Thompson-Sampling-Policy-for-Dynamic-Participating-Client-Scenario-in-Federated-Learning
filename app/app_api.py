# fedts_dashboard_fl.py
# Flask-based dashboard + FL loop + malicious client detection + label flipping simulation

from flask import Flask, render_template, jsonify
import random
import torch
import numpy as np
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# ------------------- CONFIG ------------------- #
NUM_CLIENTS = 10
NUM_ORIGINAL = 5
ROUNDS = 20
CLIENTS_PER_ROUND = 4
LABEL_FLIP_ATTACK = [5, 9]  # client IDs to simulate label flipping

# ------------------- DATASET ------------------ #
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
INPUT_SIZE = 28*28

def partition_data(dataset, num_clients):
    label_map = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        label_map[label].append(idx)
    client_data = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        l1, l2 = i % 10, (i+1)%10
        client_data[i] += label_map[l1][:300] + label_map[l2][:300]
        label_map[l1] = label_map[l1][300:]
        label_map[l2] = label_map[l2][300:]
    return client_data

class LabelFlippedDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
    def __getitem__(self, i):
        x, y = self.dataset[self.indices[i]]
        return x, (y + 1) % 10  # label flipping
    def __len__(self):
        return len(self.indices)

# ------------------- MODEL ------------------- #
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def get_path_drift(m1, m2):
    v1 = torch.cat([p.view(-1) for p in m1.parameters()])
    v2 = torch.cat([p.view(-1) for p in m2.parameters()])
    return torch.norm(v1 - v2).item()

# ------------------- FED-TS LOGIC ------------------- #
clients = partition_data(dataset, NUM_CLIENTS)
dataloaders = []
for i, idx in enumerate(clients):
    subset = Subset(dataset, idx)
    if i in LABEL_FLIP_ATTACK:
        dataloaders.append(DataLoader(LabelFlippedDataset(subset), batch_size=32, shuffle=True))
    else:
        dataloaders.append(DataLoader(subset, batch_size=32, shuffle=True))

test_loader = DataLoader(testset, batch_size=128)
global_model = SimpleNN()
beta_params = {i: [1, 1] for i in range(NUM_ORIGINAL, NUM_CLIENTS)}
bad_clients = set()

history = []

def train_local(model, data):
    model = SimpleNN()
    model.load_state_dict(global_model.state_dict())
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for x, y in data:
        opt.zero_grad()
        pred = model(x)
        loss = F.cross_entropy(pred, y)
        loss.backward()
        opt.step()
    return model

def run_federated_round():
    selected = random.sample(range(NUM_ORIGINAL), CLIENTS_PER_ROUND // 2)
    ts_candidates = [cid for cid in range(NUM_ORIGINAL, NUM_CLIENTS) if cid not in bad_clients]
    sampled = sorted(ts_candidates, key=lambda c: np.random.beta(*beta_params[c]), reverse=True)[:CLIENTS_PER_ROUND - len(selected)]
    selected += sampled

    models, drifts = [], []
    for cid in selected:
        local_model = train_local(global_model, dataloaders[cid])
        models.append(local_model)
        if cid >= NUM_ORIGINAL:
            drift = get_path_drift(global_model, local_model)
            drifts.append((cid, drift))

    if len(drifts) >= 2:
        X = np.array([[d] for _, d in drifts])
        kmeans = KMeans(n_clusters=2, n_init=5).fit(X)
        threshold = np.mean(kmeans.cluster_centers_)
        for cid, drift in drifts:
            if drift < threshold:
                beta_params[cid][0] += 1
            else:
                beta_params[cid][1] += 1
                if beta_params[cid][1] > 2:
                    bad_clients.add(cid)
    elif len(drifts) == 1:
        # fallback threshold logic for a single client
        cid, drift = drifts[0]
        if drift < 0.5:  # you can adjust this threshold
            beta_params[cid][0] += 1
        else:
            beta_params[cid][1] += 1
            if beta_params[cid][1] > 2:
                bad_clients.add(cid)


    # aggregate
    state = global_model.state_dict()
    for k in state:
        state[k] = torch.stack([m.state_dict()[k] for m in models], 0).mean(0)
    global_model.load_state_dict(state)
    acc = evaluate(global_model, test_loader)
    history.append({"round": len(history)+1, "accuracy": acc, "bad_clients": list(bad_clients)})

# ------------------- FLASK ROUTES ------------------- #
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/train_one_round")
def train_one():
    run_federated_round()
    return jsonify(history[-1])

@app.route("/history")
def get_history():
    return jsonify(history)

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    with open("templates/dashboard.html", "w") as f:
        f.write("""
        <!doctype html>
        <html>
        <head><title>Fed-TS Dashboard</title></head>
        <body>
        <h1>Fed-TS Federated Learning Simulation</h1>
        <button onclick=fetchRound()>Run Next Round</button>
        <pre id='output'></pre>
        <script>
        async function fetchRound() {
            const res = await fetch('/train_one_round');
            const data = await res.json();
            document.getElementById('output').innerText = JSON.stringify(data, null, 2);
        }
        </script>
        </body>
        </html>
        """)
    app.run(debug=True)
