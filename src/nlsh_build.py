#!/usr/bin/env python3
import os
import pickle #Python’s binary serialize/deserialize

import numpy as np #is the main numeric array
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import kahip

from data_reader import *


# ---------- Simple MLP (later parameterize layers/nodes) ----------
class MLPClassifier(nn.Module):
    def __init__(self, d_in: int, m_out: int, hidden_nodes: int, num_layers: int):
        super().__init__()
        """
        d_in = dimensions input
        m_out = number of partitions
        hidden_nodes = width of each hidden layer 
        num_layers = No. of layers (hidden + output)
        super().init() = initializes the parent class (nn.Module).
        """
        layers = []
        in_dim = d_in
        for _ in range(num_layers - 1):      # hidden layers
            layers.append(nn.Linear(in_dim, hidden_nodes))
            layers.append(nn.ReLU())
            in_dim = hidden_nodes
        # final layer to m_out logits
        layers.append(nn.Linear(in_dim, m_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

# ---------- Step 1: build k-NN graph ----------
def build_knn_graph(dataset: np.ndarray, k: int):
    n, d = dataset.shape
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nbrs.fit(dataset)
    # indices: (n, k+1) including self as first neighbor
    distances, indices = nbrs.kneighbors(dataset)

    # remove self (index 0)
    indices = indices[:, 1:]   # (n, k)
    return indices  # neighbors[i] = list of k neighbors of i


# ---------- Step 2: symmetrize graph + weights ----------
def build_weighted_edges(neighbors: np.ndarray):
    """
    n = No. of Neighbours
    neighbors[i] = list of directed neighbors j
    Return undirected edges with weights 1 or 2:
      - 2 if mutual neighbors
      - 1 otherwise
    """
    n, k = neighbors.shape
    # store directed edges in a set for quick lookup
    directed = set()
    for i in range(n): 
        for j in neighbors[i]:
            directed.add((i, int(j)))

    # build undirected weighted edges
    edge_dict = {}  # key: (min(i,j), max(i,j)), value: weight
    for i in range(n):
        for j in neighbors[i]:
            j = int(j)
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in edge_dict:
                # check if mutual
                mutual = (j, i) in directed
                edge_dict[(a, b)] = 2 if mutual else 1

    return edge_dict  # { (i,j): w }


# ---------- Step 3: convert to CSR for KaHIP ----------
def edges_to_csr(n: int, edge_dict):
    """
    Build CSR arrays: xadj, adjncy, adjcwgt, vwgt
    Graph is undirected; for each edge (i,j) we store both directions.
    """
    # adjacency list first
    adj = [[] for _ in range(n)]
    wts = [[] for _ in range(n)]

    for (i, j), w in edge_dict.items():
        adj[i].append(j)
        wts[i].append(w)
        adj[j].append(i)
        wts[j].append(w)

    xadj = [0]
    adjncy = []
    adjcwgt = []

    for i in range(n):
        neighbors_i = adj[i]
        weights_i = wts[i]
        adjncy.extend(neighbors_i)
        adjcwgt.extend(weights_i)
        xadj.append(len(adjncy))

    vwgt = [1] * n
    return xadj, adjncy, adjcwgt, vwgt


# ---------- Step 4: run KaHIP ----------
def run_kahip(xadj, adjncy, adjcwgt, vwgt, m: int, imbalance: float, mode: int, seed: int):
    suppress_output = 1  # 1 = no KaHIP console spam
    edgecut, blocks = kahip.kaffpa(
        vwgt,        # list of node weights or None
        xadj,        # CSR row offsets
        adjcwgt,     # edge weights
        adjncy,      # CSR column indices
        m,           # number of blocks
        imbalance,   # imbalance, e.g. 0.03
        suppress_output,
        seed,
        mode         # 0=FAST,1=ECO,2=STRONG,...
    )
    return np.array(blocks, dtype=np.int32)



# ---------- Step 5: train MLP ----------
def train_mlp(X: np.ndarray, labels: np.ndarray,
              m: int, layers: int, nodes: int,
              epochs: int, batch_size: int, lr: float, seed: int):

    torch.manual_seed(seed) #equivelant of srand()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #will run computations on gpu

    n, d = X.shape
    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(labels.astype(np.int64))
    dataset = TensorDataset(X_tensor, y_tensor) #Breaks dataset into x,y tensors
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Splits the dataset into mini-batches

    model = MLPClassifier(d_in=d, m_out=m, hidden_nodes=nodes, num_layers=layers).to(device) #Creates MLP
    opt = torch.optim.Adam(model.parameters(), lr=lr) # Adam maintains adaptive learning rates for params
    loss_fn = nn.CrossEntropyLoss()

    model.train() #As of now, does not change behaviour
    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / n
        print(f"Epoch {epoch+1}/{epochs} - loss = {avg_loss:.4f}")

    return model


# ---------- Step 6: build inverted index ----------
def build_inverted_lists(labels: np.ndarray, m: int):
    inv_lists = {r: [] for r in range(m)}
    for i, r in enumerate(labels):
        inv_lists[int(r)].append(int(i))
    return inv_lists

# ---------- Step 0: "Main" Function of Build Program ----------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=== Neural LSH build phase started ===")
    print(f"Seed           : {args.seed}")
    print(f"Dataset path   : {args.d}")
    print(f"Index prefix   : {args.i}")
    print(f"Type           : {args.type}")
    print(f"kNN            : {args.knn}")
    print(f"m (partitions) : {args.m}")
    print(f"KaHIP mode     : {args.kahip_mode} (0=FAST,1=ECO,2=STRONG)")
    print(f"Imbalance      : {args.imbalance}")
    print(f"MLP layers     : {args.layers}")
    print(f"MLP nodes      : {args.nodes}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"LR             : {args.lr}")
    print("-" * 50)

    print(f"Loading dataset from {args.d} (type={args.type})")
    dataset = load_dataset(args.d, args.type)  # Done in data_reader.py

    dataset = dataset[:5000]
    n, d = dataset.shape
    print(f"Loaded {n} points of dimension {d}")

    print(f"Building k-NN graph with k={args.knn}")
    neighbors = build_knn_graph(dataset, k=args.knn)

    print("Building weighted edges")
    edge_dict = build_weighted_edges(neighbors)

    print("Converting to CSR format for KaHIP")
    xadj, adjncy, adjcwgt, vwgt = edges_to_csr(n, edge_dict)

    print(f"Running KaHIP with m={args.m}, imbalance={args.imbalance}, mode={args.kahip_mode}")
    labels = run_kahip(xadj, adjncy, adjcwgt, vwgt,
                       m=args.m, imbalance=args.imbalance,
                       mode=args.kahip_mode, seed=args.seed)
    print("KaHIP done.")

    print("Training MLP classifier")
    model = train_mlp(dataset, labels,
                      m=args.m,
                      layers=args.layers,
                      nodes=args.nodes,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      lr=args.lr,
                      seed=args.seed)

    print("Building inverted lists")
    inv_lists = build_inverted_lists(labels, m=args.m)

    # saving
    os.makedirs(os.path.dirname(args.i), exist_ok=True) if os.path.dirname(args.i) else None
    model_path = args.i + "_model.pth"
    index_path = args.i + "_index.pkl"

    print(f"Saving model to {model_path}")
    torch.save(model, model_path)

    print(f"Saving index to {index_path}")
    with open(index_path, "wb") as f:
        pickle.dump(
            {
                "points": dataset.astype(np.float32),
                "labels": labels,
                "inv_lists": inv_lists,
                "m": args.m,
            },
            f,
        )

    print("nlsh_build finished.")


if __name__ == "__main__":
    main()

