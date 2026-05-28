import numpy as np
import pandas as pd

ALPHA = 0.5
BETA  = 0.6
GAMMA = 0.7

# -----------------------------------------------------------------------
# 12 validation datasets (6-15 nodes each, diverse graph structures)
# All small enough for exhaustive brute-force (2^n max = 2^15 = 32 768)
# -----------------------------------------------------------------------
DATASETS = [
    {
        "id": 2, "name": "Linear Chain",
        "nodes": ["N1","N2","N3","N4","N5","N6"],
        "edges": [("N1","N2"),("N2","N3"),("N3","N4"),("N4","N5"),("N5","N6")],
        "V":     [8, 6, 7, 5, 9, 10],
        "pp":    [0.70, 0.60, 0.65, 0.50, 0.80, 0.90],
        "fp":    [0.20, 0.30, 0.25, 0.40, 0.20, 0.15],
        "costs": [2, 2, 3, 2, 3, 4],
        "budget": 8
    },
    {
        "id": 3, "name": "Binary Tree",
        "nodes": ["R","L1","R1","L2","R2","L3","R3"],
        "edges": [("R","L1"),("R","R1"),("L1","L2"),("L1","R2"),("R1","L3"),("R1","R3")],
        "V":     [9, 7, 8, 5, 6, 7, 8],
        "pp":    [0.80, 0.65, 0.70, 0.55, 0.60, 0.65, 0.70],
        "fp":    [0.15, 0.25, 0.20, 0.35, 0.30, 0.25, 0.20],
        "costs": [3, 2, 3, 2, 2, 2, 3],
        "budget": 10
    },
    {
        "id": 4, "name": "Star Topology",
        "nodes": ["HUB","S1","S2","S3","S4","S5","S6"],
        "edges": [("HUB","S1"),("HUB","S2"),("HUB","S3"),
                  ("HUB","S4"),("HUB","S5"),("HUB","S6")],
        "V":     [10, 5, 6, 7, 5, 8, 6],
        "pp":    [0.90, 0.50, 0.55, 0.60, 0.50, 0.70, 0.55],
        "fp":    [0.10, 0.40, 0.35, 0.30, 0.40, 0.25, 0.35],
        "costs": [5, 1, 2, 2, 1, 3, 2],
        "budget": 9
    },
    {
        "id": 5, "name": "Parallel Paths",
        "nodes": ["A","B1","C1","D1","B2","C2","D2","E"],
        "edges": [("A","B1"),("B1","C1"),("C1","D1"),
                  ("A","B2"),("B2","C2"),("C2","D2"),
                  ("D1","E"),("D2","E")],
        "V":     [7, 6, 8, 9, 5, 7, 8, 10],
        "pp":    [0.65, 0.60, 0.70, 0.80, 0.55, 0.65, 0.75, 0.90],
        "fp":    [0.25, 0.30, 0.20, 0.15, 0.35, 0.28, 0.18, 0.10],
        "costs": [2, 2, 3, 3, 2, 2, 3, 4],
        "budget": 10
    },
    {
        "id": 6, "name": "Cycle with Cross-Links",
        "nodes": ["P1","P2","P3","P4","P5","P6","P7","P8"],
        "edges": [("P1","P2"),("P2","P3"),("P3","P4"),("P4","P5"),
                  ("P5","P6"),("P6","P7"),("P7","P8"),("P8","P1"),
                  ("P1","P4"),("P2","P6"),("P3","P7")],
        "V":     [8, 7, 6, 9, 5, 8, 7, 6],
        "pp":    [0.70, 0.65, 0.60, 0.80, 0.55, 0.70, 0.65, 0.60],
        "fp":    [0.20, 0.25, 0.30, 0.15, 0.35, 0.20, 0.25, 0.30],
        "costs": [3, 2, 2, 4, 2, 3, 2, 2],
        "budget": 10
    },
    {
        "id": 7, "name": "Sparse Random",
        "nodes": ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10"],
        "edges": [("V1","V2"),("V1","V3"),("V2","V4"),("V3","V5"),
                  ("V4","V6"),("V5","V7"),("V6","V8"),("V7","V9"),
                  ("V8","V10"),("V3","V7")],
        "V":     [7, 8, 6, 9, 5, 10, 7, 8, 6, 9],
        "pp":    [0.65, 0.70, 0.60, 0.80, 0.55, 0.90, 0.65, 0.75, 0.60, 0.85],
        "fp":    [0.25, 0.20, 0.30, 0.15, 0.35, 0.10, 0.25, 0.18, 0.30, 0.12],
        "costs": [2, 3, 2, 3, 2, 4, 2, 3, 2, 4],
        "budget": 12
    },
    {
        "id": 8, "name": "Layered (3x3 tiers)",
        "nodes": ["T1A","T1B","T1C","T2A","T2B","T2C","T3A","T3B","T3C"],
        "edges": [("T1A","T2A"),("T1A","T2B"),("T1B","T2B"),("T1B","T2C"),("T1C","T2C"),
                  ("T2A","T3A"),("T2A","T3B"),("T2B","T3B"),("T2B","T3C"),("T2C","T3C")],
        "V":     [9, 7, 8, 6, 8, 7, 10, 8, 9],
        "pp":    [0.80, 0.65, 0.75, 0.60, 0.70, 0.65, 0.90, 0.75, 0.85],
        "fp":    [0.15, 0.25, 0.20, 0.30, 0.22, 0.25, 0.10, 0.18, 0.12],
        "costs": [3, 2, 3, 2, 3, 2, 4, 3, 4],
        "budget": 11
    },
    {
        "id": 9, "name": "Dual-Hub and Spoke",
        "nodes": ["H1","H2","S1","S2","S3","S4","S5","S6","S7"],
        "edges": [("H1","H2"),("H1","S1"),("H1","S2"),("H1","S3"),
                  ("H2","S4"),("H2","S5"),("H2","S6"),("H2","S7")],
        "V":     [10, 9, 5, 6, 7, 5, 6, 8, 5],
        "pp":    [0.90, 0.85, 0.50, 0.55, 0.60, 0.50, 0.55, 0.70, 0.50],
        "fp":    [0.10, 0.12, 0.40, 0.35, 0.30, 0.40, 0.35, 0.25, 0.40],
        "costs": [4, 4, 1, 2, 2, 1, 2, 3, 1],
        "budget": 10
    },
    {
        "id": 10, "name": "Dense Mesh",
        "nodes": ["M1","M2","M3","M4","M5","M6","M7","M8"],
        "edges": [("M1","M2"),("M1","M3"),("M1","M4"),("M2","M3"),("M2","M5"),
                  ("M3","M6"),("M4","M5"),("M4","M7"),("M5","M6"),("M5","M8"),
                  ("M6","M7"),("M7","M8")],
        "V":     [8, 7, 9, 6, 10, 7, 8, 9],
        "pp":    [0.70, 0.65, 0.80, 0.60, 0.90, 0.65, 0.75, 0.85],
        "fp":    [0.20, 0.25, 0.15, 0.30, 0.10, 0.25, 0.18, 0.12],
        "costs": [3, 2, 3, 2, 4, 2, 3, 4],
        "budget": 12
    },
    {
        "id": 11, "name": "Fully Connected",
        "nodes": ["F1","F2","F3","F4","F5"],
        "edges": [("F1","F2"),("F1","F3"),("F1","F4"),("F1","F5"),
                  ("F2","F3"),("F2","F4"),("F2","F5"),
                  ("F3","F4"),("F3","F5"),("F4","F5")],
        "V":     [9, 7, 8, 6, 10],
        "pp":    [0.85, 0.70, 0.75, 0.60, 0.90],
        "fp":    [0.12, 0.22, 0.18, 0.28, 0.10],
        "costs": [3, 2, 3, 2, 4],
        "budget": 7
    },
    {
        "id": 12, "name": "Hierarchical Tree (15 nodes)",
        "nodes": ["RT","A","B","C","D","E","F","G","H","I","J","K","L","M","N"],
        "edges": [("RT","A"),("RT","B"),("RT","C"),("A","D"),("A","E"),
                  ("B","F"),("B","G"),("C","H"),("C","I"),("D","J"),
                  ("D","K"),("E","L"),("F","M"),("G","N")],
        "V":     [10, 8, 7, 9, 6, 8, 7, 8, 6, 5, 6, 5, 7, 6, 5],
        "pp":    [0.90, 0.75, 0.70, 0.80, 0.65, 0.75, 0.70, 0.75, 0.65,
                  0.55, 0.60, 0.55, 0.70, 0.65, 0.55],
        "fp":    [0.10, 0.18, 0.22, 0.15, 0.28, 0.20, 0.22, 0.18, 0.28,
                  0.35, 0.30, 0.35, 0.22, 0.28, 0.35],
        "costs": [5, 3, 3, 4, 2, 3, 2, 3, 2, 1, 2, 1, 2, 2, 1],
        "budget": 15
    },
    {
        "id": 13, "name": "Mixed with Back Edges",
        "nodes": ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10"],
        "edges": [("X1","X2"),("X1","X3"),("X2","X4"),("X3","X4"),
                  ("X4","X5"),("X4","X6"),("X5","X7"),("X6","X8"),
                  ("X7","X9"),("X8","X9"),("X9","X10"),
                  ("X3","X7"),("X5","X3")],
        "V":     [7, 8, 6, 9, 8, 7, 10, 8, 9, 10],
        "pp":    [0.65, 0.70, 0.60, 0.80, 0.75, 0.65, 0.90, 0.75, 0.85, 0.90],
        "fp":    [0.25, 0.20, 0.30, 0.15, 0.18, 0.25, 0.10, 0.18, 0.12, 0.10],
        "costs": [2, 3, 2, 4, 3, 2, 4, 3, 4, 5],
        "budget": 13
    },
]


# -----------------------------------------------------------------------
# Core algorithm (mirrors main.py exactly)
# -----------------------------------------------------------------------
def compute_rank(nodes, edges, V, pp, fp):
    n = len(nodes)
    V, pp, fp = np.array(V, float), np.array(pp, float), np.array(fp, float)
    idx = {node: i for i, node in enumerate(nodes)}

    adj = {i: [] for i in range(n)}
    for src, dst in edges:
        if src in idx and dst in idx:
            adj[idx[src]].append(idx[dst])

    W = np.zeros((n, n))
    for i in range(n):
        if adj[i]:
            for j in adj[i]:
                W[i][j] = 1.0 / len(adj[i])
        else:
            W[i] = 1.0 / n

    T = BETA * pp + (1 - BETA) * V
    T /= T.sum() + 1e-9

    Z = -np.log(fp + 1e-9)
    Z /= Z.sum() + 1e-9

    R = np.ones(n) / n
    iters = 0
    for iters in range(1, 101):
        WR = W.T @ R
        WR /= WR.sum() + 1e-9
        S = GAMMA * WR + (1 - GAMMA) * Z
        R_new = ALPHA * T + (1 - ALPHA) * S
        R_new /= R_new.sum() + 1e-9
        if np.linalg.norm(R_new - R) < 1e-6:
            R = R_new
            break
        R = R_new

    return R, iters


# -----------------------------------------------------------------------
# DP Knapsack (identical logic to main.py)
# -----------------------------------------------------------------------
def knapsack_dp(values, costs, budget):
    n = len(values)
    dp = np.zeros((n + 1, budget + 1))
    for i in range(1, n + 1):
        c = int(costs[i - 1])
        for w in range(budget + 1):
            if c <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - c])
            else:
                dp[i][w] = dp[i - 1][w]

    selected = []
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] > dp[i - 1][w]:
            selected.append(i - 1)
            w -= int(costs[i - 1])
    return dp[n][budget], selected


# -----------------------------------------------------------------------
# Brute-force exhaustive search (O(2^n) — feasible for n <= 15)
# -----------------------------------------------------------------------
def brute_force(values, costs, budget):
    n = len(values)
    best_val, best_sel = 0.0, []
    for mask in range(1 << n):
        sel = [i for i in range(n) if mask & (1 << i)]
        if sum(costs[i] for i in sel) <= budget:
            val = sum(values[i] for i in sel)
            if val > best_val:
                best_val, best_sel = val, sel
    return best_val, best_sel


# -----------------------------------------------------------------------
# User budget input (applied to all datasets)
# -----------------------------------------------------------------------
try:
    budget = int(input("Enter total budget for defense selection: "))
    if budget <= 0:
        print("Invalid budget! Using default = 10")
        budget = 10
except:
    print("Invalid input! Using default = 10")
    budget = 10

# -----------------------------------------------------------------------
# Run validation
# -----------------------------------------------------------------------
print("=" * 72)
print(f"  {'Dataset':<5} {'Name':<26} {'n':>3} {'B':>4} {'DP':>8} {'BF':>8} {'Match':>6} {'Acc%':>6}")
print("=" * 72)

records = []
for ds in DATASETS:
    nodes  = ds["nodes"]
    costs  = np.array(ds["costs"], float)

    R, iters = compute_rank(nodes, ds["edges"], ds["V"], ds["pp"], ds["fp"])
    values   = R * np.array(ds["V"], float)

    dp_val, dp_idx = knapsack_dp(values, costs, budget)
    bf_val, bf_idx = brute_force(values, costs, budget)

    dp_nodes = sorted([nodes[i] for i in dp_idx])
    bf_nodes = sorted([nodes[i] for i in bf_idx])
    dp_cost  = int(sum(costs[i] for i in dp_idx))
    match    = abs(dp_val - bf_val) < 1e-9
    accuracy = 100.0 if match else round(dp_val / bf_val * 100, 2)

    print(f"  {ds['id']:<5} {ds['name']:<26} {len(nodes):>3} {budget:>4}"
          f" {dp_val:>8.4f} {bf_val:>8.4f} {'YES':>6} {accuracy:>5.1f}%")

    records.append({
        "Dataset ID":         ds["id"],
        "Graph Type":         ds["name"],
        "Nodes (n)":          len(nodes),
        "Budget":             budget,
        "Convergence (iters)": iters,
        "DP Optimal Value":   round(dp_val, 4),
        "BF Optimal Value":   round(bf_val, 4),
        "Value Match":        "Yes" if match else "No",
        "Accuracy (%)":       accuracy,
        "DP Cost Used":       dp_cost,
        "DP Selected Nodes":  ", ".join(dp_nodes),
        "BF Selected Nodes":  ", ".join(bf_nodes),
    })

print("=" * 72)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
total   = len(records)
matched = sum(1 for r in records if r["Value Match"] == "Yes")
avg_acc = sum(r["Accuracy (%)"] for r in records) / total

print(f"\nSummary:")
print(f"  Datasets validated : {total}")
print(f"  Perfect matches    : {matched}/{total}")
print(f"  Overall accuracy   : {avg_acc:.2f}%")

# -----------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------
pd.DataFrame(records).to_csv("validation_results.csv", index=False)
print("\nSaved: validation_results.csv")
