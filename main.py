import pandas as pd
import numpy as np

# -----------------------------
# LOAD DATA
# -----------------------------
node_data  = pd.read_csv("node_data.csv")
graph_data = pd.read_csv("graph.csv")
params     = pd.read_csv("parameters.csv")
cost_data  = pd.read_csv("defense_cost.csv")

alpha = params["alpha"][0]
beta  = params["beta"][0]
gamma = params["gamma"][0]

results_summary  = []
validation_rows  = []

# -----------------------------
# USER INPUT FOR BUDGET (ONCE)
# -----------------------------
try:
    budget = int(input("Enter total budget for defense selection: "))
    if budget <= 0:
        print("Invalid budget! Using default = 10")
        budget = 10
except:
    print("Invalid input! Using default = 10")
    budget = 10

# -----------------------------
# BRUTE FORCE KNAPSACK (O(2^n))
# Only feasible for n <= 20
# -----------------------------
def brute_force_knapsack(values, costs, budget):
    n = len(values)
    best_val, best_sel = 0.0, []
    for mask in range(1 << n):
        sel = [i for i in range(n) if mask & (1 << i)]
        if sum(costs[i] for i in sel) <= budget:
            val = sum(values[i] for i in sel)
            if val > best_val:
                best_val, best_sel = val, sel
    return best_val, best_sel

# -----------------------------
# LOOP OVER AVAILABLE DATASETS
# -----------------------------
for d in node_data["Dataset"].unique():

    df = node_data[node_data["Dataset"] == d].copy()

    if df.empty:
        print(f"Dataset {d} is empty, skipping...")
        continue

    nodes = df["Node"].tolist()
    V  = df["V"].values
    pp = df["pp"].values
    fp = df["fp"].values
    n  = len(nodes)

    node_index = {node: i for i, node in enumerate(nodes)}

    # ---------------- GRAPH ----------------
    gdf   = graph_data[graph_data["Dataset"] == d]
    graph = {i: [] for i in range(n)}

    for _, row in gdf.iterrows():
        if row["From"] in node_index and row["To"] in node_index:
            i = node_index[row["From"]]
            j = node_index[row["To"]]
            graph[i].append(j)

    # ---------------- T ----------------
    T = beta * pp + (1 - beta) * V
    T = T / (np.sum(T) + 1e-9)

    # ---------------- Z ----------------
    Z = -np.log(fp + 1e-9)
    Z = Z / (np.sum(Z) + 1e-9)

    # ---------------- TRANSITION MATRIX ----------------
    W = np.zeros((n, n))

    for i in range(n):
        if len(graph[i]) > 0:
            p = 1 / len(graph[i])
            for j in graph[i]:
                W[i][j] = p
        else:
            W[i] = np.ones(n) / n

    # ---------------- INITIAL R ----------------
    R = np.ones(n) / n

    # ---------------- ITERATION ----------------
    for _ in range(100):
        WR    = np.dot(W.T, R)
        WR    = WR / (np.sum(WR) + 1e-9)
        S     = gamma * WR + (1 - gamma) * Z
        R_new = alpha * T + (1 - alpha) * S
        R_new = R_new / (np.sum(R_new) + 1e-9)

        if np.linalg.norm(R_new - R) < 1e-6:
            break

        R = R_new

    # ---------------- COST ----------------
    cost_df_d = cost_data[cost_data["Dataset"] == d]
    cost_map  = dict(zip(cost_df_d["Node"], cost_df_d["Cost"]))
    costs     = np.array([cost_map.get(node, 2) for node in nodes])

    # ---------------- VALUE FUNCTION ----------------
    values    = R * V
    value_map = {nodes[i]: values[i] for i in range(n)}

    # ---------------- DP KNAPSACK ----------------
    dp = np.zeros((n + 1, budget + 1))

    for i in range(1, n + 1):
        for w in range(budget + 1):
            if costs[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w],
                    values[i - 1] + dp[i - 1][w - int(costs[i - 1])]
                )
            else:
                dp[i][w] = dp[i - 1][w]

    # ---------------- TRACEBACK ----------------
    selected = []
    w = budget

    for i in range(n, 0, -1):
        if dp[i][w] > dp[i - 1][w]:
            selected.append(nodes[i - 1])
            w -= int(costs[i - 1])

    selected.reverse()

    # ---------------- OUTPUT ----------------
    print("\n==============================")
    print(f"         DATASET {d}")
    print("==============================")

    print("\nFINAL RANK:")
    for i in range(n):
        print(f"{nodes[i]} : {round(R[i], 4)}")

    print("\nOPTIMAL DEFENSE SELECTION:")

    total_cost  = 0
    total_value = 0

    for node in selected:
        c = cost_map.get(node, 2)
        v = value_map[node]
        total_cost  += c
        total_value += v
        print(f"{node} -> Cost: {c}, Value: {round(v, 4)}")

    print("\nSUMMARY:")
    print("Total Cost:", total_cost)
    print("Total Security Gain:", round(total_value, 4))

    top_node = nodes[np.argmax(R)] if len(R) > 0 else "None"

    results_summary.append([
        d,
        ",".join(selected),
        total_cost,
        round(total_value, 4),
        top_node
    ])
    # ---------------- BRUTE FORCE (skip if n > 20) ----------------
    dp_val = round(dp[n][budget], 4)

    if n <= 20:
        bf_val, bf_idx = brute_force_knapsack(values, costs, budget)
        bf_val    = round(bf_val, 4)
        bf_nodes  = sorted([nodes[i] for i in bf_idx])
        match     = abs(dp_val - bf_val) < 1e-9
        accuracy  = 100.0 if match else round(dp_val / bf_val * 100, 2)
        bf_label  = bf_val
    else:
        bf_val, bf_nodes, match, accuracy, bf_label = None, [], None, None, "N/A (n>20)"

    validation_rows.append({
        "Dataset":          d,
        "Nodes (n)":        n,
        "Budget":           budget,
        "DP Value":         dp_val,
        "BF Value":         bf_label,
        "Match":            "YES" if match else ("NO" if match is not None else "SKIPPED"),
        "Accuracy (%)":     f"{accuracy:.1f}%" if accuracy is not None else "N/A",
        "DP Selected":      ", ".join(selected),
        "BF Selected":      ", ".join(bf_nodes) if bf_nodes else "N/A",
    })

# -----------------------------
# SAVE RESULTS
# -----------------------------
pd.DataFrame(
    results_summary,
    columns=["Dataset", "Selected Nodes", "Total Cost", "Total Gain", "Top Node"]
).to_csv("final_results.csv", index=False)

pd.DataFrame(validation_rows).to_csv("validation_results.csv", index=False)

# -----------------------------
# VALIDATION TABLE
# -----------------------------
print("\n")
print("=" * 78)
print("  VALIDATION — DP KNAPSACK vs BRUTE FORCE")
print(f"  Budget: {budget}")
print("=" * 78)
print(f"  {'DS':<4} {'n':>3} {'DP Value':>10} {'BF Value':>10} {'Match':>7} {'Accuracy':>9}")
print("-" * 78)

for row in validation_rows:
    print(f"  {row['Dataset']:<4} {row['Nodes (n)']:>3} {str(row['DP Value']):>10} "
          f"{str(row['BF Value']):>10} {row['Match']:>7} {row['Accuracy (%)']:>9}")

print("=" * 78)

matched = sum(1 for r in validation_rows if r["Match"] == "YES")
checked = sum(1 for r in validation_rows if r["Match"] != "SKIPPED")
skipped = sum(1 for r in validation_rows if r["Match"] == "SKIPPED")

print(f"\n  Perfect matches : {matched}/{checked} datasets")
if skipped:
    print(f"  Skipped         : {skipped} dataset(s) with n > 20 (brute force infeasible)")
print(f"  Overall accuracy: 100.00%" if matched == checked else "")
print("\nSaved: final_results.csv  |  validation_results.csv")
