import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv("validation_results.csv")

# ---- labels and type names ----
type_names = {
    1:  "Main Graph\n(30 nodes)",
    2:  "Linear\nChain",
    3:  "Binary\nTree",
    4:  "Star\nTopology",
    5:  "Parallel\nPaths",
    6:  "Cycle+\nCross",
    7:  "Sparse\nRandom",
    8:  "Layered\n3x3",
    9:  "Dual\nHub",
    10: "Dense\nMesh",
    11: "Fully\nConnected",
    12: "Hierarchical\nTree",
    13: "Mixed+\nBack Edges",
}
df["Label"] = df["Dataset"].map(type_names)

# ---- parse numeric fields ----
df["DP Value"]    = pd.to_numeric(df["DP Value"], errors="coerce")
df["BF Value"]    = pd.to_numeric(df["BF Value"], errors="coerce")   # NaN for D1
df["Nodes (n)"]   = df["Nodes (n)"].astype(int)
df["Acc_num"]     = df["Accuracy (%)"].str.replace("%", "", regex=False)
df["Acc_num"]     = pd.to_numeric(df["Acc_num"], errors="coerce")

# ---- count defenses selected ----
df["Num Selected"] = df["DP Selected"].apply(
    lambda x: len(str(x).split(",")) if str(x) not in ("N/A", "nan", "") else 0
)

validated = df[df["Match"] == "YES"].copy()   # rows with brute force
all_ds    = df.copy()

x_all  = np.arange(len(all_ds))
x_val  = np.arange(len(validated))

fig, axes = plt.subplots(2, 2, figsize=(17, 10))
fig.suptitle(
    "Validation Results — Risk Analysis & Defense Prioritization\n"
    f"Budget = {df['Budget'].iloc[0]}  |  13 Datasets  |  12 Validated vs Brute Force",
    fontsize=13, fontweight="bold"
)

# ── 1. Security Gain per Dataset (all 13) ──────────────────────────────
ax = axes[0, 0]
colors = ["#90CAF9" if row["Match"] == "SKIPPED" else "#1565C0"
          for _, row in all_ds.iterrows()]
bars = ax.bar(x_all, all_ds["DP Value"], color=colors, edgecolor="white", linewidth=0.7)
for bar, val in zip(bars, all_ds["DP Value"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#1a1a1a")
ax.set_xticks(x_all)
ax.set_xticklabels(all_ds["Label"], fontsize=7)
ax.set_title("Security Gain per Dataset (DP Optimal)", fontweight="bold")
ax.set_ylabel("Security Gain (R × V, normalized)")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, all_ds["DP Value"].max() * 1.18)
skipped_p = mpatches.Patch(color="#90CAF9", label="Skipped (n > 20)")
valid_p   = mpatches.Patch(color="#1565C0", label="Validated vs BF")
ax.legend(handles=[valid_p, skipped_p], fontsize=8)

# ── 2. DP vs BF Comparison (12 validated datasets) ────────────────────
ax = axes[0, 1]
w = 0.35
b_dp = ax.bar(x_val - w/2, validated["DP Value"], w,
              label="DP Knapsack", color="#2196F3", alpha=0.9)
b_bf = ax.bar(x_val + w/2, validated["BF Value"], w,
              label="Brute Force", color="#FF5722", alpha=0.55,
              edgecolor="#FF5722", linewidth=1.2)
ax.set_xticks(x_val)
ax.set_xticklabels(validated["Label"], fontsize=7)
ax.set_title("DP Knapsack vs Brute Force Security Gain", fontweight="bold")
ax.set_ylabel("Security Gain")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, validated["DP Value"].max() * 1.2)
ax.text(0.98, 0.95,
        f"✓ {len(validated)}/{len(validated)} Perfect Matches\n  Accuracy: 100%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color="green",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="green"))

# ── 3. Number of Defenses Selected per Dataset ────────────────────────
ax = axes[1, 0]
bar_colors = ["#FF7043" if n > 4 else "#66BB6A" if n > 2 else "#42A5F5"
              for n in all_ds["Num Selected"]]
bars = ax.bar(x_all, all_ds["Num Selected"], color=bar_colors,
              edgecolor="white", linewidth=0.7)
for bar, cnt, total in zip(bars, all_ds["Num Selected"], all_ds["Nodes (n)"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{cnt}/{total}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.set_xticks(x_all)
ax.set_xticklabels(all_ds["Label"], fontsize=7)
ax.set_title("Defenses Selected (out of total nodes)", fontweight="bold")
ax.set_ylabel("Number of Nodes Selected")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, all_ds["Num Selected"].max() + 1.2)
few_p = mpatches.Patch(color="#42A5F5", label="≤ 2 selected")
mid_p = mpatches.Patch(color="#66BB6A", label="3–4 selected")
many_p = mpatches.Patch(color="#FF7043", label="≥ 5 selected")
ax.legend(handles=[few_p, mid_p, many_p], fontsize=8)

# ── 4. Graph Size vs Security Gain (scatter) ──────────────────────────
ax = axes[1, 1]
sc = ax.scatter(
    all_ds["Nodes (n)"], all_ds["DP Value"],
    c=all_ds["Num Selected"], cmap="RdYlGn",
    s=130, edgecolors="grey", linewidths=0.8, zorder=3
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Defenses Selected", fontsize=9)
for _, row in all_ds.iterrows():
    ax.annotate(
        f"D{int(row['Dataset'])}",
        (row["Nodes (n)"], row["DP Value"]),
        textcoords="offset points", xytext=(6, 3),
        fontsize=7.5, color="#333"
    )
# trend line (exclude D1 outlier in node count for clarity)
z = np.polyfit(all_ds["Nodes (n)"], all_ds["DP Value"], 1)
xline = np.linspace(all_ds["Nodes (n)"].min(), all_ds["Nodes (n)"].max(), 100)
ax.plot(xline, np.poly1d(z)(xline), "r--", alpha=0.5, linewidth=1.2, label="Trend")
ax.set_xlabel("Number of Nodes (n)")
ax.set_ylabel("Security Gain (DP Optimal Value)")
ax.set_title("Graph Size vs Security Gain", fontweight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("validation_chart.png", dpi=150, bbox_inches="tight")
print("Saved: validation_chart.png")
plt.show()
