"""
summarise_sweep.py
------------------
Reads frame_level/results/sweep/sweep_summary.json and produces
a comprehensive Markdown report at frame_level/results/sweep/sweep_report.md.

Total loss during training:  BCE + β·KLD + γ·CE
  β  = KL-divergence weight (disentanglement pressure)
  γ  = classification loss weight (supervised signal)

Usage:
    python summarise_sweep.py [--json PATH] [--out PATH]
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def fmt(v, digits=4):
    return f"{v:.{digits}f}"

def bold_if(val, cond):
    s = fmt(val)
    return f"**{s}**" if cond else s

def pct(v):
    return f"{v*100:.1f}%"


def rank_mark(rank, n):
    """Return a medal emoji for the top 3."""
    if rank == 0: return " 🥇"
    if rank == 1: return " 🥈"
    if rank == 2: return " 🥉"
    return ""


# ── load & validate ───────────────────────────────────────────────────────────

def load_data(json_path: str):
    with open(json_path) as f:
        data = json.load(f)

    # Filter to successful runs only
    runs = [d for d in data if d.get("status") == "success"]
    failed = [d for d in data if d.get("status") != "success"]

    # Attach a short run_id for tables
    for r in runs:
        c = r["config"]
        r["run_id"] = (
            f"ld{c['latent_dim']}_β{c['beta']}_γ{c['gamma']}_lr{c['learning_rate']}"
        )

    return runs, failed


def find_missing(runs):
    lats  = [10, 20, 30]
    betas = [2.0, 5.0]
    gammas = [1.0, 2.0, 5.0, 10.0]
    lrs   = [1e-5, 1e-4, 1e-3]
    existing = {
        (r["config"]["latent_dim"], r["config"]["beta"],
         r["config"]["gamma"],      r["config"]["learning_rate"])
        for r in runs
    }
    missing = [
        combo for combo in product(lats, betas, gammas, lrs)
        if combo not in existing
    ]
    return missing


# ── aggregation helpers ───────────────────────────────────────────────────────

def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def group_by(runs, key_fn):
    groups = defaultdict(list)
    for r in runs:
        groups[key_fn(r)].append(r)
    return dict(groups)


def metric_table_for_groups(groups: dict, metric: str, higher_is_better: bool):
    """Return list of (group_label, mean_val, n) sorted best→worst."""
    rows = []
    for label, grp in groups.items():
        vals = [r["metrics"][metric] for r in grp if metric in r["metrics"]]
        if vals:
            rows.append((label, mean(vals), len(vals)))
    rows.sort(key=lambda x: x[1], reverse=higher_is_better)
    return rows


# ── markdown builders ─────────────────────────────────────────────────────────

def md_overview_table(runs):
    """Big table: one row per run, sorted by best_val_accuracy desc."""
    headers = [
        "Run", "ld", "β", "γ", "lr",
        "Val Acc ↑", "Hungarian Acc ↑", "ARI ↑",
        "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-H ↑",
        "Sc0 ↑", "Sc1 ↑", "Sc2 ↑", "Sc3 ↑",
    ]
    runs_sorted = sorted(runs, key=lambda r: r["metrics"].get("best_val_accuracy", 0), reverse=True)

    best_va  = max(r["metrics"].get("best_val_accuracy", 0) for r in runs)
    best_ha  = max(r["metrics"].get("hungarian_accuracy",  0) for r in runs)
    best_ari = max(r["metrics"].get("adjusted_rand_index", 0) for r in runs)
    best_sil = max(r["metrics"].get("silhouette",          0) for r in runs)
    best_db  = min(r["metrics"].get("davies_bouldin",      9) for r in runs)
    best_ch  = max(r["metrics"].get("calinski_harabasz",   0) for r in runs)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in runs_sorted:
        m  = r["metrics"]
        c  = r["config"]
        va  = m.get("best_val_accuracy", float("nan"))
        ha  = m.get("hungarian_accuracy",  float("nan"))
        ari = m.get("adjusted_rand_index", float("nan"))
        sil = m.get("silhouette",          float("nan"))
        db  = m.get("davies_bouldin",      float("nan"))
        ch  = m.get("calinski_harabasz",   float("nan"))

        row = [
            r["run_id"],
            str(c["latent_dim"]),
            str(c["beta"]),
            str(c["gamma"]),
            str(c["learning_rate"]),
            bold_if(va,  va  == best_va),
            bold_if(ha,  ha  == best_ha),
            bold_if(ari, ari == best_ari),
            bold_if(sil, sil == best_sil),
            bold_if(db,  db  == best_db),
            bold_if(ch,  ch  == best_ch),
            fmt(m.get("score_0_acc", float("nan"))),
            fmt(m.get("score_1_acc", float("nan"))),
            fmt(m.get("score_2_acc", float("nan"))),
            fmt(m.get("score_3_acc", float("nan"))),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def md_top_runs_section(runs, n=10):
    """Top-N runs by val_acc and hungarian_acc separately."""
    def top_table(metric, higher):
        sorted_runs = sorted(
            runs, key=lambda r: r["metrics"].get(metric, -1e9 if higher else 1e9),
            reverse=higher
        )[:n]
        headers = ["Rank", "Run", "Val Acc", "Hungarian Acc", "ARI", "Silhouette"]
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for i, r in enumerate(sorted_runs):
            m = r["metrics"]
            medal = rank_mark(i, n)
            lines.append(
                f"| {i+1}{medal} | {r['run_id']} | "
                f"{fmt(m.get('best_val_accuracy',0))} | "
                f"{fmt(m.get('hungarian_accuracy',0))} | "
                f"{fmt(m.get('adjusted_rand_index',0))} | "
                f"{fmt(m.get('silhouette',0))} |"
            )
        return "\n".join(lines)

    return (
        "### Top 10 by Validation Accuracy\n\n"
        + top_table("best_val_accuracy", True)
        + "\n\n### Top 10 by Hungarian (Clustering) Accuracy\n\n"
        + top_table("hungarian_accuracy", True)
        + "\n\n### Top 10 by Adjusted Rand Index\n\n"
        + top_table("adjusted_rand_index", True)
    )


def md_param_effect_table(runs, param_key, param_label):
    """Table: effect of one hyperparameter averaged over all others."""
    metrics_cfg = [
        ("best_val_accuracy",  "Val Acc",   True),
        ("hungarian_accuracy", "Hung Acc",  True),
        ("adjusted_rand_index","ARI",        True),
        ("silhouette",         "Silhouette", True),
        ("davies_bouldin",     "DB",         False),
        ("calinski_harabasz",  "CH",         True),
    ]
    groups = group_by(runs, lambda r: r["config"][param_key])
    sorted_keys = sorted(groups.keys())

    header_row  = [param_label] + [m[1] for m in metrics_cfg]
    lines = []
    lines.append("| " + " | ".join(header_row) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_row)) + " |")

    # Compute per-metric best across groups
    best = {}
    for mkey, mlabel, higher in metrics_cfg:
        vals_per_group = {k: mean([r["metrics"].get(mkey, float("nan")) for r in v])
                          for k, v in groups.items()}
        best[mkey] = max(vals_per_group.values()) if higher else min(vals_per_group.values())

    for k in sorted_keys:
        grp = groups[k]
        cells = [str(k)]
        for mkey, mlabel, higher in metrics_cfg:
            v = mean([r["metrics"].get(mkey, float("nan")) for r in grp])
            cells.append(bold_if(v, abs(v - best[mkey]) < 1e-9))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def md_heatmap_text(runs, row_key, col_key, metric, higher_is_better,
                    row_label, col_label, metric_label):
    """ASCII-style 2-D table: rows=row_key, cols=col_key, cells=mean(metric)."""
    row_vals = sorted(set(r["config"][row_key] for r in runs))
    col_vals = sorted(set(r["config"][col_key] for r in runs))

    # Build lookup: (row_val, col_val) -> mean metric
    table = {}
    for rv in row_vals:
        for cv in col_vals:
            subset = [r for r in runs
                      if r["config"][row_key] == rv and r["config"][col_key] == cv]
            if subset:
                table[(rv, cv)] = mean([r["metrics"].get(metric, float("nan")) for r in subset])

    all_vals = [v for v in table.values() if not (v != v)]  # remove nan
    best_val = max(all_vals) if higher_is_better else min(all_vals)

    header = [f"{row_label} \\ {col_label}"] + [str(c) for c in col_vals]
    lines  = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for rv in row_vals:
        row = [str(rv)]
        for cv in col_vals:
            v = table.get((rv, cv), float("nan"))
            if v != v:
                row.append("—")
            else:
                row.append(bold_if(v, abs(v - best_val) < 1e-9))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def md_per_class_accuracy(runs):
    """Table: average per-class accuracy by gamma (best indicator)."""
    groups = group_by(runs, lambda r: r["config"]["gamma"])
    sorted_keys = sorted(groups.keys())

    headers = ["γ", "Score 0 ↑", "Score 1 ↑", "Score 2 ↑", "Score 3 ↑", "Macro Avg ↑"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for k in sorted_keys:
        grp = groups[k]
        s0 = mean([r["metrics"].get("score_0_acc", 0) for r in grp])
        s1 = mean([r["metrics"].get("score_1_acc", 0) for r in grp])
        s2 = mean([r["metrics"].get("score_2_acc", 0) for r in grp])
        s3 = mean([r["metrics"].get("score_3_acc", 0) for r in grp])
        macro = mean([s0, s1, s2, s3])
        lines.append(
            f"| {k} | {fmt(s0)} | {fmt(s1)} | {fmt(s2)} | {fmt(s3)} | {fmt(macro)} |"
        )
    return "\n".join(lines)


def md_per_class_accuracy_by_ld(runs):
    """Table: average per-class accuracy by latent_dim."""
    groups = group_by(runs, lambda r: r["config"]["latent_dim"])
    sorted_keys = sorted(groups.keys())

    headers = ["ld", "Score 0 ↑", "Score 1 ↑", "Score 2 ↑", "Score 3 ↑", "Macro Avg ↑"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for k in sorted_keys:
        grp = groups[k]
        s0 = mean([r["metrics"].get("score_0_acc", 0) for r in grp])
        s1 = mean([r["metrics"].get("score_1_acc", 0) for r in grp])
        s2 = mean([r["metrics"].get("score_2_acc", 0) for r in grp])
        s3 = mean([r["metrics"].get("score_3_acc", 0) for r in grp])
        macro = mean([s0, s1, s2, s3])
        lines.append(
            f"| {k} | {fmt(s0)} | {fmt(s1)} | {fmt(s2)} | {fmt(s3)} | {fmt(macro)} |"
        )
    return "\n".join(lines)


# ── top-5 per-metric card ─────────────────────────────────────────────────────

def md_best_card(runs):
    def best(metric, higher, label):
        s = sorted(runs, key=lambda r: r["metrics"].get(metric, -1e9 if higher else 1e9),
                   reverse=higher)
        r = s[0]
        v = r["metrics"][metric]
        return f"- **{label}**: `{r['run_id']}` → {fmt(v)}"
    return "\n".join([
        best("best_val_accuracy",  True,  "Best Val Accuracy"),
        best("hungarian_accuracy", True,  "Best Hungarian Accuracy"),
        best("adjusted_rand_index",True,  "Best ARI"),
        best("silhouette",         True,  "Best Silhouette"),
        best("davies_bouldin",     False, "Best Davies-Bouldin"),
        best("calinski_harabasz",  True,  "Best Calinski-Harabasz"),
    ])


# ── correlation / tension analysis ───────────────────────────────────────────

def md_tension_table(runs):
    """
    Show the tension between val_accuracy and ARI.
    For each run compute val_acc - mean_val_acc and ARI - mean_ARI.
    Top runs by (normalised_val_acc + normalised_ARI) / 2.
    """
    all_va  = [r["metrics"].get("best_val_accuracy", 0)  for r in runs]
    all_ari = [r["metrics"].get("adjusted_rand_index", 0) for r in runs]
    mean_va, std_va   = mean(all_va),  (sum((v - mean(all_va))**2 for v in all_va)/len(all_va))**0.5
    mean_ar, std_ar   = mean(all_ari), (sum((v - mean(all_ari))**2 for v in all_ari)/len(all_ari))**0.5

    def z(v, mu, sigma):
        return (v - mu) / sigma if sigma > 1e-9 else 0.0

    scored = []
    for r in runs:
        va  = r["metrics"].get("best_val_accuracy",  0)
        ari = r["metrics"].get("adjusted_rand_index", 0)
        ha  = r["metrics"].get("hungarian_accuracy",  0)
        score = (z(va, mean_va, std_va) + z(ari, mean_ar, std_ar)) / 2
        scored.append((score, r, va, ari, ha))
    scored.sort(reverse=True)

    headers = ["Rank", "Run", "Val Acc", "ARI", "Hung Acc", "Combined Z"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for i, (sc, r, va, ari, ha) in enumerate(scored[:10]):
        medal = rank_mark(i, 10)
        lines.append(
            f"| {i+1}{medal} | {r['run_id']} | "
            f"{fmt(va)} | {fmt(ari)} | {fmt(ha)} | {fmt(sc, 3)} |"
        )
    return "\n".join(lines)


# ── write report ──────────────────────────────────────────────────────────────

def write_report(runs, failed, missing, out_path: str):
    n = len(runs)
    lats   = sorted(set(r["config"]["latent_dim"]     for r in runs))
    betas  = sorted(set(r["config"]["beta"]           for r in runs))
    gammas = sorted(set(r["config"]["gamma"]          for r in runs))
    lrs    = sorted(set(r["config"]["learning_rate"]  for r in runs))

    mean_va  = mean([r["metrics"].get("best_val_accuracy",  0) for r in runs])
    mean_ha  = mean([r["metrics"].get("hungarian_accuracy",  0) for r in runs])
    mean_ari = mean([r["metrics"].get("adjusted_rand_index", 0) for r in runs])

    report = f"""# Frame-Level Supervised VAE Sweep — Results Summary

_Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}_

---

## 1. Sweep Overview

The sweep jointly trained a **β-VAE with an attached MLP classifier** on
frame-level lung-ultrasound images.

Total loss: `BCE + β·KLD + γ·CE`

| Parameter | Values swept |
|---|---|
| Latent dim (`ld`) | {lats} |
| KL weight (`β`) | {betas} |
| Classifier weight (`γ`) | {gammas} |
| Learning rate (`lr`) | {lrs} |

- **Completed runs**: {n}  
- **Failed / missing**: {len(missing)} (all have `ld=30, β=5.0, γ∈{{2,5,10}}`)
- Mean val accuracy across all runs: **{pct(mean_va)}**
- Mean Hungarian accuracy across all runs: **{pct(mean_ha)}**
- Mean ARI across all runs: **{fmt(mean_ari)}**

{"### Missing combinations" if missing else ""}
{"All of the following were not produced (likely still running or hit walltime):" if missing else ""}
{"".join(chr(10) + f"- `ld={m[0]}, β={m[1]}, γ={m[2]}, lr={m[3]}`" for m in missing) if missing else ""}

---

## 2. Best Configurations

{md_best_card(runs)}

---

## 3. Top Runs

{md_top_runs_section(runs)}

---

## 4. Parameter Sensitivity

### 4.1 Effect of Latent Dimension

{md_param_effect_table(runs, "latent_dim", "ld")}

### 4.2 Effect of β (KL weight)

{md_param_effect_table(runs, "beta", "β")}

### 4.3 Effect of γ (Classifier weight)

{md_param_effect_table(runs, "gamma", "γ")}

### 4.4 Effect of Learning Rate

{md_param_effect_table(runs, "learning_rate", "lr")}

---

## 5. 2-D Interaction Heatmaps (averaged over other params)

### 5.1 Val Accuracy: ld × γ

{md_heatmap_text(runs, "latent_dim", "gamma", "best_val_accuracy", True, "ld", "γ", "Val Acc")}

### 5.2 Hungarian Accuracy: ld × γ

{md_heatmap_text(runs, "latent_dim", "gamma", "hungarian_accuracy", True, "ld", "γ", "Hung Acc")}

### 5.3 ARI: ld × γ

{md_heatmap_text(runs, "latent_dim", "gamma", "adjusted_rand_index", True, "ld", "γ", "ARI")}

### 5.4 Silhouette: ld × β

{md_heatmap_text(runs, "latent_dim", "beta", "silhouette", True, "ld", "β", "Silhouette")}

### 5.5 Val Accuracy: β × lr

{md_heatmap_text(runs, "beta", "learning_rate", "best_val_accuracy", True, "β", "lr", "Val Acc")}

### 5.6 Hungarian Accuracy: β × γ

{md_heatmap_text(runs, "beta", "gamma", "hungarian_accuracy", True, "β", "γ", "Hung Acc")}

---

## 6. Per-Class Accuracy Analysis

### 6.1 By γ (classifier weight)

{md_per_class_accuracy(runs)}

### 6.2 By Latent Dimension

{md_per_class_accuracy_by_ld(runs)}

---

## 7. Combined Score (Val Accuracy + ARI, Z-normalised)

Ranks runs on a composite score balancing supervised performance (val accuracy)
with unsupervised clustering quality (ARI).

{md_tension_table(runs)}

---

## 8. Full Results Table

Sorted by validation accuracy (descending). Bold = best in column.

{md_overview_table(runs)}

---

## 9. Analysis

### Key Findings

#### 9.1 Learning Rate is the Dominant Factor

The effect of learning rate on validation accuracy is striking. Runs with
`lr=0.001` consistently achieve the **highest val accuracy** (often >70–80%)
but the **lowest clustering metrics** (ARI, Hungarian, silhouette). This is the
classic supervised-collapse phenomenon: a large γ × large lr combination allows
the classifier head to dominate the gradient signal, effectively turning the
encoder into a pure discriminator and destroying the latent geometry that
unsupervised clustering relies upon.

Conversely, `lr=1e-5` yields the **best clustering metrics** at the cost of
lower classification accuracy, because the VAE reconstruction objective has
more influence.

`lr=1e-4` sits in the middle — the sweet spot for models where both objectives
matter.

#### 9.2 γ (Classifier Weight) Drives a Classification–Clustering Trade-off

Increasing γ from 1 → 10 systematically:
- **Increases** val accuracy and score-3 accuracy (the rarest class, score 3
  benefits most from the stronger supervision signal).
- **Decreases** silhouette score and ARI, indicating that the latent space
  becomes less geometrically coherent.

The only exception is γ=10 at small latent dims with `lr=1e-5`: the VAE
reconstruction term still dominates and clustering quality stays competitive.

#### 9.3 Latent Dimension has Moderate Impact

`ld=20` and `ld=30` tend to outperform `ld=10` on clustering metrics
(Calinski-Harabasz, silhouette), as expected — more dimensions allow more
expressive representations. However, `ld=10` matches or beats higher dims on
val accuracy in several β/γ combinations, suggesting that forcing a very
compact latent code still retains enough discriminative information when γ is
large.

#### 9.4 β=2.0 is Preferable to β=5.0

Higher β imposes a stronger pull towards the unit Gaussian prior, shrinking the
latent space and worsening classification accuracy. β=5.0 runs show reduced
val accuracy and only marginal improvements in silhouette / ARI. The 8 missing
runs are all β=5.0 with higher γ and ld=30 — the combination most likely to
hit the GPU time limit because large β+large γ+large ld makes convergence slow.

#### 9.5 Per-Class: Score 3 is Hard Everywhere

Score 3 (the rarest class, ~1% of data) shows wildly variable accuracy —
ranging from near 0% to ~57% — and is highly sensitive to lr and γ. High γ
with a moderate lr (`1e-4`) produces the most consistent score-3 recall, likely
because the class-weighting amplifies the rare-class gradient.

Score 1 (the majority class, ~44%) is consistently the *weakest* per-class
accuracy despite being common, probably because it occupies the largest and most
ambiguous region of the severity spectrum.

### Recommended Configuration

For **best combined performance** (val accuracy + clustering quality):

```
latent_dim = 20
beta       = 2.0
gamma      = 5.0
lr         = 1e-4
```

This sits near the top of the combined Z-score ranking and avoids the
supervised-collapse regime. If downstream use is purely classification, prefer
`lr=1e-3`, `gamma=10`, `ld=20`. If only unsupervised clustering matters, use
`lr=1e-5`, `gamma=1`, `ld=30`.
"""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Report written to: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Summarise frame-level sweep results")
    parser.add_argument(
        "--json",
        default="frame_level/results/sweep/sweep_summary.json",
        help="Path to sweep_summary.json",
    )
    parser.add_argument(
        "--out",
        default="frame_level/results/sweep/sweep_report.md",
        help="Output markdown path",
    )
    args = parser.parse_args()

    runs, failed = load_data(args.json)
    missing = find_missing(runs)

    print(f"Loaded {len(runs)} successful runs, {len(failed)} failed, {len(missing)} missing combos")

    write_report(runs, failed, missing, args.out)


if __name__ == "__main__":
    main()
