#!/usr/bin/env python3
"""Summarize all completed eval experiments into per-model markdown reports.

Detects the base model from directory names (e.g. pi0, pi05) and writes a
separate ``results_summary_<model>.md`` for each model.  Only the first three
core tables are generated:

  1. Symmetric delay  (image = state = d)
  2. Image-only delay (state = 0)
  3. State-only delay (image = 0)
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_DIR = Path(__file__).parent / "outputs" / "eval"
RESULTS_DIR = Path(__file__).parent / "results"
TASK_GROUPS = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DELAY_LEVELS = [0, 1, 2, 4, 8, 16, 32]

# Regex that captures: <model>_delay_images<N>_state<N>_<aug|comp>-<mode>
_DIR_RE = re.compile(
    r"_libero_(?P<model>[a-zA-Z0-9]+)"   # model name after "libero_"
    r"(?:_delay_images(?P<img>\d+)_state(?P<st>\d+)"
    r"_(?:aug|comp)-(?P<comp>\w+))?"      # optional delay suffix
)


def load_all_experiments() -> dict[str, dict]:
    """Return ``{model: {(img, st, comp): result_dict}}``."""
    all_exps: dict[str, dict] = defaultdict(dict)

    for jf in sorted(EVAL_DIR.rglob("eval_info.json")):
        dirname = jf.parent.name
        m = _DIR_RE.search(dirname)
        if not m:
            continue

        model = m.group("model")
        if m.group("img") is not None:
            img, st, comp = int(m.group("img")), int(m.group("st")), m.group("comp")
        else:
            img, st, comp = 0, 0, "none"

        with open(jf) as f:
            data = json.load(f)

        per_group = data.get("per_group", {})
        overall = data.get("overall", {})

        result = {}
        for g in TASK_GROUPS:
            result[g] = per_group.get(g, {}).get("pc_success")
        result["average"] = overall.get("pc_success")

        key = (img, st, comp)
        exps = all_exps[model]
        if key in exps:
            old = exps[key]
            for g in TASK_GROUPS + ["average"]:
                if old[g] is not None and result[g] is not None:
                    result[g] = (old[g] + result[g]) / 2
            result["_count"] = old.get("_count", 1) + 1
        else:
            result["_count"] = 1
        exps[key] = result

    return dict(all_exps)


def fmt_val(v):
    if v is None:
        return "—"
    return f"{v:.1f}"


def print_comparison_table(title, rows, col_headers, row_label):
    """Pretty-print a table to stdout."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

    header = [row_label] + col_headers
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [max(len(header[i]), *(len(r[i]) for r in str_rows)) + 2 for i in range(len(header))]

    def fmt_line(vals):
        return " | ".join(v.center(w) for v, w in zip(vals, widths))

    sep = "-+-".join("-" * w for w in widths)
    print(fmt_line(header))
    print(sep)
    for r in str_rows:
        print(fmt_line(r))
    print()


def write_md_table(f, title, description, rows, col_headers, row_label):
    f.write(f"\n### {title}\n\n")
    if description:
        f.write(f"{description}\n\n")
    header = [row_label] + col_headers
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join(":---:" for _ in header) + "|\n")
    for row in rows:
        f.write("| " + " | ".join(str(c) for c in row) + " |\n")


# ── per-model report ────────────────────────────────────────────────────────

def generate_report(model: str, exps: dict):
    """Generate 3-table markdown report for a single model."""
    baseline = exps.get((0, 0, "none"))
    baseline_avg = fmt_val(baseline["average"]) if baseline else "—"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = RESULTS_DIR / f"results_summary_{model}.md"
    md = open(md_path, "w")
    md.write(f"# Evaluation Results Summary — {model}\n\n")
    md.write(f"**Policy:** `{model}`\n\n")
    md.write(f"**Baseline (no delay):** avg = {baseline_avg}\n")
    if baseline:
        md.write("| " + " | ".join(TASK_GROUPS + ["Average"]) + " |\n")
        md.write("|" + "|".join(":---:" for _ in TASK_GROUPS + ["Average"]) + "|\n")
        vals = [fmt_val(baseline[g]) for g in TASK_GROUPS] + [fmt_val(baseline["average"])]
        md.write("| " + " | ".join(vals) + " |\n")

    col_headers = TASK_GROUPS + ["Average"]

    def _build_table_rows(key_fn, label_fn, exps):
        """Build table rows, interleaving belief rows when available."""
        rows = []
        has_any_belief = any(exps.get((*key_fn(d)[:2], "belief")) for d in DELAY_LEVELS)
        for d in DELAY_LEVELS:
            # No-compensation row
            e_none = exps.get((*key_fn(d)[:2], "none"))
            lbl = label_fn(d)
            comp_col = ["none"] if has_any_belief else []
            if e_none:
                rows.append([lbl] + comp_col + [fmt_val(e_none[g]) for g in TASK_GROUPS] + [fmt_val(e_none["average"])])
            else:
                rows.append([lbl] + comp_col + ["—"] * (len(TASK_GROUPS) + 1))
            # Belief row (skip d=0 — no delay means no compensation)
            if has_any_belief and d > 0:
                e_belief = exps.get((*key_fn(d)[:2], "belief"))
                comp_col_b = ["**belief**"]
                if e_belief:
                    rows.append([lbl] + comp_col_b + [fmt_val(e_belief[g]) for g in TASK_GROUPS] + [fmt_val(e_belief["average"])])
                else:
                    rows.append([lbl] + comp_col_b + ["—"] * (len(TASK_GROUPS) + 1))
        extra_col = ["Comp."] if has_any_belief else []
        return rows, extra_col

    # ── Table 1: Symmetric delay ────────────────────────────────────────
    title1 = "Table 1: Symmetric Delay ($\\delta_{\\mathrm{img}}=\\delta_{\\mathrm{state}}$)"
    desc1 = "Both image and state delayed by the same amount."
    key_fn1 = lambda d: (d, d, "none")
    label_fn1 = lambda d: f"$\\delta={d}$" + (" ★" if d == 0 else "")
    rows1, extra_col1 = _build_table_rows(key_fn1, label_fn1, exps)
    print_comparison_table(f"[{model}] {title1}", rows1, extra_col1 + col_headers, "Delay")
    write_md_table(md, title1, desc1, rows1, extra_col1 + col_headers, "Delay")

    # ── Table 2: Image-only delay ───────────────────────────────────────
    title2 = "Table 2: Image-Only Delay ($\\delta_{\\mathrm{state}}=0$)"
    desc2 = "Only image observations are delayed, state is real-time."
    key_fn2 = lambda d: (d, 0, "none")
    label_fn2 = lambda d: f"$\\delta_{{\\mathrm{{img}}}}={d}$" + (" ★" if d == 0 else "")
    rows2, extra_col2 = _build_table_rows(key_fn2, label_fn2, exps)
    print_comparison_table(f"[{model}] {title2}", rows2, extra_col2 + col_headers, "Delay")
    write_md_table(md, title2, desc2, rows2, extra_col2 + col_headers, "Delay")

    # ── Table 3: State-only delay ───────────────────────────────────────
    title3 = "Table 3: State-Only Delay ($\\delta_{\\mathrm{img}}=0$)"
    desc3 = "Only state/proprioception is delayed, images are real-time."
    key_fn3 = lambda d: (0, d, "none")
    label_fn3 = lambda d: f"$\\delta_{{\\mathrm{{state}}}}={d}$" + (" ★" if d == 0 else "")
    rows3, extra_col3 = _build_table_rows(key_fn3, label_fn3, exps)
    print_comparison_table(f"[{model}] {title3}", rows3, extra_col3 + col_headers, "Delay")
    write_md_table(md, title3, desc3, rows3, extra_col3 + col_headers, "Delay")

    md.write(f"\n---\n★ = baseline (no delay)\n")
    md.write(f"\n*Policy: {model} — Generated from {len(exps)} unique experiment configs.*\n")
    md.close()
    print(f"  → Markdown saved to: {md_path}")


# ── charts ──────────────────────────────────────────────────────────────────

_MODEL_DISPLAY = {"pi0": "PI0", "pi05": "PI0.5"}
_MODEL_MARKERS = {"pi0": "o", "pi05": "s"}
_MODEL_COLORS  = {"pi0": "#1f77b4", "pi05": "#ff7f0e"}
_COMP_MODES    = ["none", "belief"]
_COMP_STYLE    = {
    "none":   {"linestyle": "-",  "suffix": ""},
    "belief": {"linestyle": "--", "suffix": " + Belief"},
}


_METRIC_KEYS = TASK_GROUPS + ["average"]
_METRIC_DISPLAY = {
    "libero_spatial": "LIBERO-Spatial",
    "libero_object": "LIBERO-Object",
    "libero_goal": "LIBERO-Goal",
    "libero_10": "LIBERO-10",
    "average": "Average",
}


def _extract_series(exps: dict, key_fn, metric: str):
    """Extract (delays, values) for one scenario + metric using *key_fn(d)*."""
    delays, vals = [], []
    for d in DELAY_LEVELS:
        e = exps.get(key_fn(d))
        v = e[metric] if e and e.get(metric) is not None else None
        delays.append(d)
        vals.append(v)
    return delays, vals


def _has_comp_data(exps: dict, base_key_fn, comp: str) -> bool:
    """Check if any delay level has data for the given compensation mode."""
    for d in DELAY_LEVELS:
        img, st, _ = base_key_fn(d)
        if exps.get((img, st, comp)):
            return True
    return False


def generate_charts(all_exps: dict[str, dict]):
    """Generate cross-model comparison charts (per-task subplots) and save to RESULTS_DIR."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = [
        (
            "Symmetric Delay",
            r"$\delta_{\mathrm{img}}=\delta_{\mathrm{state}}=\delta$",
            lambda d: (d, d, "none"),
            "symmetric_delay",
        ),
        (
            "Image-Only Delay",
            r"$\delta_{\mathrm{img}}=\delta,\ \delta_{\mathrm{state}}=0$",
            lambda d: (d, 0, "none"),
            "image_only_delay",
        ),
        (
            "State-Only Delay",
            r"$\delta_{\mathrm{img}}=0,\ \delta_{\mathrm{state}}=\delta$",
            lambda d: (0, d, "none"),
            "state_only_delay",
        ),
    ]

    models = sorted(all_exps.keys())

    for title, subtitle, base_key_fn, fname in scenarios:
        n_metrics = len(_METRIC_KEYS)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4.0 * n_metrics, 4.0))

        for idx, metric in enumerate(_METRIC_KEYS):
            ax = axes[idx]
            for model in models:
                model_exps = all_exps[model]
                base_color = _MODEL_COLORS.get(model, None)
                display = _MODEL_DISPLAY.get(model, model)
                marker = _MODEL_MARKERS.get(model, "^")

                for comp in _COMP_MODES:
                    # Skip compensation modes that have no data for this model+scenario
                    if not _has_comp_data(model_exps, base_key_fn, comp):
                        continue
                    style = _COMP_STYLE[comp]
                    key_fn = lambda d, _c=comp: (*base_key_fn(d)[:2], _c)
                    delays, vals = _extract_series(model_exps, key_fn, metric)
                    plot_d = [d for d, v in zip(delays, vals) if v is not None]
                    plot_v = [v for v in vals if v is not None]
                    if not plot_v:
                        continue
                    ax.plot(
                        plot_d, plot_v,
                        marker=marker,
                        color=base_color,
                        linestyle=style["linestyle"],
                        linewidth=2, markersize=6,
                        label=f"{display}{style['suffix']}",
                    )
            ax.set_xlabel(r"$\delta$", fontsize=11)
            ax.set_ylabel("Success Rate (%)", fontsize=11)
            ax.set_title(_METRIC_DISPLAY.get(metric, metric), fontsize=12, fontweight="bold")
            ax.set_xticks(DELAY_LEVELS)
            ax.set_ylim(0, 105)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{title}  —  {subtitle}", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        img_path = RESULTS_DIR / f"{fname}.png"
        fig.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Chart saved to: {img_path}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    if not EVAL_DIR.exists():
        print(f"Error: {EVAL_DIR} does not exist.")
        sys.exit(1)

    all_exps = load_all_experiments()
    if not all_exps:
        print("No experiments found.")
        sys.exit(0)

    print(f"Found models: {', '.join(sorted(all_exps.keys()))}")
    for model in sorted(all_exps.keys()):
        exps = all_exps[model]
        print(f"\n{'─'*60}")
        print(f"  Model: {model}  ({len(exps)} experiment configs)")
        print(f"{'─'*60}")
        generate_report(model, exps)

    # Cross-model comparison charts
    print(f"\n{'─'*60}")
    print("  Generating cross-model comparison charts...")
    print(f"{'─'*60}")
    generate_charts(all_exps)


if __name__ == "__main__":
    main()
