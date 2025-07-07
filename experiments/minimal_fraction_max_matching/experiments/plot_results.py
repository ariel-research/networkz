"""
plot_results.py - pretty plots from experiments/minimal_fraction_max_matching/output/benchmarks.csv
• collapses repetitions (mean std)
• draws error bars
• distinct markers + line styles (LP is dotted)
"""

from __future__ import annotations
import argparse
import logging
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

CSV = pathlib.Path("experiments/minimal_fraction_max_matching/output") / "benchmarks.csv"
if not CSV.exists():
    log.error(f"No data to plot! Run benchmark.py first to generate {CSV}")
    exit(1)

# Read the CSV file
df = pd.read_csv(CSV).dropna(subset=["cmp_val", "gr_val"])  # Remove lp_val

# Check if C++ results exist in the dataframe
has_cpp = "cpp_time" in df.columns and "cpp_val" in df.columns
if has_cpp:
    # Check if we have at least one valid value
    has_cpp = not df["cpp_time"].isna().all() and not df["cpp_val"].isna().all()
    if has_cpp:
        log.info("Found C++ implementation results, including in plots")
    else:
        log.warning("C++ columns exist but contain only NaN values")
else:
    log.warning("No C++ implementation results found in CSV")

# Add argument parser
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lp", action="store_true",
                    help="LP implementation was skipped during benchmarking")
    args = ap.parse_args()

# Define aggregation functions for each column
agg_dict = {
    "cmp_time": ["mean", "std"],
    "cmp_val": ["mean", "std"],
    "gr_time": ["mean", "std"],
    "gr_val": ["mean", "std"],
}

# Comment out the LP sections
# if "lp_time" in df.columns:
#     agg_dict["lp_time"] = ["mean", "std"]
#     agg_dict["lp_val"] = ["mean", "std"]

# Add C++ aggregations if the columns exist
if has_cpp:
    agg_dict["cpp_time"] = ["mean", "std"]
    agg_dict["cpp_val"] = ["mean", "std"]

# Group by n and aggregate
grp = df.groupby("n").agg(agg_dict)
# Flatten the multi-level column index
grp.columns = ['_'.join(col).strip() for col in grp.columns.values]
grp = grp.reset_index()

# Helper function to plot with error bars
def _plot(ax, x, y, std, label):
    ax.errorbar(x, y, yerr=std, label=label, 
                marker=STYLES[label]["marker"], 
                linestyle=STYLES[label]["linestyle"],
                linewidth=STYLES[label]["linewidth"])

# Define plot styles
STYLES = {
    "PythonVersion": dict(marker="o", linestyle="-",  linewidth=1.6),  # solid
    "Simple":        dict(marker="^", linestyle="--", linewidth=1.6),  # dashed
    "cppOpt":           dict(marker="*", linestyle="-.", linewidth=1.6),  # dash-dot
}

# ───────────────────────── run-time plot ─────────────────────────
fig_rt, ax_rt = plt.subplots(figsize=(7, 5))
_plot(ax_rt, grp["n"], grp["cmp_time_mean"], grp["cmp_time_std"], "PythonVersion")
_plot(ax_rt, grp["n"], grp["gr_time_mean"], grp["gr_time_std"], "Simple")
if has_cpp:
    # Use proper column names after flattening
    if "cpp_time_mean" in grp.columns:
        _plot(ax_rt, grp["n"], grp["cpp_time_mean"], grp["cpp_time_std"], "cppOpt")
        log.info("Added cppOpt to runtime plot")

# ax_rt.set_yscale("log")  # Remove log scale
ax_rt.set_xlabel("n (vertices)")
ax_rt.set_ylabel("run-time (seconds)")
ax_rt.set_title("RunTime/ amount of Vertices")
ax_rt.grid(True, linestyle="--", linewidth=0.3)
ax_rt.legend(framealpha=0.92)
fig_rt.tight_layout()
fig_rt.savefig(CSV.parent / "Runtime2.png", dpi=150)
plt.close(fig_rt)

# ───────────────────────── value plot ─────────────────────────
fig_val, ax_val = plt.subplots(figsize=(7, 5))
_plot(ax_val, grp["n"], grp["cmp_val_mean"], grp["cmp_val_std"], "PythonVersion")
_plot(ax_val, grp["n"], grp["gr_val_mean"], grp["gr_val_std"], "Simple")
if has_cpp:
    # Use proper column names after flattening
    if "cpp_val_mean" in grp.columns:
        _plot(ax_val, grp["n"], grp["cpp_val_mean"], grp["cpp_val_std"], "cppOpt")
        log.info("Added cppOpt to value plot")

ax_val.set_xlabel("n (vertices)")
ax_val.set_ylabel("matching value")
ax_val.set_title("Matching Result")
ax_val.grid(True, linestyle="--", linewidth=0.3)
ax_val.legend(framealpha=0.92)
fig_val.tight_layout()
fig_val.savefig(CSV.parent / "MatchingResult2.png", dpi=150)
plt.close(fig_val)

log.info("Plots generated")
