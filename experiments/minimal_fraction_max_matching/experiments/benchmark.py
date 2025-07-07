"""
• Generates Erdős-Rényi graphs G(n, p)
• Runs:
    - minimal_fraction_max_matching  (combinatorial)
    - solve_fractional_matching_lp   (MILP via PuLP/cvxpy) # not used too slow for comparison
    - networkx.maximum_matching      (integral baseline)
    - networkx.maximal_matching      (simple Maximal matching baseline)
• Records value and wall-clock time → CSV
"""

from __future__ import annotations
import argparse, time, random, logging, pathlib
import sys, os, tempfile, json, subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import pandas as pd
import networkx as nx

from algorithms.fractional_matching import (
    minimal_fraction_max_matching,
    solve_fractional_matching_lp,
    matching_value
)

log = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

DATA_DIR = pathlib.Path("output")
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "benchmarks.csv"


# ───────────────────────── graph generator ─────────────────────────
def er_graph(n: int, p: float) -> nx.Graph:
    """Undirected G(n, p) with a fresh seed each call."""
    return nx.fast_gnp_random_graph(n, p, seed=random.randrange(1_000_000))


# ───────────────────────── one experiment ─────────────────────────
def run_one(n: int, p: float, time_cap: float = 60.0) -> dict[str, float | int]:
    G = er_graph(n, p)

    # -------- fractional (combinatorial) ----------
    t0 = time.perf_counter()
    frac_cmp = minimal_fraction_max_matching(G)
    cmp_time = time.perf_counter() - t0
    cmp_val = matching_value(frac_cmp)

    # -------- integral (greedy) ----------
    t0 = time.perf_counter()
    greedy = nx.maximal_matching(G)
    gr_time = time.perf_counter() - t0
    gr_val = len(greedy)
    
    # -------- C++ implementation ----------
    cpp_exe = pathlib.Path(project_root) / "algorithms" / "cpp_matching"
    if cpp_exe.exists():
        # Save graph to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            graph_file = f.name
            # Write graph in simple format: n m followed by edges
            f.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n")
            for u, v in G.edges():
                f.write(f"{u} {v}\n")
        
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                [str(cpp_exe), graph_file], 
                capture_output=True, 
                text=True,
                timeout=time_cap
            )
            cpp_time = time.perf_counter() - t0
            
            # Parse output
            cpp_val = float(result.stdout.strip())
            log.info(f"C++ implementation: value={cpp_val}, time={cpp_time:.3f}s")
        except (subprocess.TimeoutExpired, ValueError, subprocess.CalledProcessError) as e:
            log.error(f"C++ implementation failed: {e}")
            cpp_time = time_cap
            cpp_val = float('nan')
        finally:
            os.unlink(graph_file)
    else:
        log.warning("C++ executable not found at %s", cpp_exe)
        cpp_time = float('nan')
        cpp_val = float('nan')

    # -------- return results ----------
    return dict(
        n=n, p=p,
        cmp_val=cmp_val, cmp_time=cmp_time,
        gr_val=gr_val, gr_time=gr_time,
        cpp_val=cpp_val, cpp_time=cpp_time  
    )


# ───────────────────────── main loop ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[200, 400, 800],
                    help="vertex counts to test")
    ap.add_argument("--p", type=float, default=0.05,
                    help="edge probability for G(n,p)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="independent trials per n")
    ap.add_argument("--time-cap", type=float, default=60.0,
                    help="stop when combinatorial algo exceeds this many seconds")
    ap.add_argument("--skip-lp", action="store_true",
                    help="Skip the LP implementation")
    args = ap.parse_args()

    rows: list[dict] = []
    for n in args.sizes:
        for r in range(args.repeat):
            res = run_one(n, p=args.p, time_cap=args.time_cap)  # Pass time_cap here
            rows.append(res)
            log.info("✓ n=%d rep=%d  cmp=%.3fs  gr=%.3fs",
                     n, r, res['cmp_time'], res['gr_time'])  
            
            # If we hit the time cap, break
            if res["cmp_time"] > args.time_cap:
                log.warning("Exceeded time cap (%.1fs > %gs) - stopping", 
                           res["cmp_time"], args.time_cap)
                break
        else:
            # continue outer loop if inner did **not** break
            continue
        break  # inner break -> stop sizes loop

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    log.info("Saved %d benchmark results ➜ %s", len(rows), CSV_PATH)


if __name__ == "__main__":
    main()
