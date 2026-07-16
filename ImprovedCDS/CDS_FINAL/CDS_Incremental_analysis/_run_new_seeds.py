"""Run all variants with 5 additional seeds and merge results into CSVs."""
import os
import sys
import subprocess
import csv
import re
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

DIR = Path(__file__).resolve().parent
TIMEOUT = 3600
NEW_SEEDS = [7, 21, 42, 58, 83]


def find_variant_files():
    files = []
    for f in sorted(DIR.glob("*.py")):
        name = f.stem
        if name.startswith('_') or name == 'run_all':
            continue
        files.append(f)
    return files


def parse_per_seed(output):
    seeds_data = []
    current_seed = None
    for line in output.split('\n'):
        if re.match(r'\s*SUMMARY', line) or '+/-' in line:
            current_seed = None
            continue
        sm = re.match(r'\s+Seed\s+(\d+):', line)
        if sm:
            current_seed = int(sm.group(1))
            seeds_data.append({'seed': current_seed})
            continue
        if current_seed is None or not seeds_data:
            continue
        entry = seeds_data[-1]
        m10 = re.match(r'\s+10-fold CV:\s+([\d.]+)%', line)
        if m10:
            entry['10fold_cv'] = float(m10.group(1))
        m90 = re.match(r'\s+90/10 multiclass:\s+([\d.]+)%\s+binary:\s+([\d.]+)%', line)
        if m90:
            entry['9010_multi'] = float(m90.group(1))
            entry['9010_binary'] = float(m90.group(2))
        m60 = re.match(r'\s+60/40 multiclass:\s+([\d.]+)%', line)
        if m60:
            entry['6040_multi'] = float(m60.group(1))
    return seeds_data


def run_one(fpath):
    name = fpath.stem
    seed_str = ",".join(str(s) for s in NEW_SEEDS)
    env = os.environ.copy()
    env["CDS_SEEDS"] = seed_str
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(fpath)],
            capture_output=True, text=True, timeout=TIMEOUT,
            cwd=str(DIR), env=env
        )
        elapsed = time.time() - t0
        output = result.stdout + result.stderr
        seeds = parse_per_seed(output)
        return name, elapsed, seeds, result.returncode, None
    except subprocess.TimeoutExpired:
        return name, time.time() - t0, [], -1, 'TIMEOUT'
    except Exception as e:
        return name, time.time() - t0, [], -1, str(e)


def main():
    workers = os.cpu_count() or 4
    args = sys.argv[1:]
    pattern = None
    i = 0
    while i < len(args):
        if args[i] == '--filter' and i + 1 < len(args):
            pattern = args[i + 1]; i += 2
        elif args[i] == '--workers' and i + 1 < len(args):
            workers = int(args[i + 1]); i += 2
        else:
            i += 1

    files = find_variant_files()
    if pattern:
        files = [f for f in files if pattern.lower() in f.stem.lower()]

    print(f"Running {len(files)} variants with NEW seeds {NEW_SEEDS}")
    print(f"Workers: {workers}, timeout: {TIMEOUT}s each")
    print(f"{'='*70}", flush=True)

    # Read existing per-seed data
    existing_rows = []
    seed_csv = DIR / 'results_per_seed.csv'
    if seed_csv.exists():
        with open(seed_csv, encoding='utf-8') as f:
            existing_rows = list(csv.DictReader(f))
        print(f"Loaded {len(existing_rows)} existing per-seed rows")

    new_rows = []
    completed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, f): f for f in files}
        for future in as_completed(futures):
            name, elapsed, seeds, rc, err = future.result()
            completed += 1
            status = 'OK' if err is None else err
            n_seeds = len(seeds)
            print(f"  [{completed}/{len(files)}] {name:40s} {elapsed:6.0f}s  "
                  f"{n_seeds} seeds  {status}", flush=True)
            for sd in seeds:
                new_rows.append({'variant': name, **sd})

    # Merge and write per-seed CSV
    all_rows = existing_rows + new_rows
    all_rows.sort(key=lambda r: (r['variant'], int(r.get('seed', 0))))

    seed_fields = ['variant', 'seed', '10fold_cv', '9010_multi', '9010_binary', '6040_multi']
    with open(seed_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=seed_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    # Rebuild summary CSV with peak values
    by_variant = defaultdict(lambda: defaultdict(list))
    variant_order = []
    seen = set()
    for row in all_rows:
        v = row['variant']
        if v not in seen:
            seen.add(v)
            variant_order.append(v)
        for col in ['10fold_cv', '9010_multi', '9010_binary', '6040_multi']:
            val = row.get(col, '')
            if val:
                by_variant[v][col].append(float(val))

    metric_names = {
        '10fold_cv': '10-fold CV',
        '9010_multi': '90/10 multiclass',
        '9010_binary': '90/10 binary',
        '6040_multi': '60/40 multiclass',
    }
    summary_fields = ['variant', '10-fold CV', '90/10 multiclass',
                      '90/10 binary', '60/40 multiclass']

    summary_csv = DIR / 'results_summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for v in variant_order:
            row = {'variant': v}
            for col, name in metric_names.items():
                vals = by_variant[v].get(col, [])
                row[name] = f'{max(vals):.1f}' if vals else ''
            writer.writerow(row)

    wall = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Done in {wall:.0f}s. Merged {len(new_rows)} new + "
          f"{len(existing_rows)} existing = {len(all_rows)} total rows")
    print(f"Results saved to:")
    print(f"  {seed_csv}")
    print(f"  {summary_csv}")
    print(f"{'='*70}")

    print(f"\n{'Variant':<40s} {'10-fold':>8s} {'90/10m':>8s} {'Binary':>8s} {'60/40':>8s} {'Seeds':>6s}")
    print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for v in variant_order:
        d = by_variant[v]
        n_seeds = len(d.get('10fold_cv', d.get('9010_multi', [])))
        cv = f"{max(d['10fold_cv']):.1f}" if d.get('10fold_cv') else '?'
        m9 = f"{max(d['9010_multi']):.1f}" if d.get('9010_multi') else '?'
        b9 = f"{max(d['9010_binary']):.1f}" if d.get('9010_binary') else '?'
        m6 = f"{max(d['6040_multi']):.1f}" if d.get('6040_multi') else '?'
        print(f"{v:<40s} {cv:>8s} {m9:>8s} {b9:>8s} {m6:>8s} {n_seeds:>6d}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
