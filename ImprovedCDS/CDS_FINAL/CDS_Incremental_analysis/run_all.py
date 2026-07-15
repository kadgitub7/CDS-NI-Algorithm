"""Run all CDS incremental analysis variants in PARALLEL and collect results.

Runs each variant file as a subprocess, captures output, and generates a summary CSV.
Usage: python run_all.py [--filter PATTERN] [--workers N]
"""
import os
import sys
import subprocess
import csv
import re
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

DIR = Path(__file__).resolve().parent
TIMEOUT = 3600


def find_variant_files():
    files = []
    for f in sorted(DIR.glob("*.py")):
        name = f.stem
        if name.startswith('_') or name == 'run_all':
            continue
        files.append(f)
    return files


def parse_summary(output):
    results = {}
    for line in output.split('\n'):
        m = re.match(r'\s+([\w/ -]+?):\s+([\d.]+)%\s*\+/-\s*([\d.]+)%\s*\(range:\s*([\d.]+)%-([\d.]+)%\)', line)
        if m:
            key = m.group(1).strip()
            results[key] = {
                'mean': float(m.group(2)),
                'std': float(m.group(3)),
                'min': float(m.group(4)),
                'max': float(m.group(5)),
            }
    return results


def parse_per_seed(output):
    """Parse individual seed results for detailed CSV."""
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
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(fpath)],
            capture_output=True, text=True, timeout=TIMEOUT,
            cwd=str(DIR)
        )
        elapsed = time.time() - t0
        output = result.stdout + result.stderr
        summary = parse_summary(output)
        seeds = parse_per_seed(output)
        return name, elapsed, summary, seeds, result.returncode, None
    except subprocess.TimeoutExpired:
        return name, time.time() - t0, {}, [], -1, 'TIMEOUT'
    except Exception as e:
        return name, time.time() - t0, {}, [], -1, str(e)


def main():
    pattern = None
    workers = os.cpu_count() or 4
    args = sys.argv[1:]
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

    print(f"Running {len(files)} variants in parallel ({workers} workers, "
          f"timeout={TIMEOUT}s each)")
    print(f"{'='*70}", flush=True)

    all_results = []
    all_seeds = []
    completed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, f): f for f in files}
        for future in as_completed(futures):
            name, elapsed, summary, seeds, rc, err = future.result()
            completed += 1
            status = 'OK' if err is None else err
            print(f"  [{completed}/{len(files)}] {name:40s} {elapsed:6.0f}s  {status}",
                  flush=True)

            row = {'variant': name, 'time_s': f"{elapsed:.0f}"}
            for metric in ['10-fold CV', '90/10 multiclass',
                           '90/10 binary', '60/40 multiclass']:
                if metric in summary:
                    row[f'{metric}_mean'] = summary[metric]['mean']
                    row[f'{metric}_std'] = summary[metric]['std']
                    row[f'{metric}_min'] = summary[metric]['min']
                    row[f'{metric}_max'] = summary[metric]['max']
                else:
                    row[f'{metric}_mean'] = ''
                    row[f'{metric}_std'] = ''
                    row[f'{metric}_min'] = ''
                    row[f'{metric}_max'] = ''
            if err:
                row['error'] = err
            all_results.append(row)

            for sd in seeds:
                all_seeds.append({'variant': name, **sd})

    all_results.sort(key=lambda r: r['variant'])
    all_seeds.sort(key=lambda r: (r['variant'], r.get('seed', 0)))

    # Write summary CSV
    csv_path = DIR / 'results_summary.csv'
    fieldnames = ['variant', 'time_s',
                  '10-fold CV_mean', '10-fold CV_std', '10-fold CV_min', '10-fold CV_max',
                  '90/10 multiclass_mean', '90/10 multiclass_std', '90/10 multiclass_min', '90/10 multiclass_max',
                  '90/10 binary_mean', '90/10 binary_std', '90/10 binary_min', '90/10 binary_max',
                  '60/40 multiclass_mean', '60/40 multiclass_std', '60/40 multiclass_min', '60/40 multiclass_max',
                  'error']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    # Write per-seed CSV
    seed_csv = DIR / 'results_per_seed.csv'
    seed_fields = ['variant', 'seed', '10fold_cv', '9010_multi', '9010_binary', '6040_multi']
    with open(seed_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=seed_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_seeds)

    wall = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Done in {wall:.0f}s wall time. Results saved to:")
    print(f"  {csv_path}")
    print(f"  {seed_csv}")
    print(f"{'='*70}")

    # Print summary table
    print(f"\n{'Variant':<40s} {'10-fold':>8s} {'90/10m':>8s} {'Binary':>8s} {'60/40':>8s}")
    print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        name = r['variant'][:40]
        def fmt(k):
            v = r.get(k, '')
            return f"{v}" if v != '' else '?'
        print(f"{name:<40s} {fmt('10-fold CV_mean'):>8s} "
              f"{fmt('90/10 multiclass_mean'):>8s} "
              f"{fmt('90/10 binary_mean'):>8s} "
              f"{fmt('60/40 multiclass_mean'):>8s}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
