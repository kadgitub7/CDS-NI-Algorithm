"""Computational cost and memory usage profiling."""
import json, sys, os, time, tracemalloc
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from cds_ovr import (load_data, classify_features, build_tree, train, predict,
                     run_10fold, run_split, HEALTHY)


def main():
    t0 = time.perf_counter()
    print("Computational Cost and Memory Profiling", flush=True)
    data, labels = load_data()
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    n = data.shape[0]

    # Time 10-fold CV
    print("  Timing 10-fold CV...", flush=True)
    t1 = time.perf_counter()
    run_10fold(data, labels, is_bin, seed=13)
    t_10fold = time.perf_counter() - t1
    print(f"    10-fold CV: {t_10fold:.2f}s", flush=True)

    # Time 90/10 split
    print("  Timing 90/10 split...", flush=True)
    t1 = time.perf_counter()
    run_split(data, labels, is_bin, seed=13, train_frac=0.9)
    t_90_10 = time.perf_counter() - t1
    print(f"    90/10 split: {t_90_10:.2f}s", flush=True)

    # Single training + prediction with memory tracking
    print("  Profiling memory...", flush=True)
    tracemalloc.start()
    t1 = time.perf_counter()
    nodes = build_tree(data, labels, is_bin)
    train_result = train(nodes, data, labels, is_bin, all_cls)
    t_train = time.perf_counter() - t1
    mem_after_train = tracemalloc.get_traced_memory()

    t1 = time.perf_counter()
    for uid in range(n):
        predict(uid, data, nodes, all_cls, train_result)
    t_predict_all = time.perf_counter() - t1
    mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t_per_patient = t_predict_all / n
    data_mem_mb = data.nbytes / (1024**2)

    output = {
        "10fold_cv_time_s": round(t_10fold, 2),
        "90_10_split_time_s": round(t_90_10, 2),
        "training_time_s": round(t_train, 4),
        "prediction_all_patients_s": round(t_predict_all, 4),
        "prediction_per_patient_ms": round(t_per_patient * 1000, 4),
        "memory_current_bytes": mem_after_train[0],
        "memory_peak_bytes": mem_peak[1],
        "memory_current_mb": round(mem_after_train[0] / (1024**2), 2),
        "memory_peak_mb": round(mem_peak[1] / (1024**2), 2),
        "data_matrix_mb": round(data_mem_mb, 2),
        "n_tree_nodes": len(nodes),
        "n_class_models": sum(len(train_result[0][c]) for c in all_cls),
        "n_retained_features": sum(len(train_result[1][c]) for c in all_cls),
        "platform": sys.platform,
    }

    print(f"  Training: {t_train:.4f}s", flush=True)
    print(f"  Prediction (all): {t_predict_all:.4f}s ({t_per_patient*1000:.4f}ms/patient)", flush=True)
    print(f"  Memory: {mem_after_train[0]/(1024**2):.2f}MB current, {mem_peak[1]/(1024**2):.2f}MB peak", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "results_cost.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
