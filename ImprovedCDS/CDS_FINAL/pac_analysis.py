"""PAC Figure 8 — Evidence Ratio Trajectory for an early-stopping patient.

Runs instrumented 10-fold CV to find a patient that early-stops with
multiple disease class trajectories visible, then generates a publication-
quality matplotlib figure.
"""
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(__file__))
import cds_ovr_v4_pac_twd as pac

SEEDS = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def instrumented_predict(uid, data, nodes, all_cls, train_result):
    """Predict with full PAC instrumentation."""
    class_models, class_retained = train_result
    disease_cls = [c for c in all_cls if c != pac.HEALTHY]

    ag_h = pac.AGAINST_SCALE_MAP.get(pac.HEALTHY, 0.8)
    af_h = pac._compute_af(uid, data, nodes, class_models[pac.HEALTHY],
                           class_retained[pac.HEALTHY], ag_h)
    h_score = af_h[6][-1] if af_h[6] else 1.0
    healthy_bar = min(pac.HEALTHY_WEIGHT * h_score, pac.HEALTHY_BAR_CAP)

    class_trajs = {}
    class_thresholds = {}
    class_n_features = {}

    for cls in disease_cls:
        ag = pac.AGAINST_SCALE_MAP.get(cls, 0.8)
        af = pac._compute_af(uid, data, nodes, class_models[cls],
                             class_retained[cls], ag)
        traj = af[6]
        class_trajs[cls] = traj
        class_n_features[cls] = af[2]
        t = pac.CLASS_THRESHOLDS.get(cls, 3.0)
        if h_score < pac.SUSPICION_HCUT:
            t -= pac.SUSPICION_OFFSET
        class_thresholds[cls] = max(t, healthy_bar)

    max_steps = max((len(t) for t in class_trajs.values()), default=0)
    early_stop_step = None
    early_stop_cls = None
    early_stop_excess = None

    for step in range(max_steps):
        early_candidates = {}
        for cls in disease_cls:
            traj = class_trajs[cls]
            if step >= len(traj):
                continue
            ratio = traj[step]
            t = class_thresholds[cls]
            if ratio < t:
                continue
            excess = (ratio - t) / max(t, 0.1)
            if excess >= pac.PAC_EXCESS_THRESHOLD:
                early_candidates[cls] = excess
        if early_candidates:
            early_stop_cls = max(early_candidates, key=early_candidates.get)
            early_stop_step = step
            early_stop_excess = early_candidates[early_stop_cls]
            pred = early_stop_cls
            break
    else:
        candidates = {}
        for cls in disease_cls:
            score = class_trajs[cls][-1] if class_trajs[cls] else 1.0
            t = class_thresholds[cls]
            if score < t:
                continue
            candidates[cls] = (score - t) / max(t, 0.1)
        pred = max(candidates, key=candidates.get) if candidates else pac.HEALTHY

    meta = {
        "class_trajs": {int(c): list(t) for c, t in class_trajs.items()},
        "class_thresholds": {int(c): float(t) for c, t in class_thresholds.items()},
        "early_stop": early_stop_step is not None,
        "early_stop_step": early_stop_step,
        "early_stop_cls": int(early_stop_cls) if early_stop_cls else None,
        "early_stop_excess": float(early_stop_excess) if early_stop_excess else None,
        "max_steps": max_steps,
    }
    return int(pred), meta


def main():
    print("Loading data...")
    data, labels = pac.load_data()
    is_bin = pac.classify_features(data)
    print(f"  {data.shape[0]} patients, {data.shape[1]} features")

    best = None

    for seed in SEEDS:
        print(f"\nSeed {seed}: running 10-fold CV...")
        n = data.shape[0]
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        folds = np.array_split(idx, 10)
        all_cls = sorted(set(labels))

        for fi in range(10):
            test_idx = folds[fi]
            train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
            td, tl = data[train_idx], labels[train_idx]
            nodes = pac.build_tree(td, tl, is_bin)
            train_result = pac.train(nodes, td, tl, is_bin, all_cls)

            for uid in test_idx:
                true_cls = int(labels[uid])
                pred, meta = instrumented_predict(uid, data, nodes, all_cls, train_result)
                if not meta["early_stop"]:
                    continue
                if pred != true_cls:
                    continue
                n_cls_with_traj = sum(1 for t in meta["class_trajs"].values() if len(t) >= 5)
                stop_step = meta["early_stop_step"]
                if n_cls_with_traj >= 2 and stop_step >= 3:
                    score = n_cls_with_traj * 100 + stop_step
                    if best is None or score > best["score"]:
                        best = {
                            "seed": seed, "uid": int(uid),
                            "true_cls": true_cls, "pred_cls": pred,
                            "meta": meta, "score": score,
                        }

        if best and best["score"] >= 300:
            print(f"  Found excellent example (score={best['score']}), stopping search.")
            break

    if best is None:
        print("ERROR: No suitable early-stop patient found!")
        return

    print(f"\nBest example: Patient {best['uid']}, True={best['true_cls']}, "
          f"Pred={best['pred_cls']}, Seed={best['seed']}, "
          f"Stop step={best['meta']['early_stop_step']+1}/{best['meta']['max_steps']}")

    # ── Generate Figure 8 ──
    meta = best["meta"]
    trajs = meta["class_trajs"]
    thresholds = meta["class_thresholds"]
    stop_step = meta["early_stop_step"]
    stop_cls = meta["early_stop_cls"]

    cls_names = {1: "Normal", 2: "Ischemic", 3: "Old Ant. MI", 4: "Old Inf. MI",
                 5: "Sinus Tachy", 6: "Sinus Brady", 9: "LBBB", 10: "RBBB"}

    # Select classes: early-stop class first, then 2 others with longest trajectories
    traj_lengths = {c: len(t) for c, t in trajs.items() if len(t) > 0}
    sorted_cls = sorted(traj_lengths, key=traj_lengths.get, reverse=True)
    plot_classes = []
    if stop_cls in sorted_cls:
        plot_classes.append(stop_cls)
        sorted_cls.remove(stop_cls)
    for c in sorted_cls:
        if len(plot_classes) >= 3:
            break
        plot_classes.append(c)

    colors = {plot_classes[0]: "#d62728"}
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    for i, c in enumerate(plot_classes[1:]):
        colors[c] = palette[i % len(palette)]

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 11, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'figure.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))

    def get_traj(cls):
        return trajs[str(cls)] if str(cls) in trajs else trajs[cls]

    def get_thresh(cls):
        return thresholds[str(cls)] if str(cls) in thresholds else thresholds[cls]

    for cls in plot_classes:
        traj = get_traj(cls)
        label = cls_names.get(cls, f"Class {cls}")
        lw = 2.0 if cls == stop_cls else 1.2
        ls = '-' if cls == stop_cls else '--'
        if cls == stop_cls:
            # Truncate at early-stop point — evaluation terminates here
            traj_plot = traj[:stop_step + 1]
            steps = np.arange(1, len(traj_plot) + 1)
            ax.plot(steps, traj_plot, ls, color=colors[cls], linewidth=lw,
                    label=f"Class {cls} ({label})", zorder=3)
        else:
            steps = np.arange(1, len(traj) + 1)
            ax.plot(steps, traj, ls, color=colors[cls], linewidth=lw,
                    label=f"Class {cls} ({label})", zorder=2)

    # Threshold lines — place labels at left side to avoid right-edge overlap
    for cls in plot_classes:
        t = get_thresh(cls)
        ax.axhline(y=t, color=colors[cls], linestyle=':', alpha=0.5, linewidth=0.8)

    # Single combined threshold annotation at left
    thresh_lines = []
    for cls in plot_classes:
        t = get_thresh(cls)
        name = cls_names.get(cls, f"Class {cls}")
        thresh_lines.append(f"$T_{{eff}}^{{({cls})}}$={t:.1f}")
    thresh_text = "   ".join(thresh_lines)
    # Place just above the highest threshold line
    max_thresh = max(get_thresh(c) for c in plot_classes)
    ax.text(1, max_thresh + 1.0, thresh_text,
            fontsize=7.5, color='#555555', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

    # Early-stop marker
    stop_traj = get_traj(stop_cls)
    stop_ratio = stop_traj[stop_step]
    ax.plot(stop_step + 1, stop_ratio, 'o', color=colors[stop_cls], markersize=10,
            markeredgecolor='black', markeredgewidth=1.2, zorder=5)

    # Vertical dashed line at early-stop step
    ax.axvline(x=stop_step + 1, color='black', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.annotate(f"Early stop (step {stop_step+1}/{meta['max_steps']}, $R_c$={stop_ratio:.1f})",
                xy=(stop_step + 1, stop_ratio),
                xytext=(stop_step - 8, stop_ratio * 0.6),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9))

    # PAC firing level for the stop class
    t_stop = get_thresh(stop_cls)
    firing_level = t_stop + pac.PAC_EXCESS_THRESHOLD * max(t_stop, 0.1)
    ax.axhline(y=firing_level, color='gray', linestyle='-.', alpha=0.5, linewidth=1.0)
    ax.text(stop_step - 2, firing_level + 0.8,
            f"PAC firing level",
            fontsize=7, color='gray', alpha=0.7)

    ax.set_xlabel("Feature Evaluation Step $k$ (Fisher-ordered)")
    ax.set_ylabel("Evidence Ratio $R_c(k)$")
    ax.set_title(
        f"FIGURE 8: PAC Evidence Ratio Trajectory — "
        f"Patient {best['uid']} (True: {cls_names.get(best['true_cls'], best['true_cls'])})",
        fontweight='bold', fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='none')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(left=0.5, right=stop_step + 4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "pac_trajectory_fig8.png")
    plt.savefig(fig_path, format='png', bbox_inches='tight')
    print(f"\nFigure 8 saved to {fig_path}")
    plt.close()

    # Save data for reproducibility
    fig_data = {
        "patient_uid": best["uid"],
        "true_class": best["true_cls"],
        "predicted_class": best["pred_cls"],
        "seed": best["seed"],
        "early_stop_step": stop_step + 1,
        "max_steps": meta["max_steps"],
        "plotted_classes": plot_classes,
        "trajectories": {str(c): (trajs[str(c)] if str(c) in trajs else trajs[c]) for c in plot_classes},
        "thresholds": {str(c): (thresholds[str(c)] if str(c) in thresholds else thresholds[c]) for c in plot_classes},
    }
    with open(os.path.join(OUT_DIR, "pac_fig8_data.json"), "w") as f:
        json.dump(fig_data, f, indent=2)
    print("Figure 8 data saved.")


if __name__ == "__main__":
    main()
