import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from Algorithm4 import main as run_algorithm4_pipeline, HealthDecision


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def confusion_matrix_info(records, gender_mask, gender_name):
    labels = [HealthDecision.HEALTHY, HealthDecision.UNHEALTHY, HealthDecision.SCREENING]
    label_names = [d.value for d in labels]
    cm = np.zeros((3, 3), dtype=int)
    for r, is_gender in zip(records, gender_mask):
        if not is_gender:
            continue
        true_idx = 0 if r.true_is_healthy else 1
        if r.decision == HealthDecision.SCREENING:
            pred_idx = 2
        elif r.decision == HealthDecision.UNHEALTHY:
            pred_idx = 1
        else:
            pred_idx = 0
        cm[true_idx, pred_idx] += 1
    return cm, label_names


def plot_confusion_matrix(cm, labels, title, filename):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:d}', ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=14)

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_female_error_patterns(records, output_dir):
    miscounts = Counter()
    total_by_class = Counter()
    for r in records:
        if r.true_is_healthy:
            continue
        if r.user_global_idx is None:
            continue
        # gender is available from Algorithm 4 output via data in the record's global user index;
        # we will delegate gender filtering into the caller using female records only.
        total_by_class[r.true_label] += 1
        if not r.is_correct:
            miscounts[r.true_label] += 1

    classes = sorted(miscounts.keys())
    counts = [miscounts[c] for c in classes]
    total = [total_by_class[c] for c in classes]
    error_rate = [miscounts[c] / total_by_class[c] if total_by_class[c] else 0.0 for c in classes]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    bar_width = 0.4
    x = np.arange(len(classes))
    ax1.bar(x - bar_width/2, counts, bar_width, label='Misclassified females', color='#d62728')
    ax1.bar(x + bar_width/2, total, bar_width, label='Total diseased females', color='#1f77b4', alpha=0.75)
    ax1.set_xlabel('True disease class')
    ax1.set_ylabel('Count')
    ax1.set_title('Female diseased misclassification patterns by true arrhythmia class')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in classes], rotation=45)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(x, error_rate, color='black', marker='o', linestyle='--', label='Error rate')
    ax2.set_ylabel('Error rate')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    filename = os.path.join(output_dir, 'female_misclassification_by_class.png')
    fig.savefig(filename, dpi=300)
    plt.close(fig)

    return filename


def filter_records_by_gender(records, data, gender_value):
    return [r for r in records if data[r.user_global_idx, 1] == gender_value]


def main():
    parser = argparse.ArgumentParser(description='Plot Algorithm 4 gender confusion matrices and female error patterns.')
    parser.add_argument('--data-path', default='arrhythmia.data', help='Path to arrhythmia.data')
    parser.add_argument('--max-users', type=int, default=None, help='Maximum users to process for LOOCV')
    parser.add_argument('--output-dir', default='graphs', help='Directory to save plots')
    parser.add_argument('--skip-run', action='store_true', help='Skip running pipeline and only plot if output exists')
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)

    if args.skip_run:
        print('Skipping pipeline run; no data available without running the pipeline.')
        return

    output = run_algorithm4_pipeline(
        data_path=args.data_path,
        run_loocv_flag=True,
        max_users=args.max_users,
        run_tests=False,
        fidelity=False,
        rng_seed=42,
    )

    if output.data is None:
        raise RuntimeError('Algorithm4 output did not preserve input data for plotting.')

    records = output.records
    male_records = filter_records_by_gender(records, output.data, 0)
    female_records = filter_records_by_gender(records, output.data, 1)

    cm_male, labels = confusion_matrix_info(records, [output.data[r.user_global_idx, 1] == 0 for r in records], 'Male')
    cm_female, _ = confusion_matrix_info(records, [output.data[r.user_global_idx, 1] == 1 for r in records], 'Female')

    plot_confusion_matrix(cm_male, labels,
                          'Male Confusion Matrix (Healthy / Unhealthy / Screening)',
                          os.path.join(args.output_dir, 'confusion_matrix_male.png'))
    plot_confusion_matrix(cm_female, labels,
                          'Female Confusion Matrix (Healthy / Unhealthy / Screening)',
                          os.path.join(args.output_dir, 'confusion_matrix_female.png'))

    female_diseased_records = [r for r in female_records if r.true_is_diseased]
    if female_diseased_records:
        plot_female_error_patterns(female_diseased_records, args.output_dir)

    print('Saved graphs to', os.path.abspath(args.output_dir))


if __name__ == '__main__':
    main()
