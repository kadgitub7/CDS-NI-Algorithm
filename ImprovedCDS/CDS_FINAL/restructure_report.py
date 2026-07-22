"""Restructure CDS_Overall_Report.docx into a properly organized final report.

Fixes:
- Triple-duplicate section numbering (adds Part divider headings)
- Redundant tables (adds cross-reference annotations)
- Missing file location references
- Em/en dash removal
- Appends comprehensive appendix sections (A-G) with all available data
"""
import os, sys, json, shutil, re
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = {
    1: 'Normal', 2: 'Ischemic Changes (CAD)', 3: 'Old Anterior MI',
    4: 'Old Inferior MI', 5: 'Sinus Tachycardia', 6: 'Sinus Bradycardia',
    7: 'WPW', 8: 'AV Block', 9: 'LBBB', 10: 'RBBB',
    11: 'Class 14', 12: 'Class 15', 13: 'Class 16'
}


# ============================================================
# XML Helpers for inserting/modifying elements
# ============================================================

def make_paragraph_element(text, style=None, bold=False, italic=False,
                           font_size=None, color=None, alignment=None):
    """Create a new w:p XML element with the given properties."""
    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    if style:
        pStyle = OxmlElement('w:pStyle')
        pStyle.set(qn('w:val'), style)
        pPr.append(pStyle)
    if alignment:
        jc = OxmlElement('w:jc')
        jc.set(qn('w:val'), alignment)
        pPr.append(jc)
    p.append(pPr)

    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    if bold:
        rPr.append(OxmlElement('w:b'))
    if italic:
        rPr.append(OxmlElement('w:i'))
    if font_size:
        sz = OxmlElement('w:sz')
        sz.set(qn('w:val'), str(font_size * 2))
        rPr.append(sz)
        szCs = OxmlElement('w:szCs')
        szCs.set(qn('w:val'), str(font_size * 2))
        rPr.append(szCs)
    if color:
        c = OxmlElement('w:color')
        c.set(qn('w:val'), color)
        rPr.append(c)
    r.append(rPr)
    t = OxmlElement('w:t')
    t.text = text
    t.set(qn('xml:space'), 'preserve')
    r.append(t)
    p.append(r)
    return p


def insert_before(target_element, new_element):
    target_element.addprevious(new_element)


def insert_after(target_element, new_element):
    target_element.addnext(new_element)


def make_page_break():
    p = OxmlElement('w:p')
    r = OxmlElement('w:r')
    br = OxmlElement('w:br')
    br.set(qn('w:type'), 'page')
    r.append(br)
    p.append(r)
    return p


# ============================================================
# Content Helpers
# ============================================================

def replace_em_dashes(doc):
    count = 0
    for p in doc.paragraphs:
        for run in p.runs:
            if '—' in run.text:
                run.text = run.text.replace('—', '-')
                count += 1
            if '–' in run.text:
                run.text = run.text.replace('–', '-')
                count += 1
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        if '—' in run.text:
                            run.text = run.text.replace('—', '-')
                            count += 1
                        if '–' in run.text:
                            run.text = run.text.replace('–', '-')
                            count += 1
    return count


def get_para_text(para):
    return para.text.strip()


def find_paragraph_index(doc, text_substring, start=0):
    for i in range(start, len(doc.paragraphs)):
        if text_substring in get_para_text(doc.paragraphs[i]):
            return i
    return -1


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(text, style=f'Heading {level}')
    return p


def add_para(doc, text, bold=False, italic=False, font_size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(font_size)
    return p


def add_source_ref(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    return p


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(10)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)
    return table


# ============================================================
# Data Loading
# ============================================================

def load_loocv_traces():
    base_path = os.path.join(SRC_DIR, '..', 'output', 'loocv_trace.json')
    ovr_path = os.path.join(SRC_DIR, '..', 'output', 'loocv_trace_ovr.json')
    with open(base_path) as f:
        base = json.load(f)
    with open(ovr_path) as f:
        ovr = json.load(f)
    return base, ovr


def compute_loocv_stats(trace):
    n = len(trace)
    correct = sum(1 for r in trace if r['correct'])
    cls_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in trace:
        tc = r['true_class']
        cls_stats[tc]['total'] += 1
        if r['correct']:
            cls_stats[tc]['correct'] += 1

    healthy = [r for r in trace if r['true_class'] == 1]
    diseased = [r for r in trace if r['true_class'] != 1]
    spec = sum(1 for r in healthy if r['correct']) / len(healthy) if healthy else 0
    sens = sum(1 for r in diseased if r['predicted'] != 1) / len(diseased) if diseased else 0
    bin_correct = 0
    for r in trace:
        pred = r.get('predicted', '')
        true_h = (r['true_class'] == 1)
        pred_h = (pred == 1 or pred == 'HEALTHY')
        if true_h == pred_h:
            bin_correct += 1
    ba_vals = []
    for cls in cls_stats:
        s = cls_stats[cls]
        if s['total'] > 0:
            ba_vals.append(s['correct'] / s['total'])
    ba = sum(ba_vals) / len(ba_vals) if ba_vals else 0
    return {
        'n': n, 'correct': correct, 'accuracy': correct / n,
        'specificity': spec, 'sensitivity': sens,
        'binary_accuracy': bin_correct / n,
        'balanced_accuracy': ba,
        'per_class': dict(cls_stats),
    }


def load_results_all_splits():
    path = os.path.join(SRC_DIR, 'results_all_splits.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        # Fix floating-point key issues (90/9 should be 90/10, 80/19 should be 80/20)
        key_fixes = {'90/9': '90/10', '80/19': '80/20'}
        for old_key, new_key in key_fixes.items():
            if old_key in data and new_key not in data:
                data[new_key] = data.pop(old_key)
        return data
    return None


def load_comprehensive_results():
    # Try combined file first
    path = os.path.join(SRC_DIR, 'comprehensive_results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    # Fall back to individual result files from parallel tasks
    combined = {}
    files = {
        "auc_roc": "results_auc.json",
        "stratified": "results_stratified.json",
        "computational_cost": "results_cost.json",
        "confusion_matrix_seed13": "results_confusion.json",
    }
    for key, fname in files.items():
        fpath = os.path.join(SRC_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                combined[key] = json.load(f)
    return combined if combined else None


# ============================================================
# Phase 1: Structural fixes to existing content
# ============================================================

def fix_structure(doc):
    """Add Part divider headings and file location annotations."""
    print("  Scanning paragraphs for structural boundaries...")

    part2_idx = -1
    part3_idx = -1

    for i, p in enumerate(doc.paragraphs):
        txt = get_para_text(p)
        if 'Incremental Analysis of CDS Arrhythmia Classification Algorithm' in txt:
            part2_idx = i
        if 'Comparative Algorithmic Analysis' in txt and 'CDS-OVR vs' in txt:
            part3_idx = i
        if part3_idx == -1 and txt == 'CDS-OVR vs. Benchmark Methods':
            part3_idx = i

    # Also find the start paragraph of Part 1
    part1_idx = -1
    for i, p in enumerate(doc.paragraphs):
        txt = get_para_text(p)
        if 'CDS-OVR Algorithm Specification' in txt and i < 5:
            part1_idx = i
            break

    changes = 0

    # Part I title
    if part1_idx >= 0:
        p = doc.paragraphs[part1_idx]
        # Change style to Title-like heading
        p.style = doc.styles['Heading 1']
        for run in p.runs:
            run.text = 'Part I: ' + run.text
        changes += 1
        print(f"    Part I header set at paragraph {part1_idx}")

    # Part II divider
    if part2_idx >= 0:
        p = doc.paragraphs[part2_idx]
        # Insert page break and Part II heading before this paragraph
        pb = make_page_break()
        insert_before(p._element, pb)

        part_heading = make_paragraph_element(
            'Part II: Incremental Development Analysis',
            style='Heading1', bold=True, font_size=16
        )
        insert_before(p._element, part_heading)

        # Change the original paragraph to be a subtitle
        p.style = doc.styles['Heading 2']
        changes += 1
        print(f"    Part II header inserted at paragraph {part2_idx}")

    # Part III divider
    if part3_idx >= 0:
        p = doc.paragraphs[part3_idx]
        pb = make_page_break()
        insert_before(p._element, pb)

        part_heading = make_paragraph_element(
            'Part III: Benchmark Comparison',
            style='Heading1', bold=True, font_size=16
        )
        insert_before(p._element, part_heading)

        p.style = doc.styles['Heading 2']
        changes += 1
        print(f"    Part III header inserted at paragraph {part3_idx}")

    return changes


def add_file_location_refs(doc):
    """Add source file annotations after relevant sections."""
    refs_to_add = [
        ('Table 1: Class Distribution', 'Source: ImprovedCDS/CDS_FINAL/cds_ovr.py, lines 31-38 (REMOVE_CLASSES, class filtering)'),
        ('Table 2: Complete Hyperparameter', 'Source: ImprovedCDS/CDS_FINAL/cds_ovr.py, lines 17-44 (constants section)'),
        ('Algorithm 1: build_tree', 'Source: ImprovedCDS/CDS_FINAL/cds_ovr.py, lines 55-116 (build_tree function)'),
        ('supervised chi-squared binning', 'Source: ImprovedCDS/CDS_FINAL/cds_ovr.py, lines 209-265 (chi2_binning function)'),
        ('Algorithm 8: Complete CDS-OVR Pipeline', 'Source: ImprovedCDS/CDS_FINAL/cds_ovr.py (full algorithm implementation, 587 lines)'),
        ('evidence_analysis.py', 'Source file: ImprovedCDS/CDS_FINAL/evidence_analysis.py'),
        ('evidence_ablation.py', 'Source file: ImprovedCDS/CDS_FINAL/evidence_ablation.py'),
        ('ablation_full.py', 'Source file: ImprovedCDS/CDS_FINAL/ablation_full.py'),
        ('run_all_splits.py', 'Source file: ImprovedCDS/CDS_FINAL/run_all_splits.py; Results: ImprovedCDS/CDS_FINAL/results_all_splits.json'),
        ('00_base_original', 'Source: ImprovedCDS/cds.py (original base algorithm, 491 lines)'),
        ('Per-Class Score Distributions and Threshold Rationale', 'Threshold values: ImprovedCDS/CDS_FINAL/cds_ovr.py, line 39 (CLASS_THRESHOLDS)'),
    ]

    count = 0
    inserted_refs = set()

    for i, p in enumerate(doc.paragraphs):
        txt = get_para_text(p)
        for trigger, ref_text in refs_to_add:
            if trigger in txt and trigger not in inserted_refs:
                annotation = make_paragraph_element(
                    ref_text, italic=True, font_size=9, color='666666'
                )
                insert_after(p._element, annotation)
                inserted_refs.add(trigger)
                count += 1
                break

    return count


def mark_redundancies(doc):
    """Add cross-reference notes near redundant tables/content."""
    redundancy_markers = [
        ('Table 6: Complete Hyperparameter Reference',
         '[Note: This table consolidates the hyperparameters from Table 2 in Part I, Section 2.5. Both tables are retained for contextual reference within their respective sections.]'),
    ]

    count = 0
    for i, p in enumerate(doc.paragraphs):
        txt = get_para_text(p)
        for trigger, note in redundancy_markers:
            if trigger in txt:
                annotation = make_paragraph_element(
                    note, italic=True, font_size=9, color='888888'
                )
                insert_after(p._element, annotation)
                count += 1
                break
    return count


# ============================================================
# Phase 2: Append comprehensive appendix sections
# ============================================================

def build_appendices(doc):
    """Build and append all appendix sections."""
    print("  Loading data for appendices...")

    base_trace, ovr_trace = load_loocv_traces()
    base_stats = compute_loocv_stats(base_trace)
    ovr_stats = compute_loocv_stats(ovr_trace)
    all_splits = load_results_all_splits()
    comp_results = load_comprehensive_results()

    doc.add_page_break()

    # ================================================================
    # APPENDICES HEADER
    # ================================================================
    add_heading(doc, 'Appendices', 1)
    add_para(doc,
        'The following appendices provide comprehensive evaluation results, '
        'mathematical formulations, and references for the CDS-OVR algorithm. '
        'All results were generated from the source code in ImprovedCDS/CDS_FINAL/.')

    # ================================================================
    # APPENDIX A: LOOCV Results
    # ================================================================
    doc.add_page_break()
    add_heading(doc, 'Appendix A: Leave-One-Out Cross-Validation (LOOCV) Results', 1)

    add_para(doc,
        'LOOCV provides the least biased estimate of generalization performance by training on '
        'N-1 patients and testing on the remaining one, repeated for all N patients. Unlike k-fold '
        'CV, LOOCV uses the maximum possible training set size for each prediction, eliminating '
        'variance from random fold assignments.')
    add_source_ref(doc, 'Source: ImprovedCDS/output/loocv_trace.json (base CDS), ImprovedCDS/output/loocv_trace_ovr.json (CDS-OVR)')

    # A.1 Base CDS
    add_heading(doc, 'A.1 Base CDS Algorithm LOOCV (All 13 Classes, 452 Patients)', 2)
    add_para(doc,
        'The original CDS algorithm (cds.py) was evaluated using LOOCV on the complete 452-patient '
        'dataset with all 13 arrhythmia classes. This represents the unmodified binary classifier '
        'from the initial implementation.')
    add_source_ref(doc, 'Source: ImprovedCDS/cds.py (run_loocv function); Trace: ImprovedCDS/output/loocv_trace.json')

    add_para(doc, f'Overall Multiclass Accuracy: {base_stats["correct"]}/{base_stats["n"]} = {100*base_stats["accuracy"]:.1f}%', bold=True)
    add_para(doc, f'Binary Accuracy (Healthy vs Disease): {100*base_stats["binary_accuracy"]:.1f}%', bold=True)
    add_para(doc, f'Specificity (Healthy correctly identified): {100*base_stats["specificity"]:.1f}%')
    add_para(doc, f'Sensitivity (Disease detected as non-healthy): {100*base_stats["sensitivity"]:.1f}%')
    add_para(doc, f'Balanced Accuracy (mean per-class): {100*base_stats["balanced_accuracy"]:.1f}%')

    base_classes = sorted(base_stats['per_class'].keys())
    rows = []
    for cls in base_classes:
        s = base_stats['per_class'][cls]
        acc = f'{100*s["correct"]/s["total"]:.1f}%' if s['total'] > 0 else '0.0%'
        # Binary: class 1 patients predicted as 1 = correct; disease patients predicted != 1 = correct
        bin_correct = s['correct'] if cls == 1 else sum(
            1 for r in base_trace if r['true_class'] == cls and r.get('predicted', -1) != 1)
        bin_acc = f'{100*bin_correct/s["total"]:.1f}%' if s['total'] > 0 else '0.0%'
        rows.append([str(cls), CLASS_NAMES.get(cls, f'Class {cls}'),
                      str(s['total']), str(s['correct']), acc, bin_acc])
    add_para(doc, 'Table A1: Base CDS LOOCV Per-Class Results', bold=True)
    add_table(doc, ['Class', 'Description', 'N', 'Correct (Multi)', 'Multi Acc', 'Binary Acc'], rows)

    add_para(doc,
        'The base CDS algorithm achieves high specificity (94.7%) but poor sensitivity (51.7%). '
        'It correctly identifies most healthy patients but fails to detect disease subtypes. '
        'Classes 5, 6, 7, 8, 11, 12, and 13 have 0% accuracy, confirming the binary framework '
        'cannot distinguish between specific arrhythmia types.')

    # A.2 CDS-OVR
    add_heading(doc, 'A.2 CDS-OVR Algorithm LOOCV (All 13 Classes, 452 Patients)', 2)
    add_para(doc,
        'The CDS-OVR algorithm (cds_ovr.py) was evaluated using LOOCV on the same 452-patient '
        'dataset. Note that the LOOCV evaluation includes all 13 original classes, not the 8-class '
        'filtered dataset used for other evaluation protocols. This tests the algorithm on classes '
        'it was not designed to handle (7, 8, 11, 12, 13).')
    add_source_ref(doc, 'Source: ImprovedCDS/CDS_FINAL/cds_ovr.py; Trace: ImprovedCDS/output/loocv_trace_ovr.json')

    add_para(doc, f'Overall Multiclass Accuracy: {ovr_stats["correct"]}/{ovr_stats["n"]} = {100*ovr_stats["accuracy"]:.1f}%', bold=True)
    add_para(doc, f'Binary Accuracy (Healthy vs Disease): {100*ovr_stats["binary_accuracy"]:.1f}%', bold=True)
    add_para(doc, f'Specificity: {100*ovr_stats["specificity"]:.1f}%')
    add_para(doc, f'Sensitivity: {100*ovr_stats["sensitivity"]:.1f}%')
    add_para(doc, f'Balanced Accuracy: {100*ovr_stats["balanced_accuracy"]:.1f}%')

    ovr_classes = sorted(ovr_stats['per_class'].keys())
    rows = []
    for cls in ovr_classes:
        s = ovr_stats['per_class'][cls]
        acc = f'{100*s["correct"]/s["total"]:.1f}%' if s['total'] > 0 else '0.0%'
        bin_correct = s['correct'] if cls == 1 else sum(
            1 for r in ovr_trace if r['true_class'] == cls and r.get('predicted', -1) != 1)
        bin_acc = f'{100*bin_correct/s["total"]:.1f}%' if s['total'] > 0 else '0.0%'
        rows.append([str(cls), CLASS_NAMES.get(cls, f'Class {cls}'),
                      str(s['total']), str(s['correct']), acc, bin_acc])
    add_para(doc, 'Table A2: CDS-OVR LOOCV Per-Class Results (All 13 Classes)', bold=True)
    add_table(doc, ['Class', 'Description', 'N', 'Correct (Multi)', 'Multi Acc', 'Binary Acc'], rows)

    add_para(doc,
        f'CDS-OVR improves overall accuracy from {100*base_stats["accuracy"]:.1f}% to '
        f'{100*ovr_stats["accuracy"]:.1f}% (+{100*(ovr_stats["accuracy"]-base_stats["accuracy"]):.1f} pp) '
        f'and dramatically improves sensitivity from {100*base_stats["sensitivity"]:.1f}% to '
        f'{100*ovr_stats["sensitivity"]:.1f}% '
        f'(+{100*(ovr_stats["sensitivity"]-base_stats["sensitivity"]):.1f} pp). '
        f'The number of classes detected (accuracy > 0%) increases from 5 to 10 of 13.')

    # A.3 Comparison Summary
    add_heading(doc, 'A.3 LOOCV Comparison Summary', 2)
    add_para(doc, 'Table A3: LOOCV Comparison - Base CDS vs. CDS-OVR', bold=True)
    add_table(doc, ['Metric', 'Base CDS', 'CDS-OVR', 'Improvement'],
              [['Dataset', '452 patients, 13 classes', '452 patients, 13 classes', '-'],
               ['Overall Accuracy', f'{100*base_stats["accuracy"]:.1f}%', f'{100*ovr_stats["accuracy"]:.1f}%',
                f'+{100*(ovr_stats["accuracy"]-base_stats["accuracy"]):.1f} pp'],
               ['Binary Accuracy', f'{100*base_stats["binary_accuracy"]:.1f}%',
                f'{100*ovr_stats["binary_accuracy"]:.1f}%',
                f'+{100*(ovr_stats["binary_accuracy"]-base_stats["binary_accuracy"]):.1f} pp'],
               ['Specificity', f'{100*base_stats["specificity"]:.1f}%', f'{100*ovr_stats["specificity"]:.1f}%',
                f'{100*(ovr_stats["specificity"]-base_stats["specificity"]):+.1f} pp'],
               ['Sensitivity', f'{100*base_stats["sensitivity"]:.1f}%', f'{100*ovr_stats["sensitivity"]:.1f}%',
                f'+{100*(ovr_stats["sensitivity"]-base_stats["sensitivity"]):.1f} pp'],
               ['Balanced Accuracy', f'{100*base_stats["balanced_accuracy"]:.1f}%',
                f'{100*ovr_stats["balanced_accuracy"]:.1f}%',
                f'+{100*(ovr_stats["balanced_accuracy"]-base_stats["balanced_accuracy"]):.1f} pp'],
               ['Classes Detected (>0%)', '5 of 13', '10 of 13', '+5 classes'],
               ['Decision Type', 'Binary (Healthy/Unhealthy)', 'Multiclass (8 classes)', 'Multiclass capability']])

    # ================================================================
    # APPENDIX B: AUC/ROC
    # ================================================================
    doc.add_page_break()
    add_heading(doc, 'Appendix B: AUC/ROC Analysis', 1)

    add_para(doc,
        'The Area Under the Receiver Operating Characteristic Curve (AUC-ROC) provides a '
        'threshold-independent measure of discriminative ability. Unlike accuracy, AUC evaluates '
        'performance across all possible decision thresholds, making it robust to class imbalance '
        'and threshold selection.')

    add_heading(doc, 'B.1 Binary AUC-ROC Computation', 2)
    add_para(doc,
        'For binary AUC, the disease score for each patient is computed as the maximum disease-class '
        'ratio score divided by the healthy-class score plus epsilon:')
    add_para(doc, '  disease_score = max(R_c for c != 1) / (R_1 + 0.1)')
    add_para(doc,
        'The ROC curve is then constructed by varying the threshold on this composite score. The '
        'AUC is computed via the trapezoidal rule over sorted (FPR, TPR) pairs.')

    add_heading(doc, 'B.2 Multiclass AUC-ROC (One-vs-Rest)', 2)
    add_para(doc,
        'Per-class AUC is computed using the One-vs-Rest approach: for each class c, the ratio score '
        'R_c serves as the discriminant, and a binary label indicates whether the patient belongs to '
        'class c. Macro AUC is the unweighted mean across classes; weighted AUC weights each class '
        'by its support (number of patients).')

    if comp_results and 'auc_roc' in comp_results:
        auc = comp_results['auc_roc']
        add_source_ref(doc, 'Source: ImprovedCDS/CDS_FINAL/comprehensive_experiments.py; Results: ImprovedCDS/CDS_FINAL/comprehensive_results.json')

        rows = []
        for protocol, label in [('10fold', '10-Fold CV'), ('90_10', '90/10 Split'), ('60_40', '60/40 Split')]:
            if protocol in auc:
                d = auc[protocol]
                rows.append([label,
                             f'{d["binary_auc_mean"]:.4f} +/- {d["binary_auc_std"]:.4f}',
                             f'{d["macro_auc_mean"]:.4f} +/- {d["macro_auc_std"]:.4f}',
                             f'{d["weighted_auc_mean"]:.4f} +/- {d["weighted_auc_std"]:.4f}'])

        if rows:
            add_para(doc, 'Table B1: AUC-ROC Values Across Evaluation Protocols (10 seeds)', bold=True)
            add_table(doc, ['Protocol', 'Binary AUC', 'Macro AUC (OVR)', 'Weighted AUC (OVR)'], rows)

        add_heading(doc, 'B.3 Per-Class AUC-ROC', 2)
        if '10fold' in auc and 'per_class_auc_mean' in auc['10fold']:
            pc = auc['10fold']['per_class_auc_mean']
            rows = []
            for cls_str in sorted(pc.keys(), key=lambda x: int(x)):
                cls = int(cls_str)
                rows.append([str(cls), CLASS_NAMES.get(cls, f'Class {cls}'),
                             f'{pc[cls_str]:.4f}'])
            add_para(doc, 'Table B2: Per-Class AUC-ROC (10-Fold CV, mean over 10 seeds)', bold=True)
            add_table(doc, ['Class', 'Description', 'Mean AUC'], rows)
    else:
        add_para(doc,
            '[AUC/ROC results pending - comprehensive_experiments.py has not yet completed. '
            'Re-run this script after the experiment finishes to include these results.]',
            italic=True)

    # ================================================================
    # APPENDIX C: Comprehensive Performance Results
    # ================================================================
    doc.add_page_break()
    add_heading(doc, 'Appendix C: Comprehensive Performance Results', 1)

    add_para(doc,
        'This appendix consolidates all evaluation results across protocols and seeds. '
        'All results use the 416-patient, 8-class filtered dataset unless otherwise noted.')
    add_source_ref(doc,
        'Source: ImprovedCDS/CDS_FINAL/run_all_splits.py; '
        'Results: ImprovedCDS/CDS_FINAL/results_all_splits.json')

    # C.1 10-Fold CV
    add_heading(doc, 'C.1 10-Fold Cross-Validation (10 Seeds)', 2)
    if all_splits and '10-fold CV' in all_splits:
        cv = all_splits['10-fold CV']
        s = cv['summary']
        add_para(doc, f'Mean Multiclass Accuracy: {s["multi_mean"]}% (std = {s["multi_std"]}%)', bold=True)
        add_para(doc, f'Best Multiclass Accuracy: {s["multi_best"]}% (seed 13)')
        add_para(doc, f'Worst Multiclass Accuracy: {s["multi_worst"]}%')
        add_para(doc, f'Mean Binary Accuracy: {s["binary_mean"]}% (std = {s["binary_std"]}%)')
        add_para(doc, f'Best Binary Accuracy: {s["binary_best"]}%')

        rows = []
        for sd in cv['seeds']:
            rows.append([str(sd['seed']),
                         f'{sd["multiclass_acc"]}%', f'{sd["binary_acc"]}%',
                         f'{sd["specificity"]}%', f'{sd["sensitivity"]}%',
                         f'{sd["balanced_acc"]}%'])
        add_para(doc, 'Table C1: 10-Fold CV Results per Seed', bold=True)
        add_table(doc, ['Seed', 'Multi Acc', 'Binary Acc', 'Specificity', 'Sensitivity', 'Balanced Acc'], rows)

    # C.2 All Split Ratios
    add_heading(doc, 'C.2 Train/Test Split Results Across All Ratios', 2)
    add_para(doc,
        'CDS-OVR was evaluated across 7 split ratios (50/50 through 90/10), each with 10 random '
        'seeds. This sweep reveals how accuracy depends on training data volume.')

    if all_splits:
        summary_rows = []
        for ratio in ['50/50', '60/40', '70/30', '75/25', '80/20', '85/15', '90/10']:
            if ratio in all_splits:
                s = all_splits[ratio]['summary']
                summary_rows.append([ratio,
                    f'{s["multi_mean"]}%', f'{s["multi_std"]}%',
                    f'{s["multi_best"]}%', f'{s["multi_worst"]}%',
                    f'{s["binary_mean"]}%', f'{s["binary_std"]}%'])

        add_para(doc, 'Table C2: Summary Across All Split Ratios (10 seeds each)', bold=True)
        add_table(doc,
            ['Split', 'Multi Mean', 'Multi Std', 'Multi Best', 'Multi Worst', 'Binary Mean', 'Binary Std'],
            summary_rows)

    # C.3 90/10 Detailed
    add_heading(doc, 'C.3 90/10 Split Detailed Results', 2)
    if all_splits and '90/10' in all_splits:
        s90 = all_splits['90/10']
        s = s90['summary']
        add_para(doc, f'Mean Multiclass Accuracy: {s["multi_mean"]}% (std = {s["multi_std"]}%)', bold=True)
        add_para(doc, f'Best Multiclass Accuracy: {s["multi_best"]}% (seed 76)')
        add_para(doc, f'Mean Binary Accuracy: {s["binary_mean"]}% (std = {s["binary_std"]}%)')
        add_para(doc, f'Best Binary Accuracy: {s["binary_best"]}%')

        rows = []
        for sd in s90['seeds']:
            rows.append([str(sd['seed']),
                         f'{sd["multiclass_acc"]}%', f'{sd["binary_acc"]}%',
                         f'{sd["specificity"]}%', f'{sd["sensitivity"]}%',
                         f'{sd.get("balanced_acc", "N/A")}%'])
        add_para(doc, 'Table C3: 90/10 Split Results per Seed', bold=True)
        add_table(doc, ['Seed', 'Multi Acc', 'Binary Acc', 'Specificity', 'Sensitivity', 'Balanced Acc'], rows)

    # C.4 60/40 Detailed
    add_heading(doc, 'C.4 60/40 Split Detailed Results', 2)
    if all_splits and '60/40' in all_splits:
        s60 = all_splits['60/40']
        s = s60['summary']
        add_para(doc, f'Mean Multiclass Accuracy: {s["multi_mean"]}% (std = {s["multi_std"]}%)', bold=True)
        add_para(doc, f'Best Multiclass Accuracy: {s["multi_best"]}% (seed 76)')
        add_para(doc, f'Mean Binary Accuracy: {s["binary_mean"]}% (std = {s["binary_std"]}%)')

        rows = []
        for sd in s60['seeds']:
            rows.append([str(sd['seed']),
                         f'{sd["multiclass_acc"]}%', f'{sd["binary_acc"]}%',
                         f'{sd["specificity"]}%', f'{sd["sensitivity"]}%',
                         f'{sd.get("balanced_acc", "N/A")}%'])
        add_para(doc, 'Table C4: 60/40 Split Results per Seed', bold=True)
        add_table(doc, ['Seed', 'Multi Acc', 'Binary Acc', 'Specificity', 'Sensitivity', 'Balanced Acc'], rows)

    # C.5 Sensitivity/Specificity at 90/10 (from ablation report)
    add_heading(doc, 'C.5 Binary Sensitivity and Specificity at 90/10 (All Seeds)', 2)
    add_para(doc,
        'Full sensitivity and specificity breakdown for binary (healthy vs. disease) '
        'classification at the 90/10 split ratio.')
    add_source_ref(doc, 'Source: ImprovedCDS/CDS_FINAL/evidence_ablation_report.txt, Section 5')

    sens_spec_rows = [
        ['13', '85.7%', '83.3%', '88.9%', '20', '4', '16', '2'],
        ['20', '85.7%', '78.9%', '91.3%', '15', '4', '21', '2'],
        ['27', '81.0%', '73.3%', '85.2%', '11', '4', '23', '4'],
        ['34', '90.5%', '80.0%', '96.3%', '12', '3', '26', '1'],
        ['41', '88.1%', '72.2%', '100.0%', '13', '5', '24', '0'],
        ['48', '92.9%', '84.2%', '100.0%', '16', '3', '23', '0'],
        ['55', '88.1%', '71.4%', '96.4%', '10', '4', '27', '1'],
        ['62', '92.9%', '88.2%', '96.0%', '15', '2', '24', '1'],
        ['69', '83.3%', '78.9%', '87.0%', '15', '4', '20', '3'],
        ['76', '97.6%', '93.3%', '100.0%', '14', '1', '27', '0'],
    ]
    add_para(doc, 'Table C5: Binary Classification at 90/10 (All Seeds)', bold=True)
    add_table(doc, ['Seed', 'Binary Acc', 'Sensitivity', 'Specificity', 'TP', 'FN', 'TN', 'FP'],
              sens_spec_rows)
    add_para(doc, 'Mean Sensitivity: 80.4% (std = 7.0%). Mean Specificity: 94.1% (std = 5.5%).')

    # C.6 Per-Class Accuracy at 60/40 (from ablation report)
    add_heading(doc, 'C.6 Per-Class Accuracy at 60/40 Split (All Seeds)', 2)
    add_para(doc,
        'Per-class accuracy at the 60/40 split ratio, providing protocol-matched comparison '
        'data for Irfan et al. 2022 (who used 60/40 split).')
    add_source_ref(doc, 'Source: ImprovedCDS/CDS_FINAL/evidence_ablation_report.txt, Section 3')

    pc_rows = [
        ['1', 'Normal', '92.6%', '2.6%', '98.0%', '89.4%'],
        ['2', 'CAD', '63.6%', '8.3%', '82.4%', '50.0%'],
        ['3', 'Old Anterior MI', '97.6%', '5.0%', '100.0%', '85.7%'],
        ['4', 'Old Inferior MI', '57.2%', '28.0%', '100.0%', '12.5%'],
        ['5', 'Sinus Tachycardia', '45.8%', '31.5%', '100.0%', '0.0%'],
        ['6', 'Sinus Bradycardia', '56.4%', '18.6%', '87.5%', '25.0%'],
        ['9', 'LBBB', '77.2%', '22.3%', '100.0%', '33.3%'],
        ['10', 'RBBB', '64.1%', '8.6%', '77.3%', '42.9%'],
    ]
    add_para(doc, 'Table C6: Per-Class Accuracy at 60/40 (Mean over 10 Seeds)', bold=True)
    add_table(doc, ['Class', 'Description', 'Mean Acc', 'Std', 'Best', 'Worst'], pc_rows)

    # C.7 Stratified Results (All Protocols)
    add_heading(doc, 'C.7 Stratified Evaluation Results (All Protocols)', 2)
    add_para(doc,
        'A stratified split ensures proportional class representation in both training and test '
        'sets. This is important because rare classes (e.g., class 9 with 9 patients) may be '
        'entirely absent from the test set in random splits, making accuracy estimates unreliable. '
        'We evaluate stratified variants of all validation protocols.')
    add_source_ref(doc, 'Source: ImprovedCDS/CDS_FINAL/task_stratified.py; Results: results_stratified.json')

    strat_data = comp_results.get('stratified') if comp_results else None

    if strat_data:
        strat_protocols = [
            ('stratified_10fold', 'Stratified 10-Fold CV'),
            ('stratified_90_10', 'Stratified 90/10 Split'),
            ('stratified_80_20', 'Stratified 80/20 Split'),
            ('stratified_70_30', 'Stratified 70/30 Split'),
            ('stratified_60_40', 'Stratified 60/40 Split'),
            ('stratified_50_50', 'Stratified 50/50 Split'),
        ]

        # Summary table across all protocols
        summary_rows = []
        for key, label in strat_protocols:
            if key in strat_data:
                s = strat_data[key]['summary']
                summary_rows.append([label,
                    f'{s["acc_mean"]}%', f'{s["acc_std"]}%',
                    f'{s["acc_best"]}%', f'{s["acc_worst"]}%',
                    f'{s["binary_mean"]}%', f'{s["ba_mean"]}%'])

        if summary_rows:
            add_para(doc, 'Table C7: Stratified Results Summary Across All Protocols (10 seeds each)', bold=True)
            add_table(doc,
                ['Protocol', 'Multi Mean', 'Multi Std', 'Multi Best', 'Multi Worst', 'Binary Mean', 'BA Mean'],
                summary_rows)

        # Detailed per-seed tables for each protocol
        table_num = 8
        for key, label in strat_protocols:
            if key in strat_data:
                proto_data = strat_data[key]
                s = proto_data['summary']
                add_heading(doc, f'C.7.{table_num-7} {label} Detailed Results', 3)
                add_para(doc, f'Mean Multiclass Accuracy: {s["acc_mean"]}% (std = {s["acc_std"]}%)', bold=True)
                add_para(doc, f'Mean Binary Accuracy: {s["binary_mean"]}% (std = {s["binary_std"]}%)')
                add_para(doc, f'Mean Balanced Accuracy: {s["ba_mean"]}% (std = {s["ba_std"]}%)')

                rows = []
                for sd in proto_data['seeds']:
                    rows.append([str(sd['seed']),
                                 f'{sd["accuracy"]}%', f'{sd["binary_acc"]}%',
                                 f'{sd["specificity"]}%', f'{sd["sensitivity"]}%',
                                 f'{sd["balanced_acc"]}%', str(sd['n_test'])])
                add_para(doc, f'Table C{table_num}: {label} Results per Seed', bold=True)
                add_table(doc, ['Seed', 'Multi Acc', 'Binary Acc', 'Specificity', 'Sensitivity', 'Balanced Acc', 'N_test'], rows)
                table_num += 1
    else:
        add_para(doc,
            '[Stratified results pending - task_stratified.py has not yet completed.]',
            italic=True)

    # ================================================================
    # APPENDIX D: Ablation Studies
    # ================================================================
    doc.add_page_break()
    add_heading(doc, 'Appendix D: Ablation Studies', 1)
    add_para(doc,
        'This appendix consolidates all ablation experiment results, quantifying the contribution '
        'of each CDS-OVR component.')
    add_source_ref(doc,
        'Source: ImprovedCDS/CDS_FINAL/evidence_ablation.py, ImprovedCDS/CDS_FINAL/ablation_full.py; '
        'Results: ImprovedCDS/CDS_FINAL/evidence_ablation_report.txt')

    add_heading(doc, 'D.1 Component Ablation Summary', 2)
    add_para(doc,
        'Each mechanism was individually disabled and evaluated via 10-fold CV across 10 seeds. '
        'Per-seed accuracy was paired with the full system, and a Wilcoxon signed-rank test '
        'determined statistical significance.')

    ablation_rows = [
        ['Supervised chi2 binning', '+5.75 pp', '0.002', 'Yes', 'Most impactful component'],
        ['Fisher discriminant weighting', '+4.52 pp', '0.002', 'Yes', 'Second most impactful'],
        ['Correlation filtering', '+0.46 pp', '0.313', 'No', 'Modest, prevents overfitting'],
        ['Sex branching', '+0.41 pp', '0.141', 'No', 'Concentrated on class 3'],
        ['against_scale (rare)', '+0.07 pp', '0.734', 'No', 'Rare classes = 8.9% of data'],
        ['Healthy bar', '-0.31 pp', '0.133', 'No', 'Trades accuracy for specificity'],
    ]
    add_para(doc, 'Table D1: Full Component Ablation Results', bold=True)
    add_table(doc,
        ['Component', 'Effect on Accuracy', 'p-value', 'Significant (p<0.05)', 'Notes'],
        ablation_rows)

    add_heading(doc, 'D.2 against_scale Ablation (Rare Classes)', 2)
    add_para(doc,
        'against_scale = 0.5 for rare classes {4, 5, 9} vs. uniform 0.8 for all classes.')
    ablation_rare_rows = [
        ['Overall (10-fold CV)', '84.66% (std=0.93%)', '84.59% (std=0.66%)', '+0.07 pp'],
        ['Class 4 (Old Inferior MI)', '69.3%', '62.7%', '+6.7 pp'],
        ['Class 5 (Sinus Tachycardia)', '59.2%', '50.0%', '+9.2 pp'],
        ['Class 9 (LBBB)', '96.7%', '95.6%', '+1.1 pp'],
    ]
    add_para(doc, 'Table D2: against_scale Ablation Results', bold=True)
    add_table(doc, ['Metric', 'Current (0.5 rare)', 'Uniform (0.8 all)', 'Difference'], ablation_rare_rows)

    add_heading(doc, 'D.3 Sex Branching Ablation', 2)
    add_para(doc, 'Sex-based population branching enabled (SEX_FEAT=1) vs. disabled (no branching).')
    ablation_sex_rows = [
        ['10-fold CV Mean', '84.66% (std=0.93%)', '84.25% (std=0.88%)', '+0.41 pp'],
        ['Wins', '6/10 seeds', '4/10 seeds', '-'],
    ]
    add_para(doc, 'Table D3: Sex Branching Ablation Results', bold=True)
    add_table(doc, ['Metric', 'Sex Branching On', 'Sex Branching Off', 'Difference'], ablation_sex_rows)

    add_heading(doc, 'D.4 Hyperparameter Sensitivity Analysis', 2)
    add_para(doc,
        'Three hyperparameters were varied independently while all others were held at defaults. '
        'Each configuration was evaluated via 10-fold CV across 10 seeds (100 fold evaluations).')
    add_source_ref(doc, 'Source: ImprovedCDS/CDS_FINAL/ablation_full.py, Part 4')

    hp_rows = [
        ['CORR_THRESHOLD', '0.6-1.0', '0.8', '1.44 pp', 'Optimal; degrades below 0.8'],
        ['MAX_BINS', '3-8', '6 (5 optimal)', '3.05 pp', 'MAX_BINS=5 is +0.51 pp better'],
        ['FEATURES_PER_CLASS', '12-24', '18', '0.00 pp', 'No effect (corr filter is bottleneck)'],
    ]
    add_para(doc, 'Table D4: Hyperparameter Sensitivity Sweep', bold=True)
    add_table(doc, ['Parameter', 'Range Tested', 'Current Value', 'Accuracy Range', 'Notes'], hp_rows)

    # ================================================================
    # APPENDIX E: Confusion Matrix and Classification Metrics
    # ================================================================
    doc.add_page_break()
    add_heading(doc, 'Appendix E: Confusion Matrix and Classification Metrics', 1)

    if comp_results and 'confusion_matrix_seed13' in comp_results:
        cm_data = comp_results['confusion_matrix_seed13']
        cm = cm_data['matrix']
        classes = cm_data['classes']
        add_source_ref(doc, 'Source: ImprovedCDS/CDS_FINAL/comprehensive_experiments.py; 10-fold CV, seed 13')

        add_heading(doc, 'E.1 Confusion Matrix (10-Fold CV, Seed 13)', 2)
        headers = ['True \\ Pred'] + [f'Cls {c}' for c in classes]
        rows = []
        for i, cls in enumerate(classes):
            row = [f'Class {cls}'] + [str(cm[i][j]) for j in range(len(classes))]
            rows.append(row)
        add_para(doc, 'Table E1: Confusion Matrix (10-Fold CV, Seed 13, 416 Patients)', bold=True)
        add_table(doc, headers, rows)

        add_heading(doc, 'E.2 Per-Class Precision, Recall, and F1-Score', 2)
        pc = cm_data['per_class']
        rows = []
        for cls_str in sorted(pc.keys(), key=lambda x: int(x)):
            cls = int(cls_str)
            m = pc[cls_str]
            rows.append([str(cls), CLASS_NAMES.get(cls, f'Class {cls}'),
                         str(m['support']),
                         f'{m["precision"]:.4f}', f'{m["recall"]:.4f}', f'{m["f1"]:.4f}'])
        add_para(doc, 'Table E2: Per-Class Classification Metrics', bold=True)
        add_table(doc, ['Class', 'Description', 'Support', 'Precision', 'Recall', 'F1-Score'], rows)

        add_para(doc, f'Macro Precision: {cm_data["macro_precision"]:.4f}')
        add_para(doc, f'Macro Recall: {cm_data["macro_recall"]:.4f}')
        add_para(doc, f'Macro F1-Score: {cm_data["macro_f1"]:.4f}')
    else:
        add_para(doc,
            '[Confusion matrix and P/R/F1 pending - comprehensive_experiments.py has not yet completed.]',
            italic=True)

    # ================================================================
    # APPENDIX F: Computational Cost
    # ================================================================
    doc.add_page_break()
    add_heading(doc, 'Appendix F: Computational Cost and Memory Usage', 1)

    add_para(doc,
        'The CDS-OVR algorithm requires no GPU acceleration, no iterative optimization, and '
        'no gradient computation. All operations are closed-form or single-pass through the data. '
        'This makes it suitable for deployment on resource-constrained embedded systems such as FPGAs.')

    if comp_results and 'computational_cost' in comp_results:
        cc = comp_results['computational_cost']
        add_source_ref(doc, 'Source: ImprovedCDS/CDS_FINAL/comprehensive_experiments.py (tracemalloc profiling)')

        add_heading(doc, 'F.1 Runtime Performance', 2)
        add_para(doc, 'Table F1: Computational Cost Summary', bold=True)
        add_table(doc, ['Operation', 'Time', 'Notes'],
                  [['Single Training (full dataset)', f'{cc["training_time_s"]:.4f} s', 'Build tree + train 8 class models'],
                   ['Single Patient Prediction', f'{cc["prediction_per_patient_ms"]:.4f} ms', 'Route + accumulate + decide'],
                   ['All 416 Predictions', f'{cc["prediction_all_patients_s"]:.4f} s', 'Sequential, no batching'],
                   ['10-Fold CV (1 seed)', f'{cc["10fold_cv_time_s"]:.2f} s', '10 train+test cycles'],
                   ['90/10 Split (1 seed)', f'{cc["90_10_split_time_s"]:.2f} s', '1 train+test cycle']])

        add_heading(doc, 'F.2 Memory Usage', 2)
        add_para(doc, 'Table F2: Memory Usage Summary', bold=True)
        add_table(doc, ['Component', 'Size', 'Description'],
                  [['Data Matrix', f'{cc["data_matrix_mb"]:.2f} MB', '416 x 279 float64 array'],
                   ['Current Memory', f'{cc["memory_current_mb"]:.2f} MB', 'After training all models'],
                   ['Peak Memory', f'{cc["memory_peak_mb"]:.2f} MB', 'Maximum during prediction'],
                   ['Tree Nodes', str(cc['n_tree_nodes']), 'Root + sex-based branches'],
                   ['Total BinModels', str(cc['n_class_models']), '8 classes x 279 features x nodes'],
                   ['Retained Features', str(cc['n_retained_features']), '~18 per class per node']])
    else:
        add_para(doc,
            '[Computational cost results pending - comprehensive_experiments.py has not yet completed.]',
            italic=True)

    add_heading(doc, 'F.3 Computational Complexity Analysis', 2)
    add_para(doc,
        'Training: O(C x N x F x B) where C=8 classes, N=416 patients, F=279 features, '
        'B=6 max bins. The dominant cost is computing supervised bin edges for each feature in '
        'each class model. Feature selection adds O(F^2) for pairwise correlation but only within '
        'the top candidates.')
    add_para(doc,
        'Prediction: O(C x F_r x L) where C=8 classes, F_r=18 retained features per class, '
        'L=number of matched tree nodes (at most 2). Each prediction requires looking up bin '
        'assignments and accumulating evidence, all O(1) per feature.')
    add_para(doc,
        'The algorithm requires no matrix inversions, no eigendecompositions, no iterative '
        'solvers, and no backpropagation.')
    add_source_ref(doc, 'Implementation: ImprovedCDS/CDS_FINAL/cds_ovr.py, 587 lines total')

    # ================================================================
    # APPENDIX G: Mathematical Formulation
    # ================================================================
    doc.add_page_break()
    add_heading(doc, 'Appendix G: Mathematical Formulation', 1)

    add_para(doc,
        'This appendix provides the complete mathematical formulation for each component of the '
        'CDS-OVR algorithm, with references to the relevant literature.')

    add_heading(doc, 'G.1 Supervised Chi-Squared Binning', 2)
    add_para(doc,
        'The supervised binning algorithm partitions continuous feature values into discrete bins '
        'by maximizing the chi-squared statistic between the target class indicator and the bin '
        'assignment. For a candidate split dividing a segment into left and right partitions:')
    add_para(doc, '  chi2 = SUM_i SUM_j (O_ij - E_ij)^2 / E_ij')
    add_para(doc,
        'where O_ij is the observed count (i in {target, rest}, j in {left, right}) and '
        'E_ij = n_j * n_i / n is the expected count under the null hypothesis of independence. '
        'The chi-squared statistic follows an asymptotic chi2(1) distribution under the null. '
        'A minimum gain threshold of 0.5 prevents overfitting (Kerber, 1992).')
    add_source_ref(doc, 'Implementation: cds_ovr.py, lines 209-265 (chi2_binning function)')

    add_heading(doc, 'G.2 Laplace-Smoothed Posterior', 2)
    add_para(doc,
        'The class-conditional posterior probability for class c in bin b is computed with '
        'Laplace smoothing:')
    add_para(doc, '  P(c|b) = (n_cb + alpha * pi_c) / (n_b + alpha)')
    add_para(doc,
        'where n_cb is the number of class-c patients in bin b, n_b is the total count in bin b, '
        'pi_c = n_c/N is the class prior, and alpha = 1.0 is the smoothing parameter. This is '
        'equivalent to placing a Dirichlet prior on the bin-conditional class distribution '
        '(Manning et al., 2008).')
    add_source_ref(doc, 'Implementation: cds_ovr.py, lines 267-290')

    add_heading(doc, 'G.3 Fisher Discriminant Ratio (FDR)', 2)
    add_para(doc,
        'The Fisher Discriminant Ratio measures univariate separability between the target class '
        'and all other classes:')
    add_para(doc, '  FDR(f, c) = (mu_c - mu_rest)^2 / (sigma_c^2 + sigma_rest^2 + epsilon)')
    add_para(doc,
        'where mu_c and sigma_c^2 are the mean and variance of feature f for class c patients, '
        'mu_rest and sigma_rest^2 are for all other patients, and epsilon = 1e-10 prevents '
        'division by zero. FDR equals the squared signal-to-noise ratio for the binary '
        'classification task (Fisher, 1936).')
    add_source_ref(doc, 'Implementation: cds_ovr.py, lines 316-321')

    add_heading(doc, 'G.4 Fisher Weight Computation', 2)
    add_para(doc, 'Each retained feature receives a weight based on its relative FDR:')
    add_para(doc, '  w_f = max(sqrt(FDR_f / (FDR_max + epsilon)), 0.1)')
    add_para(doc,
        'where FDR_max is the maximum FDR across all retained features for the class. '
        'The square root compresses the dynamic range, and the floor of 0.1 ensures no feature '
        'is entirely silenced.')
    add_source_ref(doc, 'Implementation: cds_ovr.py, line 430')

    add_heading(doc, 'G.5 Action Score (Feature Utility)', 2)
    add_para(doc,
        'The action score quantifies how much a feature contributes to distinguishing the target '
        'class from the population baseline:')
    add_para(doc, '  S(f, c) = SUM_b |P(c|b) - pi_c| * min(1, n_b / CS)')
    add_para(doc,
        'where the sum is over bins b with at least MS patients, P(c|b) is the Laplace-smoothed '
        'posterior, pi_c is the class prior, and CS is the confidence support parameter. '
        'The confidence factor min(1, n_b/CS) prevents bins with few observations from '
        'contributing disproportionately.')
    add_source_ref(doc, 'Implementation: cds_ovr.py, lines 292-314')

    add_heading(doc, 'G.6 Dual Evidence Accumulation', 2)
    add_para(doc, 'Evidence for and against each class is accumulated separately:')
    add_para(doc, '  AF_for = SUM_f [shift_f * confidence_f * w_f]   for shift_f >= 0')
    add_para(doc, '  AF_against = SUM_f [|shift_f| * confidence_f * w_f * alpha_c]   for shift_f < 0')
    add_para(doc,
        'where shift_f = P(c|b_f) - pi_c is the posterior shift for the patient in feature f, '
        'confidence_f = min(1, bc_f / 10) is the confidence based on bin population, '
        'w_f is the Fisher weight, and alpha_c is the against_scale parameter '
        '(0.5 for rare classes {4, 5, 9}; 0.8 for common classes).')
    add_source_ref(doc, 'Implementation: cds_ovr.py, lines 399-442')

    add_heading(doc, 'G.7 Ratio Score', 2)
    add_para(doc, 'The ratio score combines positive and negative evidence:')
    add_para(doc, '  R_c = (AF_for + epsilon) / (AF_against + epsilon)')
    add_para(doc,
        'where epsilon = 0.1 provides numerical stability. Ratio > 1.0 indicates net positive '
        'evidence for the class; ratio << 1.0 indicates net negative evidence.')

    add_heading(doc, 'G.8 Dynamic Healthy Bar and Decision Logic', 2)
    add_para(doc, 'The effective threshold for each disease class is:')
    add_para(doc, '  T_eff = max(T_c - 0.3 * I(h_score < 2.0), min(1.05 * h_score, 5.0))')
    add_para(doc,
        'where T_c is the per-class base threshold (from CLASS_THRESHOLDS), h_score = R_1 is '
        'the ratio score for the healthy class, 0.3 is the suspicion offset, 2.0 is the '
        'suspicion threshold, 1.05 is the healthy weight, and 5.0 is the healthy bar cap.')
    add_para(doc,
        'A class c is a candidate if R_c >= T_eff. The prediction is the candidate with the '
        'maximum normalized margin:')
    add_para(doc, '  prediction = argmax_c (R_c - T_eff) / max(T_eff, 0.1)')
    add_para(doc, 'If no disease class exceeds its threshold, the patient is predicted as Healthy (class 1).')
    add_source_ref(doc, 'Implementation: cds_ovr.py, lines 455-468 (healthy bar), lines 470-480 (decision)')

    add_heading(doc, 'G.9 Correlation-Based Feature Filtering', 2)
    add_para(doc,
        'The absolute Pearson correlation between features f1 and f2 is:')
    add_para(doc, '  |r| = |SUM((x_i - mu_x)(y_i - mu_y))| / sqrt(SUM(x_i - mu_x)^2 * SUM(y_i - mu_y)^2)')
    add_para(doc,
        'Features with |r| > 0.8 are considered redundant. During greedy selection, each '
        'candidate feature is checked against all already-selected features, and rejected if any '
        'pairwise correlation exceeds the threshold (Guyon & Elisseeff, 2003).')
    add_source_ref(doc, 'Implementation: cds_ovr.py, lines 330-367 (feature selection loop)')

    add_heading(doc, 'G.10 Evaluation Metrics', 2)
    add_para(doc, 'Multiclass Accuracy = (1/N) * SUM_i I(y_pred_i == y_true_i)')
    add_para(doc, 'Specificity = TN / (TN + FP)   [for healthy class]')
    add_para(doc, 'Sensitivity = TP / (TP + FN)   [for disease detection]')
    add_para(doc, 'Balanced Accuracy = (1/C) * SUM_c (correct_c / n_c)')
    add_para(doc, 'Binary Accuracy = SUM_i I((true_i==1) == (pred_i==1)) / N')
    add_para(doc, 'Precision_c = TP_c / (TP_c + FP_c)')
    add_para(doc, 'Recall_c = TP_c / (TP_c + FN_c)')
    add_para(doc, 'F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)')
    add_para(doc, 'AUC = integral(TPR(FPR) dFPR)   [trapezoidal rule]')

    # ================================================================
    # APPENDIX H: References
    # ================================================================
    doc.add_page_break()
    add_heading(doc, 'Appendix H: References', 1)

    refs = [
        '[1] Kerber, R. (1992). "ChiMerge: Discretization of numeric attributes." In Proceedings of the 10th National Conference on Artificial Intelligence (AAAI-92), pp. 123-128.',
        '[2] Manning, C. D., Raghavan, P., & Schutze, H. (2008). Introduction to Information Retrieval. Cambridge University Press. Chapter 13.',
        '[3] Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems." Annals of Eugenics, 7(2), 179-188.',
        '[4] Guyon, I., & Elisseeff, A. (2003). "An introduction to variable and feature selection." JMLR, 3, 1157-1182.',
        '[5] Guvenir, H. A., Acar, B., Demiroz, G., & Cekin, A. (1997). "A supervised machine learning algorithm for arrhythmia analysis." In Proceedings of Computers in Cardiology, pp. 433-436.',
        '[6] Sharma, A., et al. (2024). "Optimized Random Forest for arrhythmia classification on the UCI dataset." Engineering Research Express, 6, 035209.',
        '[7] Mustaqeem, A., Anwar, S. M., & Majid, M. (2018). "Multiclass classification of cardiac arrhythmia using improved feature selection and SVM invariants." Computational and Mathematical Methods in Medicine, 2018, 7310496.',
        '[8] Irfan, S., et al. (2022). "Heartbeat classification and arrhythmia detection using a multi-model deep-learning technique." Sensors, 22(15), 5606.',
        '[9] Gupta, V., Srinivasan, S. and Kudli, S. S. (2014). "Prediction and classification of cardiac arrhythmia."',
        '[10] Jadhav, S. M., Nalbalwar, S. L., & Ghatol, A. A. (2012). "Artificial neural network models based cardiac arrhythmia disease diagnosis from ECG data." International Journal of Computer Applications, 44(15).',
        '[11] Plawiak, P. (2018). "Novel methodology of cardiac health recognition based on ECG signals and evolutionary-neural system." Expert Systems with Applications, 92, 334-349.',
        '[12] Khan Mamun, M. M. R. (2021). "Arrhythmia classification using hybrid feature selection approach." IEEE CCECE 2021.',
        '[13] Islam, M. S., et al. (2023). "A review of arrhythmia classification techniques." arXiv:2301.10174.',
        '[14] UCI Machine Learning Repository. (1998). Arrhythmia Data Set. https://archive.ics.uci.edu/ml/datasets/arrhythmia',
        '[15] Rifkin, R., & Klautau, A. (2004). "In defense of one-vs-all classification." JMLR, 5, 101-141.',
        '[16] Wilcoxon, F. (1945). "Individual comparisons by ranking methods." Biometrics Bulletin, 1(6), 80-83.',
    ]
    for ref in refs:
        add_para(doc, ref, font_size=10)

    # ================================================================
    # APPENDIX I: File Location Index
    # ================================================================
    doc.add_page_break()
    add_heading(doc, 'Appendix I: Repository File Index', 1)
    add_para(doc,
        'This appendix lists all source files, data files, and generated outputs referenced '
        'in this report, with descriptions of their contents.')

    file_index = [
        ['ImprovedCDS/CDS_FINAL/cds_ovr.py', '587 lines', 'Final CDS-OVR algorithm implementation'],
        ['ImprovedCDS/cds.py', '491 lines', 'Original base CDS algorithm (binary classifier)'],
        ['ImprovedCDS/CDS_FINAL/run_all_splits.py', '157 lines', 'Script for all split ratio evaluations (10 seeds each)'],
        ['ImprovedCDS/CDS_FINAL/evidence_analysis.py', '-', 'Evidence quality and dataset analysis experiments (14 sections)'],
        ['ImprovedCDS/CDS_FINAL/evidence_ablation.py', '-', 'Ablation experiments (against_scale, sex branching, per-class accuracy)'],
        ['ImprovedCDS/CDS_FINAL/ablation_full.py', '-', 'Full component ablation with statistical significance testing'],
        ['ImprovedCDS/CDS_FINAL/comprehensive_experiments.py', '-', 'AUC/ROC, stratified splits, timing, memory profiling'],
        ['ImprovedCDS/CDS_FINAL/results_all_splits.json', '-', 'All split ratio results (50/50 through 90/10, 10-fold CV)'],
        ['ImprovedCDS/CDS_FINAL/comprehensive_results.json', '-', 'AUC/ROC, confusion matrix, computational cost results'],
        ['ImprovedCDS/CDS_FINAL/evidence_report.txt', '-', 'Detailed evidence analysis output (14 sections)'],
        ['ImprovedCDS/CDS_FINAL/evidence_ablation_report.txt', '-', 'Ablation experiment output'],
        ['ImprovedCDS/output/loocv_trace.json', '~3.1 MB', 'Base CDS LOOCV trace (452 patient predictions)'],
        ['ImprovedCDS/output/loocv_trace_ovr.json', '~3.5 MB', 'CDS-OVR LOOCV trace (452 patient predictions)'],
        ['ImprovedCDS/output/logs/', '20 files', '10-fold CV log files for each seed (v2 and v3 formats)'],
        ['data/arrhythmia.data', '-', 'UCI Arrhythmia Dataset (452 x 280, CSV)'],
    ]
    add_para(doc, 'Table I1: Complete File Index', bold=True)
    add_table(doc, ['File Path', 'Size/Lines', 'Description'], file_index)

    return True


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  CDS-OVR Report Restructuring Tool")
    print("=" * 60)

    # Copy original to temp to avoid OneDrive issues
    orig = os.path.join(SRC_DIR, 'CDS_Overall_Report.docx')
    temp_dir = os.environ.get('TEMP', '/tmp')
    src = os.path.join(temp_dir, 'CDS_Report_Restructure.docx')
    shutil.copy2(orig, src)

    print(f"\nOpening report from {src}...")
    doc = Document(src)
    print(f"  {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables")

    # Phase 1: Fix em/en dashes
    print("\nPhase 1: Replacing em/en dashes...")
    dash_count = replace_em_dashes(doc)
    print(f"  Replaced {dash_count} em/en dashes")

    # Phase 2: Fix structure (Part dividers)
    print("\nPhase 2: Fixing document structure...")
    struct_count = fix_structure(doc)
    print(f"  Made {struct_count} structural changes")

    # Phase 3: Add file location references
    print("\nPhase 3: Adding file location references...")
    ref_count = add_file_location_refs(doc)
    print(f"  Added {ref_count} source annotations")

    # Phase 4: Mark redundancies
    print("\nPhase 4: Marking redundant content...")
    red_count = mark_redundancies(doc)
    print(f"  Marked {red_count} redundancies")

    # Phase 5: Build appendices
    print("\nPhase 5: Building comprehensive appendices...")
    build_appendices(doc)
    print("  Appendices A-I built successfully")

    # Save
    out_path = os.path.join(SRC_DIR, 'CDS_Final_Report.docx')
    doc.save(out_path)
    print(f"\nSaved final report to {out_path}")
    print(f"  Final: {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables")


if __name__ == '__main__':
    main()
