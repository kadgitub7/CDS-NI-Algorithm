"""Generate all CDS incremental analysis variant files.

Each variant is a short Python script that configures CDSConfig
and calls run_variant(). Run this script once to generate all files.
"""
import os

DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = '''"""{doc}"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
{config_lines}
)

if __name__ == "__main__":
    run_variant(cfg, label="{label}")
'''

# ─── Variant definitions ───
# Each: (filename, label, docstring, config_dict)

variants = []

def add(filename, label, doc, **config):
    variants.append((filename, label, doc, config))


# ═══════════════════════════════════════════════════════════
# 01: OVR Baseline (all defaults)
# ═══════════════════════════════════════════════════════════
add("01_ovr_baseline", "01: OVR Baseline",
    "01 — OVR Baseline: Sturges binning, score-based refinement, simple AF, fixed threshold.\n\nThis is the minimal OVR conversion of the original algorithm.\nAll subsequent changes are measured against this baseline.")

# ═══════════════════════════════════════════════════════════
# 02: Supervised Binning (chi-squared)
# ═══════════════════════════════════════════════════════════
add("02_supervised_binning", "02: +Supervised Binning (MAX_BINS=6)",
    "02 — OVR + Supervised chi-squared binning (MAX_BINS=6).\n\nReplaces Sturges' uniform binning with chi-squared supervised binning\nthat finds optimal split points based on class separation.",
    binning='supervised', max_bins=6)

for mb in [4, 5, 6, 7, 8]:
    add(f"02{chr(ord('a')+[4,5,6,7,8].index(mb))}_maxbins_{mb}",
        f"02: +Supervised Binning (MAX_BINS={mb})",
        f"02 param sweep — Supervised binning with MAX_BINS={mb}.",
        binning='supervised', max_bins=mb)

# ═══════════════════════════════════════════════════════════
# 03: Correlation Filter + Features Per Class
# ═══════════════════════════════════════════════════════════
add("03_correlation_filter", "03: +Correlation Filter (0.8, FPC=18)",
    "03 — OVR + Correlation-based feature filter.\n\nLimits features per class to 18, discards features with\ncorrelation > 0.8 to remove redundancy.",
    corr_threshold=0.8, features_per_class=18)

for ct in [0.6, 0.7, 0.8, 0.9, 0.95]:
    tag = str(ct).replace('.', '')
    add(f"03{chr(ord('a')+[0.6,0.7,0.8,0.9,0.95].index(ct))}_corr_{tag}",
        f"03: +Corr Filter (threshold={ct})",
        f"03 param sweep — Correlation threshold = {ct}, FPC=18.",
        corr_threshold=ct, features_per_class=18)

for fpc in [10, 14, 18, 22, 26]:
    add(f"03{chr(ord('f')+[10,14,18,22,26].index(fpc))}_fpc_{fpc}",
        f"03: +Corr Filter (FPC={fpc})",
        f"03 param sweep — Features per class = {fpc}, CORR=0.8.",
        corr_threshold=0.8, features_per_class=fpc)

# ═══════════════════════════════════════════════════════════
# 04: Dual AF
# ═══════════════════════════════════════════════════════════
add("04_dual_af", "04: +Dual AF",
    "04 — OVR + Dual AF (for/against evidence tracking).\n\nSplits the AF score into positive evidence (FOR the class)\nand negative evidence (AGAINST), allowing the model to weigh\ncontradictory signals separately.",
    af_mode='dual')

# ═══════════════════════════════════════════════════════════
# 05: Fisher Weighting
# ═══════════════════════════════════════════════════════════
add("05_fisher_weighting", "05: +Fisher Weighting",
    "05 — OVR + Fisher discriminant weighting.\n\nWeights feature contributions by their Fisher discriminant ratio,\nnormalizing to [0.1, 1.0]. Features that better separate target\nfrom rest get higher weight.",
    fisher=True)

# ═══════════════════════════════════════════════════════════
# 06: Ratio Scoring
# ═══════════════════════════════════════════════════════════
add("06_ratio_scoring", "06: +Ratio Scoring (eps=0.1)",
    "06 — OVR + Ratio scoring.\n\nChanges final score from simple sum to:\n  Score = (AF_for + eps) / (AF_against + eps)\nAutomatically uses dual AF for the computation.\nScore=1 means balanced evidence; higher = more evidence for class.",
    af_mode='dual', scoring='ratio', ratio_eps=0.1)

for eps in [0.05, 0.1, 0.2]:
    tag = str(eps).replace('.', '')
    add(f"06{chr(ord('a')+[0.05,0.1,0.2].index(eps))}_eps_{tag}",
        f"06: +Ratio Scoring (eps={eps})",
        f"06 param sweep — Ratio scoring with epsilon = {eps}.",
        af_mode='dual', scoring='ratio', ratio_eps=eps)

# ═══════════════════════════════════════════════════════════
# 07: Healthy Bar + Threshold + Suspicion
# ═══════════════════════════════════════════════════════════
add("07_healthy_bar", "07: +Healthy Bar/Threshold/Suspicion",
    "07 — OVR + Healthy bar dynamic thresholding.\n\nAdds: healthy_bar = min(1.05 * Score(healthy), 5.0) as floor.\nSuspicion: if Score(healthy) < 2.0, thresholds -= 0.3.\nRequires ratio scoring (enabled automatically).",
    af_mode='dual', scoring='ratio', ratio_eps=0.1,
    threshold_mode='healthy_bar',
    class_thresholds={2:3.5, 3:5.0, 4:4.0, 5:3.5, 6:3.5, 9:5.0, 10:3.5},
    healthy_weight=1.05, suspicion_hcut=2.0, suspicion_offset=0.3,
    healthy_bar_cap=5.0)

for hw in [0.9, 1.0, 1.05]:
    tag = str(hw).replace('.', '')
    add(f"07a_hweight_{tag}",
        f"07: Healthy Weight={hw}",
        f"07 param sweep — HEALTHY_WEIGHT = {hw}.",
        af_mode='dual', scoring='ratio', ratio_eps=0.1,
        threshold_mode='healthy_bar',
        class_thresholds={2:3.5, 3:5.0, 4:4.0, 5:3.5, 6:3.5, 9:5.0, 10:3.5},
        healthy_weight=hw, suspicion_hcut=2.0, suspicion_offset=0.3,
        healthy_bar_cap=5.0)

for sc in [1.5, 2.0, 2.5]:
    tag = str(sc).replace('.', '')
    add(f"07d_shcut_{tag}",
        f"07: Suspicion HCut={sc}",
        f"07 param sweep — SUSPICION_HCUT = {sc}.",
        af_mode='dual', scoring='ratio', ratio_eps=0.1,
        threshold_mode='healthy_bar',
        class_thresholds={2:3.5, 3:5.0, 4:4.0, 5:3.5, 6:3.5, 9:5.0, 10:3.5},
        healthy_weight=1.05, suspicion_hcut=sc, suspicion_offset=0.3,
        healthy_bar_cap=5.0)

for so in [0.15, 0.3, 0.45]:
    tag = str(so).replace('.', '')
    add(f"07g_soffset_{tag}",
        f"07: Suspicion Offset={so}",
        f"07 param sweep — SUSPICION_OFFSET = {so}.",
        af_mode='dual', scoring='ratio', ratio_eps=0.1,
        threshold_mode='healthy_bar',
        class_thresholds={2:3.5, 3:5.0, 4:4.0, 5:3.5, 6:3.5, 9:5.0, 10:3.5},
        healthy_weight=1.05, suspicion_hcut=2.0, suspicion_offset=so,
        healthy_bar_cap=5.0)

for hc in [3.0, 5.0, 7.0]:
    tag = str(hc).replace('.', '')
    add(f"07j_hcap_{tag}",
        f"07: Healthy Bar Cap={hc}",
        f"07 param sweep — HEALTHY_BAR_CAP = {hc}.",
        af_mode='dual', scoring='ratio', ratio_eps=0.1,
        threshold_mode='healthy_bar',
        class_thresholds={2:3.5, 3:5.0, 4:4.0, 5:3.5, 6:3.5, 9:5.0, 10:3.5},
        healthy_weight=1.05, suspicion_hcut=2.0, suspicion_offset=0.3,
        healthy_bar_cap=hc)

# ═══════════════════════════════════════════════════════════
# 08: Rare Class Parameters
# ═══════════════════════════════════════════════════════════
add("08_rare_class_params", "08: +Rare Class Params (4,5,9)",
    "08 — OVR + Per-class support parameters for rare classes.\n\nRare classes {4,5,9} get: MIN_SUPPORT=2, CONF_SUPPORT=5,\nAGAINST_SCALE=0.5. Common classes keep defaults.",
    rare_classes={4, 5, 9}, rare_min_support=2,
    rare_conf_support=5, rare_against_scale=0.5)

for asc in [0.3, 0.5, 0.7]:
    tag = str(asc).replace('.', '')
    add(f"08a_ascale_{tag}",
        f"08: Rare Against Scale={asc}",
        f"08 param sweep — Rare class AGAINST_SCALE = {asc}.",
        rare_classes={4, 5, 9}, rare_min_support=2,
        rare_conf_support=5, rare_against_scale=asc)

for ms in [1, 2, 3]:
    add(f"08d_minsup_{ms}",
        f"08: Rare Min Support={ms}",
        f"08 param sweep — Rare class MIN_SUPPORT = {ms}.",
        rare_classes={4, 5, 9}, rare_min_support=ms,
        rare_conf_support=5, rare_against_scale=0.5)

for cs in [3, 5, 8]:
    add(f"08g_confsup_{cs}",
        f"08: Rare Conf Support={cs}",
        f"08 param sweep — Rare class CONF_SUPPORT = {cs}.",
        rare_classes={4, 5, 9}, rare_min_support=2,
        rare_conf_support=cs, rare_against_scale=0.5)

# ═══════════════════════════════════════════════════════════
# 09: Remove Classes
# ═══════════════════════════════════════════════════════════
add("09_remove_classes", "09: +Remove Classes {7,8,11,12,13}",
    "09 — OVR + Remove sparse classes from dataset.\n\nClasses {7,8,11,12,13} have very few samples and add noise.\nRemoving them focuses the model on classes with enough data.",
    remove_classes={7, 8, 11, 12, 13})

# ═══════════════════════════════════════════════════════════
# 10: Laplace Smoothing
# ═══════════════════════════════════════════════════════════
add("10_laplace", "10: +Laplace Smoothing (alpha=1.0)",
    "10 — OVR + Laplace smoothing on bin probabilities.\n\nP(class|bin) = (target_count + alpha*prior) / (bin_count + alpha)\nPrevents zero probabilities in sparse bins.",
    laplace_alpha=1.0)

for la in [0.5, 1.0, 2.0]:
    tag = str(la).replace('.', '')
    add(f"10{chr(ord('a')+[0.5,1.0,2.0].index(la))}_laplace_{tag}",
        f"10: Laplace alpha={la}",
        f"10 param sweep — LAPLACE_ALPHA = {la}.",
        laplace_alpha=la)

# ═══════════════════════════════════════════════════════════
# 11: Per-class Thresholds
# ═══════════════════════════════════════════════════════════
add("11_class_thresholds", "11: +Per-Class Thresholds",
    "11 — OVR + Per-class decision thresholds (fixed mode).\n\nDifferent classes need different evidence thresholds.\nUses the same thresholds as the final model but without\nhealthy bar or suspicion.",
    class_thresholds={2:3.5, 3:5.0, 4:4.0, 5:3.5, 6:3.5, 9:5.0, 10:3.5})


# ═══════════════════════════════════════════════════════════
# Generate files
# ═══════════════════════════════════════════════════════════

def format_value(v):
    if isinstance(v, set):
        return '{' + ', '.join(str(x) for x in sorted(v)) + '}'
    elif isinstance(v, dict):
        return '{' + ', '.join(f'{k}: {v}' for k, v in sorted(v.items())) + '}'
    elif isinstance(v, str):
        return f"'{v}'"
    else:
        return repr(v)


def generate():
    count = 0
    for filename, label, doc, config in variants:
        config_lines = []
        for k, v in sorted(config.items()):
            config_lines.append(f"    {k}={format_value(v)},")
        if not config_lines:
            config_str = "    # All defaults"
        else:
            config_str = '\n'.join(config_lines)

        content = TEMPLATE.format(
            doc=doc,
            config_lines=config_str,
            label=label
        )

        path = os.path.join(DIR, f"{filename}.py")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        count += 1
        print(f"  Generated: {filename}.py")

    print(f"\nGenerated {count} variant files.")


if __name__ == "__main__":
    generate()
