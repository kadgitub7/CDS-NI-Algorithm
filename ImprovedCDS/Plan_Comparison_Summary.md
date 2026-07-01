# Implementation Plans: Comparison Summary

## Three Plans at a Glance

| Aspect | Plan A: SPRT | Plan B: Dual-AF Evidence | Plan C: Conservative |
|--------|-------------|-------------------------|---------------------|
| **Core change** | Replace AF with log-likelihood ratio test | Replace healthy range with probabilistic bins, keep two AFs | Keep healthy ranges, add second AF |
| **Healthy range** | Eliminated | Eliminated | Kept (adaptive) |
| **Evidence model** | Log-likelihood ratios from posteriors | P(H\|bin) * confidence * r_H | Inside/outside range with dual increment |
| **Decision rule** | SPRT boundaries (alpha/beta) | AF >= T and gap >= delta | AF >= T and gap >= delta |
| **Theoretical basis** | Wald's SPRT (optimal stopping) | Bayesian evidence accumulation | Extended CDS framework |
| **Number of ideas used** | 19 | 20 | 12 |
| **Code complexity** | High (major rewrite of Alg 2+4) | High (major rewrite of Alg 2+4) | Low (moderate changes to Alg 2+4) |
| **Risk** | Medium (new framework) | Medium (new evidence model) | Low (incremental change) |
| **Recommended for** | Best theoretical paper | Best practical system | Baseline comparison |

## Idea Coverage Matrix

Shows which ideas appear in which plan.

| ID | Idea | Plan A | Plan B | Plan C |
|----|------|--------|--------|--------|
| **Core Architecture** | | | | |
| A8+A5 | Probabilistic bins | Y | Y | - |
| Idea 2 | Adaptive healthy range | - | - | Y |
| K8 | SPRT framework | Y | - | - |
| N3/N4 | Bayes factor / posterior odds | Y | - | - |
| D1 | Additive accumulation | - | Y | Y |
| **Evidence Quality** | | | | |
| B1 | Posterior probability table | Y | Y | - |
| B3 | Confidence-weighted evidence | - | Y | - |
| B4 | Posterior smoothing (Laplace) | Y | Y | Y |
| I8 | Minimum support requirement | Y | Y | - |
| I1 | Capped contributions | Y | Y | Y |
| **Feature Selection** | | | | |
| C1 | Dual weights r_H, r_U | Y | Y | Y |
| I4 | Correlation penalty | Y | Y | - |
| L6 | Backward elimination | Y | Y | - |
| **Decision Rule** | | | | |
| E1 | Symmetric thresholds | - | Y | Y |
| E4 | Gap-based decision | - | Y | Y |
| E9 | Minimum feature count | Y | Y | - |
| **Prediction Flow** | | | | |
| F1 | Most discriminative first | Y | Y | - |
| F8 | Deterministic first action | Y | Y | Y |
| G1 | AF inheritance | Y | Y | Y |
| K2 | Contradictory evidence detector | Y | Y | - |
| **Methodology** | | | | |
| I3 | Inner CV for thresholds | Y | Y | Y |
| M1 | Multi-metric evaluation | Y | Y | Y |
| **Edge Cases** | | | | |
| H1 | Missing data as evidence | Y | - | - |
| H5 | Outliers to nearest bin | Y | Y | Y |

## Recommended Execution Order

### If time permits all three:
1. **Implement Plan C first** (1-2 days) — smallest change, establishes
   the dual-AF baseline. Run LOOCV. Record all metrics.
2. **Implement Plan B** (3-5 days) — probabilistic bins + dual-AF.
   Compare against Plan C to measure the value of eliminating healthy ranges.
3. **Implement Plan A** (3-5 days) — SPRT. Compare against Plan B to
   measure the value of the formal SPRT framework.

### If time is limited:
- **Implement Plan A directly** — it's the strongest approach and the
  most publishable. Plan C can be implemented later as a baseline for
  the paper's evaluation section.

### For the paper:
Present all three with a progression narrative:
1. Plan C shows that dual-AF alone improves accuracy.
2. Plan B shows that probabilistic bins further improve accuracy.
3. Plan A shows that formalizing as SPRT provides theoretical guarantees
   and optimal stopping.

## Conflicts Between Plans

These plans are **mutually exclusive** — you implement ONE of them, not
all three simultaneously. The conflicts:

1. **Healthy range vs. probabilistic bins**: Plans A and B eliminate the
   healthy range. Plan C keeps it. You can't do both.
2. **SPRT vs. dual-AF accumulation**: Plan A uses one cumulative LLR.
   Plans B and C use two separate AFs. Different decision frameworks.
3. **Evidence function**: Each plan computes evidence differently.
   - Plan A: log(P(x|U)/P(x|H))
   - Plan B: P(H|bin) * confidence * r_H
   - Plan C: afi(p, r, g) applied to inside/outside

However, all three share common infrastructure:
- Inner CV for thresholds (I3) — same concept, different parameters
- Multi-metric reporting (M1) — identical
- Laplace smoothing (B4) — identical
- Deterministic ordering (F8) — identical
- AF inheritance (G1) — same concept
- Outlier handling (H5) — identical

## My Recommendation

**Plan A (SPRT) is the best approach** for the following reasons:

1. **Theoretical foundation**: The SPRT has provable optimality properties.
   This transforms the paper from "we tried something and it worked" to
   "we applied a well-established statistical framework."

2. **Simplicity of decision rule**: One number (cumulative LLR), two
   boundaries. No need to tune T, delta, and k_min separately.

3. **Optimal stopping**: Wald proved that among all sequential tests with
   the same error rates, the SPRT uses the fewest observations on average.
   This directly addresses the CDS goal of minimizing PACs.

4. **Subsumes other ideas**: K8 + N3/N4 naturally incorporates the Bayes
   factor, posterior odds, and log-likelihood frameworks. You get the
   theoretical elegance of all three without implementing them separately.

5. **Inner CV is cleaner**: Only two primary hyperparameters (alpha, beta)
   plus cap and k_min. Plans B and C have T, delta, cap, k_min — more
   parameters with less theoretical justification for their ranges.

**Plan B is the best practical alternative** if the SPRT feels like too
large a departure from the paper, or if interpretability of separate
AF_H and AF_U values is important for the clinical audience.

**Plan C should be implemented regardless** as a baseline for comparison.
