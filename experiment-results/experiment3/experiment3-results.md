# Experiment 3 Results — Mention Candidate Strategy (mBERT, CorefUD 1.4)

**Task:** Czech coreference resolution, evaluated on CorefUD Czech-PDT dev
**System:** Span-based coreference model (attended antecedent, coarse-to-fine)
**Fixed variables:** Encoder = mBERT (`bert-base-multilingual-cased`), training data =
Czech PDT only (CorefUD 1.4, same as Experiment 2 condition A), model architecture
**Varied variable:** Mention candidate strategy — span width limit, span pruning
aggressiveness, and context window extension, each varied independently
**Evaluation:** CorefUD scorer, head match, best model checkpoint, `cs_pdt-corefud-dev.conllu`
**Primary metric:** CoNLL F1 (average of MUC, B³, CEAFe) **without singletons**, plus
candidate-stage and post-pruning-stage gold-mention recall (new diagnostics added for
this experiment — see `experiment-runbooks/experiment3/experiment3-mention-candidate-strategy.md`)

---

## Results Table

| Condition | Axis | Value | Candidate-stage recall | Post-pruning recall | MUC F1 | B³ F1 | CEAFe F1 | **CoNLL** |
|-----------|------|-------|------------------------|----------------------|--------|-------|---------|-----------|
| Base | (all, system default) | width=30, ratio=0.4, context=0 | 97.04% (20142/20757) | 93.18% (19341/20757) | 72.59 | 65.67 | 64.45 | **67.57** |
| Width: small | span width | max_span_width=15 | 90.52% (18789/20757) | 87.97% (18259/20757) | 68.75 | 60.31 | 57.54 | **62.20** |
| Width: extended | span width | max_span_width=46 | 99.03% (20555/20757) | 95.21% (19763/20757) | 73.93 | 67.34 | 66.57 | **69.28** |
| Pruning: tight | pruning | top_span_ratio=0.2 | 97.04% (20142/20757) | 83.43% (17317/20757) | 70.10 | 62.79 | 61.72 | **64.87** |
| Pruning: loose | pruning | top_span_ratio=0.6 | 97.04% (20142/20757) | 94.38% (19590/20757) | 72.43 | 65.41 | 64.38 | **67.41** |
| Context: small | context window | context=16, max_segment_len=480 | 97.04% (20142/20757) | 93.35% (19376/20757) | 72.56 | 65.52 | 64.17 | **67.42** |
| Context: extended | context window | context=32, max_segment_len=448 | 97.04% (20142/20757) | 93.56% (19420/20757) | 72.53 | 65.45 | 64.21 | **67.40** |
| Context: wide | context window | context=64, max_segment_len=384 | 97.04% (20142/20757) | 93.36% (19379/20757) | 72.30 | 65.12 | 63.97 | **67.13** |

Zero anaphora F1 (best checkpoint): base 84.73, width-small 82.62, width-extended
85.24, pruning-tight 83.72, pruning-loose 83.92, context-small 84.53, context-extended
84.36, context-wide 84.98.

Reference: Exp2 condition A (mBERT, CorefUD 1.4, cs_pdt only): CoNLL 67.55 — the same
data/encoder as `train_exp3_base`, which reproduces it closely here (67.57, +0.02),
confirming the two new diagnostic hooks (`context_window_subtokens=0`,
`log_mention_recall_diagnostics`) don't perturb baseline behavior.

---

## Notes on the Context-Window Conditions

`train_exp3_context_wide` (context=64) was added after small/extended both finished:
CoNLL dropped fast base→small (67.57→67.42) then nearly flattened small→extended
(67.42→67.40), while post-pruning recall kept climbing slightly (93.18%→93.35%→93.56%)
— recall and final linking quality were decoupling, and two points weren't enough to
tell "the CoNLL cost plateaus here" from "it keeps sliding, just slowly" apart. The
third point resolved it: CoNLL kept sliding (67.40→67.13 at context=64), while
post-pruning recall did **not** continue climbing (93.56%→93.36%, essentially flat/
slightly down) — see Finding 3 below for what this decoupling likely means.

---

## Findings

### 1. Span width is the dominant sub-axis by a wide margin

Span width spans a 7.08-point CoNLL range across its two conditions (62.20 → 69.28),
dwarfing pruning's 2.70-point range (64.87 → 67.57) and context window's 0.44-point
range (67.13 → 67.57, and all three context conditions sit *below* baseline). Of the
three factors this experiment isolates, mention candidate *coverage* (which spans even
exist to be scored at all) matters far more for Czech than either how aggressively
those candidates are pruned or how much surrounding context the encoder sees.

### 2. Extending span width doesn't just recover the baseline — it beats it

`train_exp3_width_extended` (max_span_width=46) reaches **69.28 CoNLL**, the best
result across all eight Experiment 3 conditions, beating baseline by +1.71 points with
post-pruning recall climbing from 93.18% to 95.21%. This confirms H1's directional
prediction (too-tight a cap hurts) and further suggests the system default (30) is
itself mildly conservative for Czech — long case-marked NPs and clausal mentions
apparently extend past 30 subtokens often enough to matter. This is a strong candidate
to carry into the thesis's final combined pipeline.

### 3. Recall gains only help when they close a structural coverage gap — not when they come from wider context

This is the sharpest contrast in the experiment. Width-extended's recall gain (+2.03
points post-pruning, from closing a hard structural gap — mentions literally couldn't
be represented before) converted cleanly into a CoNLL gain (+1.71). But the context
conditions' recall movement (93.18%→93.35%→93.56%→93.36%, a wobble of ~0.4 points,
never a hard gap being closed) tracked *no* corresponding CoNLL benefit — CoNLL instead
slid steadily downward (67.57→67.42→67.40→67.13). Recall alone doesn't predict final
performance; *why* recall moved matters. A likely confound for context specifically:
`max_segment_len` had to shrink to make room for the borrowed context (512→480→448→384
across small/extended/wide), which means documents get split into more, shorter
segments — and `max_training_sentences=6` truncates training examples by *segment
count*, not token count, so more/shorter segments mean a fixed 6-segment training
window now covers proportionally less of a long document. This is exactly the
truncation-interaction risk flagged in the design doc before any runs happened; the
downward CoNLL trend without a matching recall drop is consistent with that confound
biting harder as context (and the accompanying segment-length cut) grows, rather than
with "more context hurts coreference" on its own merits. The two effects aren't
separated by this experiment's design — a caveat worth stating plainly rather than
over-claiming "context extension doesn't help Czech coreference."

### 4. Pruning is asymmetric: tightening hurts structurally, loosening is close to free but not beneficial

Consistent with the eval-only sweep used to pick these values: tight (ratio=0.2) costs
2.70 CoNLL points via a real, structural drop in post-pruning recall (93.18%→83.43%).
Loose (ratio=0.6) costs only 0.16 points despite *raising* post-pruning recall to
94.38% — the extra surviving candidates are mostly low-quality distractors that don't
convert to correct links, echoing what the frozen-weight sweep predicted. Baseline's
existing `top_span_ratio=0.4` sits at or very near the local optimum for this axis.

### 5. Precision exceeds recall throughout, as in Experiment 1 and 2

Every one of the eight conditions shows MUC precision well above recall (e.g. base:
81.44 vs. 65.48; width-small: 78.32 vs. 61.27) — the system remains conservative
regardless of which mention-candidate knob is turned, consistent with the pattern
already noted across both prior experiments.

---

## Pending

Nothing outstanding. All eight conditions (base + 2 width + 2 pruning + 3 context) have
stable, reproducible results.
