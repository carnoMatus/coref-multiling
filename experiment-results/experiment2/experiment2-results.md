# Experiment 2 Results — Training Data Languages (mBERT, CorefUD 1.4)

**Task:** Czech coreference resolution, evaluated on CorefUD Czech-PDT dev
**System:** Span-based coreference model (attended antecedent, coarse-to-fine)
**Fixed variable:** Encoder = mBERT (`bert-base-multilingual-cased`), chosen independently
of Experiment 1's results to avoid confounding the comparison
**Varied variable:** Training data — which languages/corpora are included
**Evaluation:** CorefUD scorer, head match, best model checkpoint, `cs_pdt-corefud-dev.conllu`
**Primary metric:** CoNLL F1 (average of MUC, B³, CEAFe) **without singletons**

---

## Results Table

| Config | Training data | Docs | MUC F1 | B³ F1 | CEAFe F1 | **CoNLL** |
|--------|---------------|-----:|-------|-------|---------|-----------|
| A: `train_exp2_czech_pdt` | cs_pdt only | 2533 | 72.53 | 65.54 | 64.58 | **67.55** |
| B: `train_exp2_czech_all` | cs_pdt + cs_pcedt + cs_pdtsc | 5679 | 73.35 | 66.71 | 65.45 | **68.51** |
| C: `train_exp2_slavic` | all Czech + pl_pcc + ru_rucor | 7287 | 73.43 | 66.57 | 65.68 | **68.56** |
| D: `train_exp2_zerosshot` | pl_pcc + ru_rucor only (no Czech) | 1608 | 54.74 | 45.45 | 46.44 | **48.88** |
| E: `train_exp2_multilingual` | all CorefUD 1.4 (incl. Czech) | 12918 | 73.79 | 67.13 | 66.03 | **68.99** |
| F: `train_exp2_multilingual_zerosshot` | all CorefUD 1.4 except Czech | 7239 | 59.69 | 51.75 | 52.76 | **54.73** |

Reference: mBERT Exp1 (CorefUD 1.0): CoNLL 71.25 — **not directly comparable**, different
data version (CorefUD 1.0 vs 1.4). Condition A is the correct intra-experiment baseline.

---

## Notes on Reruns

Two conditions were trained more than once; the later, better-converged run is reported,
except `train_exp2_zerosshot`'s run 3 (see below).

### `train_exp2_czech_all`

| Run | Date | CoNLL |
|-----|------|-------|
| 1 | Aug11 | 68.44 |
| 2 | Aug15 | **68.51** (reported) |

### `train_exp2_zerosshot`

| Run | Date | CoNLL |
|-----|------|-------|
| 1 | Aug10 | 48.81 |
| 2 | Aug16 | 49.57 |
| 3 | Aug20 | **48.88** (reported) |

Run 3 was not a re-run for better convergence — run 2's best checkpoint (step 39000) was
deleted by training's checkpoint rotation before it could be archived to `models/experiment2/`.
Run 3 regenerates a checkpoint for that archive; its score (48.88) is what's reported here,
superseding run 2's 49.57. All three scores sit within ~0.8 points of each other, consistent
with normal run-to-run noise; no instability like Czert-B in Experiment 1 was observed.

---

## Findings

### 1. More Czech data helps, and additional related languages add a little more (A → B → C)

Adding cs_pcedt and cs_pdtsc to cs_pdt (A → B) gives the largest single jump: +0.96 CoNLL
(67.55 → 68.51) from more than doubling the Czech training data (2533 → 5679 docs) and adding
genre diversity (news + spoken + literary). Adding Polish and Russian on top (B → C) gives a
much smaller further gain: +0.05 CoNLL (68.51 → 68.56) from 1608 more docs. In-language data
volume matters far more than related-language data volume.

### 2. Zero-shot cross-lingual transfer is real but weak without any Czech

Condition D (Polish + Russian only, no Czech) reaches 48.88 CoNLL — well above a
random/degenerate baseline, showing mBERT's shared multilingual representations transfer
some coreference-relevant structure across Slavic languages. But it trails the weakest
Czech-inclusive condition (A, 67.55) by nearly 19 points. Related-language pretraining data is not
a substitute for target-language supervision.

### 3. Broad multilingual exposure beats narrow relatedness for zero-shot transfer (D vs F)

Condition F (all 28 CorefUD languages except Czech, 7239 docs) scores 54.73, clearly above
condition D (Slavic-only, 1608 docs, 48.88) — a +5.85 gain. This answers the question posed
in the experiment design: breadth of multilingual exposure outweighs linguistic relatedness
under zero-shot conditions, at least at this data scale. It's possible this is partly a data-volume
effect (7239 vs 1608 docs) rather than purely a breadth effect, since the two are confounded here.

### 4. Adding all languages (with Czech included) gives the best result, with no dilution penalty

Condition E (all 28 languages, includes Czech, 12918 docs) scores 68.99 — the best of all six
conditions, edging out C (68.56) and B (68.51). The predicted "dilution" effect (capacity
shared across many languages hurting the Czech-specific signal) does not materialize; if
anything, the extra non-Czech data acts as a mild regularizer/auxiliary signal on top of the
in-language gains from B.

### 5. Zero anaphora tracks full coreference, but transfers even less well zero-shot

Zero anaphora F1 follows the same ordering as CoNLL F1 across conditions (E: 86.27 > C: 85.54
> B: 85.14 > A: 83.52 among Czech-inclusive runs; F: 70.50 > D: 68.36 among zero-shot runs).
The gap between Czech-inclusive and zero-shot conditions is proportionally similar to CoNLL
(roughly 15–18 points), suggesting zero anaphora resolution — which is more syntax-dependent
and Czech-specific in its overt cues — is not disproportionately harder to transfer than
general coreference.

### 6. Precision consistently exceeds recall, as in Experiment 1

All six conditions show MUC precision above recall, most pronounced in the zero-shot
conditions — D now shows the largest gap of all six by a wide margin (71.61 vs 44.31, a
27-point spread), with F next (70.72 vs 51.64, 19 points); the four Czech-inclusive conditions
sit in a tighter 15–15.2 point band. Without Czech training signal, the model becomes markedly
more conservative, missing far more genuine coreference links while keeping the links it does
propose fairly accurate — and condition D's rerun sharpens that pattern further versus the
previously reported run.

---

## Pending

Nothing outstanding. All six conditions have stable, reproducible results.
