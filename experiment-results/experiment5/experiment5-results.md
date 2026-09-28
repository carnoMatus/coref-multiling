# Experiment 5 Results — Linguistic Feature Enrichment (mBERT, CorefUD 1.4)

**Task:** Czech coreference resolution, evaluated on CorefUD Czech-PDT dev
**System:** Span-based coreference model (attended antecedent, coarse-to-fine)
**Fixed variables:** Encoder = mBERT (`bert-base-multilingual-cased`), training data =
Czech PDT only (CorefUD 1.4), all baseline hyperparameters
**Varied variable:** Gold morphological information of the mention's syntactic head —
per-span UPOS / FEATS one-hots (conditions 2–4) or pairwise Gender/Number/Person
agreement (condition 5). Implementation and all design decisions:
`experiment-runbooks/experiment5/experiment5-implementation.md`
**Evaluation:** CorefUD scorer, head match, best model checkpoint, `cs_pdt-corefud-dev.conllu`
**Primary metric:** CoNLL F1 (average of MUC, B³, CEAFe) **without singletons**

---

## Results Table

| # | Condition | Config | MUC F1 | B³ F1 | CEAFe F1 | **CoNLL** | Zeros F1 |
|---|-----------|--------|--------|-------|----------|-----------|----------|
| 1 | Baseline  | `train_exp5_baseline` |  72.61 | 65.45 | 64.22 | **67.42** | 84.40 |
| 2 | POS-only | `train_exp5_pos` | 72.39 | 65.34 | 64.04 | **67.26** | 83.63 |
| 3 | Morph-only | `train_exp5_morph` | 72.49 | 65.43 | 64.60 | **67.51** | 83.89 |
| 4 | Combined | `train_exp5_combined` | 72.62 | 65.58 | 64.41 | **67.54** | 85.26 |
| 5 | Pairwise agreement | `train_exp5_agreement` | 72.58 | 65.67 | 64.41 | **67.55** | 84.17 |

Candidate-stage recall is 97.04% (20142/20757) for every condition (it depends only on
`max_span_width`). Run-to-run noise reference: two independent runs of the identical
baseline config (Exp 2 condition A and `train_exp3_base`) scored 67.55 and 67.57.

All scores are the official scorer's result for the **best checkpoint** ("Best model
evaluation" section of the log); each CoNLL value equals the mean of its MUC/B³/CEAFe.
Careful when grepping: the first "CoNLL score" in a log belongs to the *final* checkpoint
(pos 67.21, morph 67.27, combined 67.24, agreement 67.06), not the reported best one.

| Condition | Best checkpoint (step) | Internal max F1 | Log |
|---|---|---|---|
| Baseline (`train_exp3_base`) | 108000 | 60.40 | `data/train_exp3_base/log_Aug23_11-28-31.txt` |
| POS-only | 109000 | 60.14 | `data/train_exp5_pos/log_Sep25_09-41-14.txt` |
| Morph-only | 110000 | 60.16 | `data/train_exp5_morph/log_Sep25_18-26-08.txt` |
| Combined | 102000 | 60.36 | `data/train_exp5_combined/log_Sep26_03-30-26.txt` |
| Agreement | 113000 | 60.32 | `data/train_exp5_agreement/log_Sep24_22-02-09.txt` |

---

## Implementation verification

Because no condition moved the score, the implementation was checked independently
(`verify_features.py`, output `verify_features.txt`; checkpoint shapes checked directly):

- **Features reach the model.** The span scorer's input width is 2324 + 17 / +16 / +33
  for POS / morph / combined, and the coarse and pairwise scorers grow accordingly.
- **Preprocessing is exact.** Re-parsing the dev CoNLL-U without udapi: 90508/90508 words
  have the correct UPOS/FEATS bits, 82926/82926 dependency parents point at the correct word.
- **The training cache is consistent.** After truncation to 6 segments, all 1144054 parent
  pointers inside the window still land on the first subtoken of a word; every one of the
  136984 training gold mentions (≤ 30 subtokens) gets a head with a UPOS tag, 94% also with
  FEATS. Gold head UPOS distribution: NOUN 50%, PRON 16%, PROPN 13%, DET 10%, ADJ 6%.
- **The models use the features** (see the weight analyses below).

---

## Head-ambiguity diagnostics (best checkpoint)

| Level | Condition 5 |
|---|---|
| All candidate spans | 60.23% (1905954/3164307) |
| Top spans after pruning | 14.57% (9992/68589) |
| Gold mentions | 0.61% (122/20142) |

Top-span ambiguity fell during training (27.05% at the first evaluation → ~14.5%) as the
model learned to keep fewer spans that cut across the dependency tree. About a seventh
of the spans that reach antecedent scoring are not syntactic subtrees, so their head
features depend on the tie-breaking rule.

---

## Conditions 2–4: no gain from per-mention tags

CoNLL 67.26 / 67.51 / 67.54 against the baseline's 67.57. Morph-only and combined are
within run-to-run noise; POS-only is 0.31 lower, which one run per condition cannot
separate from noise (the baseline rerun `train_exp5_baseline` adds a noise data point).
Post-pruning mention recall is unchanged (93.30–93.37% vs. 93.18%), so the tags do not
help mention detection either.

### Did the model use the features? — weight analysis

Script: `span_feature_weights.py`; raw output: `span_feature_weights.txt`. Same method as
for condition 5 below, applied to the mention scorer (`span_emb_score_ffnn`): the
linearised effect of each head tag on the mention score of a span.

Column norms of the new inputs: 4.43 (POS), 3.27 (morph), 3.57 (combined), against 2.3 for
the baseline span-representation inputs and ≈1.10 for untouched columns. The model uses
the tags, and in a linguistically sensible way. POS-only, from most to least favoured head:

- strongly favoured: DET +12.2, PRON +10.3 (pronouns, possessives, demonstratives);
- mildly favoured: ADJ +3.9, NOUN +2.1, PROPN +0.5;
- strongly penalised: VERB −12.1, CCONJ −20.0, PART −20.5, SCONJ −28.2, AUX −31.8, ADP −41.2.

Spans headed by function words are pushed out of the mention candidates. Pronouns and
determiners get the largest boost, probably because they are short and easy to miss
otherwise, while nouns are already recognised from the text. In the morph model, having
any case / person value raises the score (it marks a nominal head), most of all
Case=Nom +8.9 and Person=3 +8.6. The combined model learns the same pattern.

As with condition 5, the features are **used but redundant**: mention detection and the
final score stay the same, so the encoder already provides this information. mBERT sees
the word forms, and the POS and morphology of a Czech word are largely visible in its form.

## Condition 5: no effect on the score

CoNLL 67.55 vs. 67.57 (−0.02), with every component metric within ±0.05 and zero anaphora
−0.56. This is inside run-to-run noise: explicit agreement information **does not change**
coreference performance.

### Did the model use the feature? — weight analysis

Script: `experiment-results/experiment5/agreement_weights.py`; raw output:
`agreement_weights.txt`. It reads the first layer `W1` (3000 × 7061) and output layer
`w2` of the fine-grained pairwise scorer; the agreement inputs are the last 9 of the 7061
pair-representation columns. The "linearised effect" of an input is `w2 · W1[:, c]`: how
much switching that one-hot on would move the pairwise score, ignoring the ReLU between
the layers (a guide to direction and relative size, not an exact value).

Best checkpoint (step 113000):

| Attribute | agree | disagree | unknown | spread (max − min) |
|---|---|---|---|---|
| Gender | +6.93 | −7.22 | −0.51 | 14.2 |
| Number | +7.97 | −7.50 | −4.65 | 15.5 |
| Person | +6.06 | **−17.78** | +2.50 | 23.8 |
| *Antecedent distance (10 buckets), for comparison* | | | | *26.6* |

Column norms of `W1`: agreement inputs 5.06 on average, antecedent-distance inputs 4.22,
span-representation inputs 2.43; untouched columns would stay at the initial ≈1.10
(task parameters have no weight decay). The weights grew steadily across checkpoints
(agreement column norm 1.44 → 2.66 → 4.17 → 5.06 at steps 1k / 10k / 44k / 113k).

**Reading:** the model learned to use the feature, strongly and in the linguistically
expected direction: agreement raises the score of a mention–antecedent pair, disagreement
lowers it, and Person disagreement (e.g. a 1st-person pronoun vs. a 3rd-person
antecedent) is the strongest penalty. Its overall influence is of the same order as
the antecedent-distance feature.

Since the feature is used but F1 does not move, the information is **redundant rather
than ignored**: the model gets the same decisions without it. The most likely source is
mBERT's subword representation. Czech marks gender, number and case in word endings,
which mBERT sees, and the pair representation's `g_i ∘ g_j` term lets the scorer compare
them. With the explicit flags, the scorer takes this signal from the flags instead of
from the embeddings, and the decisions stay the same. Further factors: the feature only
re-ranks the top 50 antecedents chosen by the coarse scorer, which never sees agreement,
and Person is "unknown" for every noun head.

Caveat: the weight analysis shows that the feature is *used*, not what the model would
lose without it. A direct check would be an evaluation-time ablation (set all agreement
inputs to "unknown" on the trained model); it was not run.
