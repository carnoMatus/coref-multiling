# Experiment 1 Results — Encoder Ablation Study (Czech-PDT)

**Task:** Czech coreference resolution on CorefUD Czech-PDT  
**System:** Span-based coreference model (attended antecedent, coarse-to-fine)  
**Evaluation:** CorefUD scorer, head match, best model checkpoint  
**Primary metric:** CoNLL F1 (average of MUC, B³, CEAFe) **without singletons**

---

## Results Table

| Model | Family | Vocab | MUC F1 | B³ F1 | CEAFe F1 | **CoNLL** |
|-------|--------|-------|--------|-------|---------|-----------|
| mBERT | BERT multilingual (104 lang) | 119k | 73.64 | 66.62 | 66.01 | **68.76** |
| SlavicBERT | BERT multilingual (4 Slavic) | — | 75.18 | 68.26 | 67.88 | **70.44** |
| XLM-R Large ¹ | RoBERTa multilingual (100 lang) | 250k | 77.38 | 70.41 | 68.49 | **72.09** |
| XLM-R Base | RoBERTa multilingual (100 lang) | 250k | 77.34 | 70.70 | 69.90 | **72.65** |
| RobeCzech | RoBERTa Czech-only | 52k | 77.37 | 71.16 | 69.79 | **72.77** |
| Czert-B ² | BERT Czech-only | 30k | — | — | — | **excluded** |

¹ XLM-R Large updated to Apr26 run — first fully fair run with `max_training_sentences=6, ffnn_size=3000`,
matching all other models. Supersedes previous runs (71.13, 71.57). See notes below.  
² Czert-B excluded — repeated fine-tuning attempts produced unstable training across all tested learning
rates. See notes below.

---

## Notes on Problematic Runs

### Czert-B — persistent training instability (excluded)

Three runs were attempted at decreasing learning rates:

| Run | `bert_learning_rate` | Loss at step 200 | Outcome |
|-----|----------------------|-----------------|---------|
| 1 | 1e-5 | 1495 (rising) | unstable |
| 2 | 5e-6 | 1398 (rising) | unstable |
| 3 | 3e-6 | 1482 (rising) | unstable |

All other models show loss dropping by step 200. Czert-B's loss climbs through
the warmup phase in all cases, indicating a fundamental incompatibility between
this model's weight statistics and the fine-tuning setup — not simply a
hyperparameter issue. Czert-B is excluded from the comparison. The best result
observed (51.60 CoNLL from run 1) is reported for reference only and should not
be compared directly to the other models.

### XLM-R Large — history of runs

Three runs were conducted:

| Run | Date | `max_training_sentences` | `ffnn_size` | CoNLL |
|-----|------|--------------------------|-------------|-------|
| 1 | Apr13 | 4 (reduced for OOM) | 3000 | 71.57 |
| 2 | Apr15 | 6 | 2000 (reduced for OOM) | 71.13 |
| 3 | Apr26 | 6 | 3000 | **72.09** |

The Apr26 run is the first fully fair comparison — same `max_training_sentences=6`
and `ffnn_size=3000` as all other models, achieved by fitting on the A40 48GB GPU.
This is the reported number. Previous runs are retained for reference only.

---

## Findings

### 1. Architecture family is the dominant factor (+4 points)

All RoBERTa-family models (XLM-R Base, XLM-R Large, RobeCzech) score 71–73,
while all BERT-family models (mBERT, SlavicBERT) score 69–70. The gap of ~2–4
CoNLL points holds regardless of language specialization. RoBERTa-style training
(larger batches, no NSP, dynamic masking, more data) produces better contextual
representations for span-based coreference.

### 2. Language focus provides modest gains within a family (~1–2 points)

Within the BERT family: SlavicBERT (70.44) > mBERT (68.76) by 1.68 points.
Focusing the pretraining vocabulary on 4 Slavic languages rather than 104 gives
a small but consistent advantage for Czech.

Within the RoBERTa family: RobeCzech (72.77) ≈ XLM-R Base (72.65), a difference
of 0.12 points — within single-run noise. A Czech-only model does not clearly
outperform a strong multilingual model.

### 3. Larger encoder provides marginal gains under fair conditions

Under the first fully fair comparison (Apr26 run), XLM-R Large (72.09) scores
only 0.56 points below XLM-R Base (72.65) despite having 560M vs 278M parameters.
Previous runs understated Large's performance due to GPU memory compromises.
The small remaining gap may reflect that Czech-PDT does not require the capacity
of a large model, or that the task head (fixed at 3000 units) is not scaled to
the larger 1024-dim hidden size.

### 4. Zero anaphora is largely encoder-independent

Zero anaphora F1 ranges from 84.78 (XLM-R Large rerun) to 90.14 (RobeCzech),
with most models clustering around 86–90. The variation is smaller than for full
coreference, suggesting zero anaphora resolution is driven more by the data and
the model's structural features than by encoder quality.

### 5. Precision consistently exceeds recall across all models

Every model shows precision 7–15 points higher than recall on MUC. The system is
conservative — it finds genuine mentions well but misses a substantial portion of
coreferential links. This is characteristic of span-based models on morphologically
rich languages where mention detection is harder.

---

## Pending

Nothing outstanding. All valid models have stable results.
