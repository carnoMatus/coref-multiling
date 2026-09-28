# Experiment 4 Results — Training Regime Sensitivity and the Noise Floor (mBERT, CorefUD 1.4)

**Task:** Czech coreference resolution, evaluated on CorefUD Czech-PDT dev
**System:** Span-based coreference model (attended antecedent, coarse-to-fine)
**Fixed variables:** Encoder = mBERT (`bert-base-multilingual-cased`), training data =
Czech PDT only (CorefUD 1.4, same as Experiment 2 condition A / `train_exp3_base`),
mention-candidate settings and architecture at Experiment 3 baseline values
**Varied variable:** Optimisation hyperparameters — encoder learning rate, task-head
learning rate, effective batch size (gradient accumulation) — each varied independently
from the inherited baseline (`bert_learning_rate=1e-5`, `task_learning_rate=2e-4`,
`gradient_accumulation_steps=1`)
**Evaluation:** CorefUD scorer, head match, best model checkpoint, `cs_pdt-corefud-dev.conllu`
**Primary metric:** CoNLL F1 (average of MUC, B³, CEAFe) **without singletons**

**Status:** Part B (5 conditions) complete. **Part A — the four unseeded baseline
replicas that define the noise floor δ — has not been run.** Part C (convergence
analysis from logs already on disk) has not been written up. See
`experiment-runbooks/experiment4/` for the full design; §7 there is explicit that δ is
this experiment's precondition for interpreting any of the numbers below.

---

## Results Table

| Condition | Axis | Value | Ratio (head:encoder) | MUC F1 | B³ F1 | CEAFe F1 | zero F1 | **CoNLL** |
|---|---|---|---|---|---|---|---|---|
| Baseline | — | bert_lr=1e-5, task_lr=2e-4, grad_accum=1 | 20× | 72.37 | 65.48 | 64.35 | 84.61 | **67.40** |
| blr_low | encoder LR | 5e-6 | 40× | 71.94 | 64.79 | 63.43 | 84.26 | **66.72** |
| blr_high | encoder LR | 2e-5 | 10× | 72.66 | 65.57 | 64.53 | 84.46 | **67.59** |
| tlr_low | task LR | 5e-5 | 5× | 72.63 | 65.79 | 64.70 | 84.00 | **67.71** |
| tlr_high | task LR | 8e-4 | 80× | 72.94 | 65.98 | 64.88 | 84.25 | **67.93** |
| accum4 | grad accum | 4 | 20× | 71.84 | 64.62 | 63.76 | 84.12 | **66.74** |

Final-checkpoint CoNLL (the "Test eval" block, separating end-state variance from
best-checkpoint selection variance per the runbook): baseline 67.33, blr_low 66.19,
blr_high 67.61, tlr_low 67.43, tlr_high 67.72, accum4 66.54. All five Part B conditions
land within ~1.3 points of their own best-ckpt score on final-ckpt, the same order of
magnitude as the gaps between conditions themselves.

---

## Notes

### No noise floor measured — Part A was not run

None of the deltas above have been tested against δ, because δ requires the four
identical-configuration replicas (`train_exp4_rep1`–`rep4`) that Part A defines, and
those runs were skipped in favor of Part B. Without δ, "blr_high is +0.02 over baseline"
and "blr_low is −0.85" cannot be distinguished from "these are both within normal
run-to-run spread" — the same uncertainty this experiment exists to resolve for
Experiments 1–3. The findings below are reported as directional and descriptive, not as
established effects, consistent with the runbook's own framing (§6a, §7).

### Config inheritance deviation from the runbook

The runbook (`experiment4-runbook.md` §2) specifies all Experiment 4 conditions as
`${train_exp3_base}{...}`. As actually written in `experiments.conf`, `train_exp4_blr_low`,
`blr_high`, `tlr_low`, `tlr_high`, and `accum4` instead inherit from
`${train_exp2_czech_pdt}{...}` directly. The only substantive difference between the two
parents is `log_mention_recall_diagnostics` (`true` on `train_exp3_base`, unset/`false`
on `train_exp2_czech_pdt`) — `data_dir` and every mention-candidate setting are identical
between them, so the reported CoNLL/MUC/B³/CEAFe numbers are unaffected. The practical
consequence is that none of the five Part B logs contain "Mention recall diagnostics"
lines, unlike `train_exp3_base` and the Experiment 3 conditions, so candidate-stage and
post-pruning recall can't be reported per condition here the way they were in the
Experiment 3 results table.

---

## Findings

### 1. The task-head learning rate is the one axis where both tested points beat baseline

`tlr_low` (5e-5, ratio 5×) reaches 67.71 and `tlr_high` (8e-4, ratio 80×) reaches 67.93 —
the best result of all five Part B conditions, +0.36 over baseline. Moving the head
learning rate in *either* direction from the inherited 2e-4 improved on baseline, which
is a different shape of result than a simple "higher is better" or "lower is better"
axis: it suggests the inherited value may not sit at a local optimum for mBERT on Czech,
consistent with H3's premise that the head (trained from scratch) and the encoder
(fine-tuned) have no obvious reason to share a well-tuned ratio inherited from an English
OntoNotes setup. Whether either point clears δ is unknown.

### 2. The encoder learning rate is asymmetric: halving it costs far more than doubling it

`blr_high` (2e-5) lands at 67.59, essentially matching baseline (+0.02). `blr_low` (5e-6)
drops to 66.72 (−0.85), the second-largest movement in either direction across all five
conditions. This is directionally consistent with the design doc's grounding: the
conventional BERT fine-tuning range starts around 2e-5, so the inherited 1e-5 already
sits below it, and pushing further down (5e-6) moves toward the range where Czert-B
struggled in Experiment 1, while pushing up toward 2e-5 costs nothing measurable. If this
holds up against δ, it would support H2 only in the "raising it is safe" direction, not
in the stronger "1e-5 is already optimal" sense.

### 3. Raising the effective batch size costs almost as much as halving the encoder LR

`accum4` (gradient_accumulation_steps=4) drops to 66.74 (−0.83), matching `blr_low`'s
magnitude of loss. This is consistent with H4's prediction: at a fixed 50-epoch budget,
quadrupling accumulation quarters the number of optimiser updates
(`total_update_steps = len(examples_train) * epochs // grad_accum`), and the run's own
best checkpoint arrived at step 28,000 versus baseline's 108,000 — a much shorter
optimisation trajectory. As the design doc states up front, this is a confounded
manipulation ("larger batch at fixed epoch budget," not "larger batch, all else equal"),
so the size of the drop shouldn't be read as a clean batch-size effect.

### 4. The ratio reading doesn't explain the pattern — absolute learning rate does

The five conditions sit on the ratio ladder 5× / 10× / 20× / 40× / 80×, and if the
encoder:head ratio alone drove performance, the two most extreme ratios (`blr_low` at
40× and `tlr_high` at 80×) should move the same direction. They don't: `blr_low` is the
second-worst condition (−0.85) and `tlr_high` is the best (+0.36). What tracks the
outcome instead is which absolute value moved — a too-low absolute encoder LR hurts
regardless of the resulting ratio, and a higher absolute head LR helps regardless of
the resulting ratio. This matches the design doc's own caveat that the ratio reading is
"suggestive, not a controlled ratio experiment," and the data here reinforces that
caution rather than rescuing the ratio story.

### 5. Precision exceeds recall throughout, consistent with Experiments 1–3

Every condition's MUC score shows precision well above recall (e.g. baseline 82.01 vs.
65.09; accum4 80.02 vs. 65.19; tlr_high 81.36 vs. 66.09) — the system stays conservative
regardless of which optimisation knob moves, extending the pattern already noted in
Experiments 1, 2, and 3.

---

## Pending

- **Part A (4 baseline replicas) — not run.** This blocks everything above from being
  read as anything more than descriptive. It is the single highest-priority remaining
  piece of Experiment 4: without it, δ doesn't exist, the retro-validation table in the
  runbook (§6b) can't be filled in, and the Final Combined Configuration section has no
  threshold to test its improvement claim against.
- **Part C (convergence analysis) — not written up.** Zero additional GPU cost; the data
  (`Eval_Avg_F1` per 1000 steps) already exists in every log used above, including the
  five Part B logs, which were not part of the original Part C scope but can now be
  included.
- **`[DECIDE]` in the runbook (§7)** — whether Experiment 4 must nominate a "best
  setting" — is still open. On the current numbers alone (pending δ), `tlr_high` would be
  the nominee.
