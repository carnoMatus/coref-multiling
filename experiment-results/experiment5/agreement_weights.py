""" Experiment 5, condition 5: did the model learn to use the agreement inputs?

Reads the first layer (W1) and output layer (w2) of the fine-grained pairwise scorer
(coref_score_ffnn) from saved checkpoints and reports, for the 9 agreement one-hot inputs
(last 9 columns of the pair representation) compared with the existing antecedent-distance
feature:

  1. column norms ||W1[:, c]|| -- untouched columns stay at the init value
     0.02 * sqrt(3000) ~= 1.10 (task parameters use no weight decay);
  2. linearised score effect w2 . W1[:, c] of each agreement state, and the spread
     (max - min) per attribute, next to the spread over the 10 antecedent-distance buckets.
     This ignores the ReLU between the layers, so it is an indication of direction and
     relative size, not the exact change in the pairwise score.

Usage: python experiment-results/experiment5/agreement_weights.py <log_dir> <run_suffix> <step> [<step> ...]
"""
import sys
import math
import torch

SPAN_EMB = 3 * 768 + 20  # start, end, head_attn, width embedding
PAIR_FEATURES = ['same_speaker', 'genre', 'segment_distance', 'antecedent_distance']  # 20 dims each, in model.py order
FEATURE_START = 3 * SPAN_EMB
DIST_START = FEATURE_START + 20 * PAIR_FEATURES.index('antecedent_distance')
AGREE_START = FEATURE_START + 20 * len(PAIR_FEATURES)
ATTRS, STATES = ['Gender', 'Number', 'Person'], ['agree', 'disagree', 'unknown']
INIT_NORM = 0.02 * math.sqrt(3000)


def analyse(path):
    sd = torch.load(path, map_location='cpu')
    w1, w2 = sd['coref_score_ffnn.0.weight'], sd['coref_score_ffnn.3.weight'][0]
    assert w1.shape[1] == AGREE_START + 9, w1.shape
    col_norm = w1.norm(dim=0)
    lines = ['column norms (init ~%.2f): span-emb mean %.2f | antecedent-distance mean %.2f | agreement mean %.2f' % (
        INIT_NORM, col_norm[:FEATURE_START].mean(), col_norm[DIST_START:DIST_START + 20].mean(),
        col_norm[AGREE_START:].mean())]
    dist_effect = w2 @ (w1[:, DIST_START:DIST_START + 20] @ sd['emb_top_antecedent_distance.weight'].T)  # [10 buckets]
    lines.append('antecedent distance, linearised effect per bucket: %s  (spread %.3f)' % (
        ' '.join('%.3f' % e for e in dist_effect.tolist()), (dist_effect.max() - dist_effect.min()).item()))
    for i, attr in enumerate(ATTRS):
        cols = range(AGREE_START + 3 * i, AGREE_START + 3 * i + 3)
        effect = [(w2 @ w1[:, c]).item() for c in cols]
        norms = [col_norm[c].item() for c in cols]
        lines.append('%-7s linearised effect %s  (spread %.3f); column norms %s' % (
            attr, '  '.join('%s %+.3f' % (s, e) for s, e in zip(STATES, effect)), max(effect) - min(effect),
            ' '.join('%.2f' % n for n in norms)))
    return lines


if __name__ == '__main__':
    log_dir, suffix, steps = sys.argv[1], sys.argv[2], sys.argv[3:]
    for step in steps:
        path = '%s/model_%s_%s.bin' % (log_dir, suffix, step)
        print('== %s' % path)
        for line in analyse(path):
            print('  ' + line)
