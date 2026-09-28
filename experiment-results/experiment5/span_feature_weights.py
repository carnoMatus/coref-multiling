""" Experiment 5, conditions 2-4: did the model learn to use the head-word one-hots?

Reads the first layer (W1) and output layer (w2) of the mention scorer (span_emb_score_ffnn)
and prints, for each appended one-hot input (the columns after the 2324 baseline span-emb
columns), the linearised effect w2 . W1[:, c] on the mention score, i.e. how much that tag on
the head pushes a span towards being kept as a mention (ignores the ReLU; direction and
relative size only). Untouched columns would keep the init norm 0.02 * sqrt(3000) ~= 1.10.

Usage: python experiment-results/experiment5/span_feature_weights.py <mode> <checkpoint> [<mode> <checkpoint> ...]
       mode in {pos, morph, combined}
"""
import sys
import math
import torch

sys.path.insert(0, '.')
import ling_features as lf

BASE = 3 * 768 + 20  # start, end, head_attn, width embedding
FEATS_NAMES = ['%s=%s' % (attr, value) for attr, values in lf.FEATS_VALUES.items() for value in values]
NAMES = {'pos': lf.UPOS_TAGS, 'morph': FEATS_NAMES, 'combined': lf.UPOS_TAGS + FEATS_NAMES}

if __name__ == '__main__':
    args = sys.argv[1:]
    for mode, path in zip(args[::2], args[1::2]):
        sd = torch.load(path, map_location='cpu')
        w1, w2 = sd['span_emb_score_ffnn.0.weight'], sd['span_emb_score_ffnn.3.weight'][0]
        names = NAMES[mode]
        assert w1.shape[1] == BASE + len(names), w1.shape
        norms = w1.norm(dim=0)
        effect = (w2 @ w1[:, BASE:]).tolist()
        print('== %s (%s)' % (mode, path))
        print('   column norms (init ~%.2f): span-emb mean %.2f, new features mean %.2f' % (
            0.02 * math.sqrt(3000), norms[:BASE].mean(), norms[BASE:].mean()))
        order = sorted(range(len(names)), key=lambda i: -effect[i])
        print('   linearised effect on mention score: ' + ', '.join('%s %+.1f' % (names[i], effect[i]) for i in order))
