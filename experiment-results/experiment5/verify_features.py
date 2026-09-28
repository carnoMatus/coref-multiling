""" Experiment 5: independent check that the linguistic features the model saw are correct.

1. Preprocessing: re-parses the dev CoNLL-U without udapi and compares, for every word, the
   stored UPOS/FEATS bits and the dependency parent with the "ling" field of the jsonlines.
2. Training cache: for every (truncated) training and dev document in the tensor cache, checks
   that parent pointers still land on the first subtoken of a word after truncation, and that
   gold mentions get a head with non-zero tags.

Usage: python experiment-results/experiment5/verify_features.py [data_dir]   (default ./data/exp5)
"""
import sys
import json
import pickle
import collections

sys.path.insert(0, '.')
import ling_features as lf


def read_conllu_docs(path):
    """ docs -> sentences -> CoNLL-U columns of words (incl. empty nodes, excl. multiword ranges) """
    docs, sentence = [], []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('# newdoc'):
            docs.append([])
        elif line.startswith('#'):
            continue
        elif not line:
            if sentence:
                docs[-1].append(sentence)
            sentence = []
        else:
            columns = line.split('\t')
            if '-' not in columns[0]:
                sentence.append(columns)
    if sentence:
        docs[-1].append(sentence)
    return docs


def check_preprocessing(data_dir):
    docs = read_conllu_docs(f'{data_dir}/cs_pdt-corefud-dev.conllu')
    with open(f'{data_dir}/cs_pdt-corefud-dev.512.jsonlines') as f:
        jsonl = [json.loads(line) for line in f]
    assert len(jsonl) == len(docs), (len(jsonl), len(docs))
    stats = collections.Counter()
    for doc, sentences in zip(jsonl, docs):
        words = [(si, columns) for si, sentence in enumerate(sentences) for columns in sentence]
        ling = doc['ling']
        first = {}
        for pos, row in enumerate(ling):
            if row[0] >= 0:
                first.setdefault(row[0], pos)
        assert len(first) == len(words), (doc['doc_key'], len(first), len(words))
        word_idx = {(si, columns[0]): i for i, (si, columns) in enumerate(words)}
        for i, (si, columns) in enumerate(words):
            row = ling[first[i]]
            stats['words'] += 1
            stats['tags_ok'] += row[lf.UPOS_OFFSET:] == lf.encode_tags(columns[3], columns[5])
            if columns[6] not in ('_', '0'):  # regular word attached to another word
                stats['parents_checked'] += 1
                stats['parents_ok'] += ling[row[1]][0] == word_idx[(si, columns[6])]
    print('dev preprocessing:', dict(stats))


def check_cache(data_dir):
    with open(f'{data_dir}/cached.tensors.cs_pdt-corefud.512.6.bin', 'rb') as f:
        samples, _ = pickle.load(f)
    for split in ['trn', 'dev']:
        stats, head_upos = collections.Counter(), collections.Counter()
        for _, tensors in samples[split]:
            ling, gold_starts, gold_ends = tensors[10], tensors[7], tensors[8]
            n = ling.shape[0]
            stats['docs'] += 1
            parent_first, parent_last = ling[:, 1], ling[:, 2]
            inside = (parent_first >= 0) & (parent_first < n)
            pf, pl = parent_first[inside], parent_last[inside]
            consistent = (ling[pf, 0] >= 0) & (ling[pf, 0] == ling[pl, 0]) & (ling[(pf - 1).clamp(min=0), 0] != ling[pf, 0])
            stats['parents_in_window'] += inside.sum().item()
            stats['parents_consistent'] += consistent.sum().item()
            keep = (gold_ends - gold_starts) < 30
            head_pos, has_head, _, _ = lf.find_span_heads(ling, gold_starts[keep], gold_ends[keep], 30)
            upos = ling[head_pos, lf.UPOS_OFFSET:lf.FEATS_OFFSET]
            stats['gold'] += keep.sum().item()
            stats['gold_has_head'] += has_head.sum().item()
            stats['gold_head_with_upos'] += (upos.sum(1) > 0).sum().item()
            stats['gold_head_with_feats'] += (ling[head_pos, lf.FEATS_OFFSET:].sum(1) > 0).sum().item()
            for i in upos.argmax(1)[upos.sum(1) > 0].tolist():
                head_upos[lf.UPOS_TAGS[i]] += 1
        print(f'{split} cache:', dict(stats))
        print('   gold head UPOS:', head_upos.most_common(8))


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './data/exp5'
    check_preprocessing(data_dir)
    check_cache(data_dir)
