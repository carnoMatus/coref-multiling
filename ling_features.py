""" Experiment 5 (linguistic feature enrichment): gold UPOS/FEATS of a span's syntactic head.

Per-subtoken layout of the `ling` array (one row per subtoken of the flattened segments,
CLS/SEP included, same indexing as sentence_map / mention_doc in model.py):

    col 0          word id of the subtoken's word (-1 for CLS/SEP)
    col 1, 2       first/last subtoken position of the word's dependency parent
                   (NO_PARENT when the parent is the root or cannot be resolved)
    col 3          tree depth of the word (EMPTY_DEPTH for empty nodes)
    cols 4..20     UPOS one-hot (17)
    cols 21..36    FEATS: Gender(3) Number(3) Case(7) Person(3), multi-hot within an attribute
                   when the gold value is multi-valued (e.g. Gender=Fem,Neut)

Head of a span = among the words overlapping the span, those whose parent lies outside the
span qualify; the one with the smallest depth wins, ties go to the leftmost. Empty nodes and then
punctuation come last, whatever their depth. More than one
qualifying word means the span is "head-ambiguous" (see log_head_ambiguity).
"""
import sys
import json
import collections
import numpy as np

UPOS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART',
             'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
FEATS_VALUES = collections.OrderedDict([
    ('Gender', ['Masc', 'Fem', 'Neut']),
    ('Number', ['Sing', 'Plur', 'Dual']),
    ('Case', ['Nom', 'Gen', 'Dat', 'Acc', 'Voc', 'Loc', 'Ins']),
    ('Person', ['1', '2', '3']),
])

COL_WORD, COL_PARENT_FIRST, COL_PARENT_LAST, COL_DEPTH = 0, 1, 2, 3
UPOS_OFFSET = 4
UPOS_DIM = len(UPOS_TAGS)
FEATS_OFFSET = UPOS_OFFSET + UPOS_DIM
FEATS_DIM = sum(len(v) for v in FEATS_VALUES.values())
NUM_COLS = FEATS_OFFSET + FEATS_DIM

# Slices of each FEATS attribute *within* the FEATS block
FEATS_SLICES = {}
_i = 0
for _attr, _vals in FEATS_VALUES.items():
    FEATS_SLICES[_attr] = slice(_i, _i + len(_vals))
    _i += len(_vals)

AGREEMENT_ATTRS = ['Gender', 'Number', 'Person']  # Condition 5; Case deliberately excluded
AGREEMENT_DIM = 3 * len(AGREEMENT_ATTRS)  # agree / disagree / unknown one-hot per attribute

NO_PARENT = -10 ** 6  # parent range that never overlaps any span, also after truncation offsets
EMPTY_DEPTH = 1000  # empty nodes lose every tie against real words
PUNCT_COL = UPOS_OFFSET + UPOS_TAGS.index('PUNCT')

SPAN_FEATURE_COLS = {
    'none': None,
    'upos': (UPOS_OFFSET, UPOS_OFFSET + UPOS_DIM),
    'morph': (FEATS_OFFSET, FEATS_OFFSET + FEATS_DIM),
    'both': (UPOS_OFFSET, FEATS_OFFSET + FEATS_DIM),
}


def span_feature_dim(mode):
    cols = SPAN_FEATURE_COLS[mode]
    return 0 if cols is None else cols[1] - cols[0]


# ---------------------------------------------------------------------------
# Preprocessing side (udapi nodes -> per-subtoken rows)
# ---------------------------------------------------------------------------

def encode_tags(upos, feats_str, unknown_counter=None):
    """ One-hot UPOS + per-attribute (multi-)hot FEATS; missing values stay all-zero. """
    row = [0] * (UPOS_DIM + FEATS_DIM)
    if upos in UPOS_TAGS:
        row[UPOS_TAGS.index(upos)] = 1
    elif upos not in ('_', '') and unknown_counter is not None:
        unknown_counter['UPOS=' + upos] += 1
    if feats_str and feats_str != '_':
        for kv in feats_str.split('|'):
            if '=' not in kv:
                continue
            key, values = kv.split('=', 1)
            if key not in FEATS_VALUES:
                continue  # also skips layered keys such as Gender[psor]
            for value in values.split(','):
                if value in FEATS_VALUES[key]:
                    row[UPOS_DIM + FEATS_SLICES[key].start + FEATS_VALUES[key].index(value)] = 1
                elif unknown_counter is not None:
                    unknown_counter[key + '=' + value] += 1
    return row


def _node_parent(node):
    """ Basic-tree parent for regular words, first enhanced-deps parent for empty nodes. """
    if node.is_empty():
        try:
            deps = node.deps
            parent = deps[0]['parent'] if deps else None
        except Exception:
            parent = None
    else:
        parent = node.parent
    if parent is None or parent.is_root():
        return None
    return parent


def _node_depth(node):
    if node.is_empty():
        return EMPTY_DEPTH
    depth, parent = 1, node.parent
    while not parent.is_root():
        depth += 1
        parent = parent.parent
    return depth


def build_ling_rows(nodes, segments, subtoken_map, stats=None):
    """ nodes: udapi words in document order (== DocumentState.tokens order);
    segments: list of subtoken segments with CLS/SEP; subtoken_map: word id per subtoken
    WITHOUT CLS/SEP (DocumentState.subtoken_map). Returns list of NUM_COLS-long int rows. """
    stats = stats if stats is not None else collections.Counter()
    node_to_word = {id(node): idx for idx, node in enumerate(nodes)}

    # Word id for each flattened position (CLS/SEP -> -1)
    flat_words, sub_idx = [], 0
    for segment in segments:
        flat_words.append(-1)
        for _ in range(len(segment) - 2):
            flat_words.append(subtoken_map[sub_idx])
            sub_idx += 1
        flat_words.append(-1)
    assert sub_idx == len(subtoken_map)

    word_first, word_last = {}, {}
    for pos, w in enumerate(flat_words):
        if w >= 0:
            word_first.setdefault(w, pos)
            word_last[w] = pos

    word_rows = []
    for node in nodes:
        parent = _node_parent(node)
        parent_word = node_to_word.get(id(parent)) if parent is not None else None
        if node.is_empty():
            stats['empty_nodes'] += 1
            stats['empty_nodes_with_parent'] += parent_word is not None
        if parent_word is None:
            parent_first = parent_last = NO_PARENT
        else:
            parent_first, parent_last = word_first[parent_word], word_last[parent_word]
        tags = encode_tags(node.upos, str(node.feats), stats)
        word_rows.append([parent_first, parent_last, _node_depth(node)] + tags)

    empty_row = [-1, NO_PARENT, NO_PARENT, 0] + [0] * (UPOS_DIM + FEATS_DIM)
    return [empty_row if w < 0 else [w] + word_rows[w] for w in flat_words]


# ---------------------------------------------------------------------------
# Model side (torch)
# ---------------------------------------------------------------------------

def find_span_heads(ling, starts, ends, max_width):
    """ ling: [num subtokens, NUM_COLS] long tensor; starts/ends: [num spans] inclusive,
    ends - starts < max_width. Returns
        head_pos   [num spans]  subtoken position of the chosen head (valid only if has_head)
        has_head   [num spans]  False for spans covering no word (e.g. only CLS/SEP)
        num_qual   [num spans]  number of qualifying words (> 1 means head-ambiguous)
        qualifies  [num spans, max_width]  qualifying word start positions (offset from start)
    """
    import torch
    num_subtokens = ling.shape[0]
    offsets = torch.arange(max_width, device=ling.device)
    pos = starts.unsqueeze(1) + offsets.unsqueeze(0)  # [num spans, max width]
    in_span = pos <= ends.unsqueeze(1)
    pos_c = torch.clamp(pos, max=num_subtokens - 1)
    word = ling[pos_c, COL_WORD]
    prev_word = ling[torch.clamp(pos_c - 1, min=0), COL_WORD]
    word_starts_here = (offsets == 0).unsqueeze(0) | (word != prev_word)  # count each word once
    parent_inside = (ling[pos_c, COL_PARENT_FIRST] <= ends.unsqueeze(1)) & \
                    (ling[pos_c, COL_PARENT_LAST] >= starts.unsqueeze(1))
    qualifies = in_span & (word >= 0) & word_starts_here & torch.logical_not(parent_inside)
    num_qual = qualifies.sum(dim=1)

    # Punctuation attaches high in UD trees, so it only wins when nothing else qualifies (after empty nodes too)
    depth = ling[pos_c, COL_DEPTH]
    depth = torch.where(ling[pos_c, PUNCT_COL] > 0, depth + EMPTY_DEPTH, depth)
    key = depth * max_width + offsets.unsqueeze(0)  # smallest depth, then leftmost
    key = torch.where(qualifies, key, torch.full_like(key, np.iinfo(np.int64).max))
    head_offset = torch.argmin(key, dim=1)
    head_pos = starts + head_offset
    return head_pos, num_qual > 0, num_qual, qualifies


def ambiguous_with_different_tags(ling, starts, qualifies, head_pos):
    """ Among head-ambiguous spans: does any qualifying word carry different UPOS/FEATS than the
    chosen head, i.e. did the tie-breaker change the feature vector? [num spans] bool """
    import torch
    max_width = qualifies.shape[1]
    num_subtokens = ling.shape[0]
    # Pack each subtoken's 33 tag bits into one int64 to avoid a [spans, width, 33] tensor
    tag_code = (ling[:, UPOS_OFFSET:] << torch.arange(UPOS_DIM + FEATS_DIM, device=ling.device)).sum(dim=1)
    pos = torch.clamp(starts.unsqueeze(1) + torch.arange(max_width, device=ling.device).unsqueeze(0), max=num_subtokens - 1)
    differs = (tag_code[pos] != tag_code[head_pos].unsqueeze(1)) & qualifies
    return differs.any(dim=1)


def agreement_features(top_morph, top_has_head, top_antecedent_idx):
    """ Condition 5: per AGREEMENT_ATTRS attribute an agree / disagree / unknown one-hot between a
    span's head and each candidate antecedent's head. Multi-valued heads agree iff the value sets
    overlap; 'unknown' when either side lacks the attribute (or has no head).
    top_morph: [num top spans, FEATS_DIM] (0/1); returns [num top spans, max top antecedents, AGREEMENT_DIM] float """
    import torch
    morph = top_morph.to(torch.bool) & top_has_head.unsqueeze(1)
    target = morph.unsqueeze(1)  # [n, 1, FEATS_DIM]
    antecedent = morph[top_antecedent_idx]  # [n, k, FEATS_DIM]
    features = []
    for attr in AGREEMENT_ATTRS:
        sl = FEATS_SLICES[attr]
        a, b = target[:, :, sl], antecedent[:, :, sl]
        known = a.any(dim=2) & b.any(dim=2)
        agree = (a & b).any(dim=2)
        features += [known & agree, known & torch.logical_not(agree), torch.logical_not(known)]
    return torch.stack(features, dim=2).to(torch.float)


# ---------------------------------------------------------------------------
# Standalone gold-mention check:  python ling_features.py <data_dir>/<lang>-dev.512.jsonlines
# ---------------------------------------------------------------------------

def gold_head_report(jsonlines_path, max_width=30, num_examples=15):
    import torch
    total, ambiguous, differing, no_head, too_wide = 0, 0, 0, 0, 0
    examples = []
    with open(jsonlines_path) as f:
        for line in f:
            doc = json.loads(line)
            ling = torch.tensor(doc['ling'], dtype=torch.long)
            spans = [tuple(m) for m in (m for c in doc['clusters'] for m in c)]
            too_wide += sum(1 for s, e in spans if e - s >= max_width)
            spans = [(s, e) for s, e in spans if e - s < max_width]
            if not spans:
                continue
            starts = torch.tensor([s for s, _ in spans])
            ends = torch.tensor([e for _, e in spans])
            head_pos, has_head, num_qual, qualifies = find_span_heads(ling, starts, ends, max_width)
            amb = num_qual > 1
            diff = ambiguous_with_different_tags(ling, starts, qualifies, head_pos) & amb
            total += len(spans)
            ambiguous += amb.sum().item()
            differing += diff.sum().item()
            no_head += (~has_head).sum().item()
            if len(examples) < num_examples:
                subtokens = [t for seg in doc['sentences'] for t in seg]
                for i in torch.nonzero(amb).flatten().tolist():
                    s, e = spans[i]
                    quals = [subtokens[s + o] for o in torch.nonzero(qualifies[i]).flatten().tolist()]
                    examples.append('%s: "%s" -> head "%s"; qualifying: %s' % (
                        doc['doc_key'], ' '.join(subtokens[s:e + 1]), subtokens[head_pos[i]], quals))
                    if len(examples) >= num_examples:
                        break
    print('Gold mentions (width < %d): %d; skipped (too wide): %d' % (max_width, total, too_wide))
    print('Head-ambiguous: %d (%.2f%%); of those with differing tags: %d; no head: %d' % (
        ambiguous, 100 * ambiguous / max(total, 1), differing, no_head))
    for example in examples:
        print('  ' + example)


if __name__ == '__main__':
    gold_head_report(sys.argv[1])
