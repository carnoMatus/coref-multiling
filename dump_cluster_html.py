"""Render a whole CorefUD-derived corpus as HTML with gold coreference clusters
colour-highlighted, one colour per cluster, within running readable text.

Clusters are coloured per document, not globally -- colour only needs to
disambiguate clusters within the document a reader is looking at. But a
document's cluster count (median ~20, up to 200+ in this corpus) exceeds the
~8 hues that stay reliably distinguishable by eye (see the dataviz skill's
categorical-palette limit), so beyond the first 8 clusters per document,
colours repeat and every highlighted mention also gets a small printed index
number as a second, unambiguous encoding of which cluster it belongs to.

Source of truth for clusters is the same gold `clusters` field the model is
trained/evaluated against (the .jsonlines files), matching what
error_analysis.py calls "gold" -- not the raw CoNLL-U, which also encodes
bridging relations that aren't part of these flat identity clusters.

Usage:
    python dump_cluster_html.py train_exp3_base --out clusters.html
"""
import argparse
import html
import itertools
import json
from os.path import join

from transformers import AutoTokenizer

import util
import prodrop

PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
PALETTE_DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
# Past 8 clusters, colour repeats (cluster 9 reuses cluster 1's hue); border style cycles on
# each repeat as a second, non-colour channel, so same-colour same-border really does mean
# same palette slot *and* same round, while the printed index number is the final tiebreaker.
BORDER_STYLES = ['solid', 'dashed', 'dotted', 'double']

CSS = """
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; max-width: 900px;
       margin: 2em auto; padding: 0 16px; color: #0b0b0b; background: #fcfcfb; line-height: 1.8; }
.index { columns: 2; margin-bottom: 2em; font-size: 0.9em; }
.index a { color: #2a78d6; text-decoration: none; }
.doc { border-top: 1px solid #e1e0d9; padding-top: 1em; margin-top: 2em; scroll-margin-top: 1em; }
.doc h3 { color: #52514e; font-size: 0.95em; font-weight: 600; }
.doc h3 a { color: inherit; text-decoration: none; }
mark { border-radius: 3px; padding: 0 2px 2px 2px; color: #0b0b0b; border-bottom: 2px solid; }
sub.cid { font-size: 0.65em; color: #52514e; margin-left: 1px; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) body { color: #f0efec; background: #1a1a19; }
  :root:not([data-theme="light"]) .doc h3 { color: #c3c2b7; }
  :root:not([data-theme="light"]) mark { color: #0b0b0b; }
}
"""


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def kept_spans(clusters):
    """Clusters sorted by first mention; overlapping mentions from different
    clusters are pruned greedily (kept earliest-starting first) since HTML
    can't render crossing/overlapping highlights."""
    clusters = [c for c in clusters if len(c) >= 2]
    clusters = sorted(clusters, key=lambda c: min(tuple(m) for m in c))

    all_spans = sorted((tuple(m)[0], tuple(m)[1], ci)
                        for ci, c in enumerate(clusters) for m in c)
    kept = []
    occupied_until = -1
    for s, e, ci in all_spans:
        if s <= occupied_until:
            continue
        kept.append((s, e, ci))
        occupied_until = e
    return kept


def elided_word_spans(subtoken_map, word_flags):
    """One (start, end) subtoken span per distinct pro-drop (empty-node) WORD in the
    document -- not just ones that happen to be gold mentions -- so e.g. generic "pro
    někoho" reconstructions get the same "(...)" treatment as dump_readable_text.py,
    wherever they appear, not only inside a coreference cluster highlight."""
    spans = []
    i, n = 0, len(subtoken_map)
    while i < n:
        if word_flags[subtoken_map[i]]:
            j = i
            word = subtoken_map[i]
            while j < n and subtoken_map[j] == word:
                j += 1
            spans.append((i, j - 1))
            i = j
        else:
            i += 1
    return spans


def strip_special_tokens(tokens, subtoken_map, clusters, special_tokens):
    """Documents are split into BERT segments, each wrapped in [CLS]/[SEP] (or
    <s>/</s> etc.) -- drop those structural tokens and remap mention spans onto
    the resulting index space, so they don't show up as noise mid-sentence."""
    special = set(special_tokens)
    old_to_new = {}
    new_tokens = []
    new_subtoken_map = []
    for i, t in enumerate(tokens):
        if t not in special:
            old_to_new[i] = len(new_tokens)
            new_tokens.append(t)
            new_subtoken_map.append(subtoken_map[i])
    new_clusters = []
    for cluster in clusters:
        new_cluster = [[old_to_new[m[0]], old_to_new[m[1]]] for m in cluster
                        if m[0] in old_to_new and m[1] in old_to_new]
        if new_cluster:
            new_clusters.append(new_cluster)
    return new_tokens, new_subtoken_map, new_clusters


def render_doc(tokenizer, example, word_flags, dark=False):
    tokens = list(itertools.chain.from_iterable(example['sentences']))
    tokens, subtoken_map, clusters = strip_special_tokens(
        tokens, example['subtoken_map'], example['clusters'], tokenizer.all_special_tokens)
    full_text = tokenizer.convert_tokens_to_string(tokens)
    palette = PALETTE_DARK if dark else PALETTE

    cluster_spans = kept_spans(clusters)

    def overlaps_a_cluster(s, e):
        return any(s <= ce and cs <= e for cs, ce, _ in cluster_spans)

    # Pro-drop words already covered by a cluster highlight get bracketed there instead
    # (see below); this only adds the ones that aren't part of any coreference cluster.
    plain_elided = [(s, e) for s, e in elided_word_spans(subtoken_map, word_flags)
                    if not overlaps_a_cluster(s, e)]

    events = sorted([(s, e, ci) for s, e, ci in cluster_spans] +
                     [(s, e, None) for s, e in plain_elided], key=lambda x: (x[0], x[1]))

    out = []
    cursor = 0
    search_from = 0
    for s, e, ci in events:
        span_text = tokenizer.convert_tokens_to_string(tokens[s:e + 1])
        idx = full_text.find(span_text, search_from)
        if idx == -1:
            continue  # couldn't locate it; skip rather than mis-highlight
        out.append(html.escape(full_text[cursor:idx]))
        is_elided = prodrop.span_is_elided(subtoken_map, word_flags, s, e)
        display_text = f'({prodrop.clean_elided_text(span_text)})' if is_elided else span_text
        if ci is None:
            out.append(html.escape(display_text))
        else:
            color = palette[ci % len(palette)]
            border = BORDER_STYLES[(ci // len(palette)) % len(BORDER_STYLES)]
            out.append(f'<mark style="background:{color}55;border-bottom-style:{border};border-color:{color}">'
                        f'{html.escape(display_text)}<sub class="cid">{ci + 1}</sub></mark>')
        cursor = idx + len(span_text)
        search_from = cursor
    out.append(html.escape(full_text[cursor:]))
    return ''.join(out)


def dump(jsonl_paths, conllu_paths, config_name, out_path):
    conf = util.initialize_config(config_name)
    tokenizer = AutoTokenizer.from_pretrained(conf['bert_tokenizer_name'])

    word_flags = {}
    for path in conllu_paths:
        word_flags.update(prodrop.load_empty_word_flags(path))

    body = []
    index = []
    for path in jsonl_paths:
        for example in load_jsonl(path):
            doc_id = html.escape(example['doc_key'])
            index.append(f'<a href="#{doc_id}">{doc_id}</a>')
            body.append(f'<div class="doc" id="{doc_id}"><h3><a href="#{doc_id}">{doc_id}</a></h3>'
                         f'<p>{render_doc(tokenizer, example, word_flags[example["doc_key"]])}</p></div>')

    html_doc = (f'<!doctype html><html><head><meta charset="utf-8">'
                f'<style>{CSS}</style></head><body>'
                f'<h2>{html.escape(config_name)}</h2>'
                f'<div class="index">{" ".join(index)}</div>'
                f'{"".join(body)}</body></html>')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_doc)
    print('Wrote %d documents to %s' % (len(body), out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config_name', help='config to read data_dir/language/tokenizer from')
    parser.add_argument('--jsonl', nargs='*', help='explicit .jsonlines paths; default: that '
                         'config\'s train+dev files')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    conf = util.initialize_config(args.config_name)
    if args.jsonl:
        paths = args.jsonl
    else:
        base = join(conf['data_dir'], f"{conf['language']}-{{split}}.{conf['max_segment_len']}.jsonlines")
        paths = [base.format(split='train'), base.format(split='dev')]
    conllu_base = join(conf['data_dir'], f"{conf['language']}-{{split}}.conllu")
    conllu_paths = [conllu_base.format(split='train'), conllu_base.format(split='dev')]

    dump(paths, conllu_paths, args.config_name, args.out)
