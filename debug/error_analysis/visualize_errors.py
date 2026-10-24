"""Render error_analysis.py's JSON output as an HTML page (saliency-style highlights).

For each error case, shows the mention, predicted antecedent and nearest gold
antecedent as one continuous passage (via spaCy's displaCy, in manual mode --
no spaCy pipeline/model is loaded, it's used purely as a multi-span highlighting
renderer), plus a score bar for the ranked candidate list. Pure post-processing
of the JSON; does not touch the trained model or re-run inference.

Requires: pip install spacy
"""
import argparse
import html
import json
import os
import sys
from os.path import join

# This tool lives two levels under the repo root but needs util.py, a core
# training module, which only resolves if the root is on sys.path -- Python
# otherwise only looks in this script's own directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

try:
    from spacy import displacy
except ImportError:
    sys.exit('This script needs spaCy for rendering highlighted spans. Install with: pip install spacy')

from transformers import AutoTokenizer

import util
import prodrop
import coref_io

COLORS = {'MENTION': '#8ab4f8', 'PREDICTED': '#f28b82', 'GOLD': '#81c995'}

CSS = """
body { font-family: sans-serif; max-width: 900px; margin: 2em auto; color: #222; }
.case { border: 1px solid #ccc; border-radius: 8px; padding: 1em 1.5em; margin-bottom: 1.5em; }
.case h3 { margin-top: 0; }
.outcome { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; color: #fff; }
.outcome.false_new { background: #f29900; }
.outcome.false_link { background: #f28b82; }
.outcome.wrong_link { background: #d93025; }
.caption { color: #555; margin-bottom: 0.6em; }
.elision { color: #999; font-style: italic; text-align: center; margin: 0.3em 0; }
.snippet-label { font-weight: bold; margin-top: 0.8em; }
.bar-row { display: flex; align-items: center; gap: 0.5em; margin: 2px 0; font-size: 0.9em; }
.bar-track { background: #eee; width: 300px; height: 14px; border-radius: 3px; overflow: hidden; }
.bar-fill { background: #8ab4f8; height: 100%; }
.bar-fill.is-predicted { background: #f28b82; }
.bar-fill.is-gold { background: #81c995; }
.bar-fill.is-dummy { background: repeating-linear-gradient(45deg, #999, #999 4px, #bbb 4px, #bbb 8px); }
.bar-text { white-space: nowrap; }
"""


def snap_to_word_boundary(subtoken_map, start, end):
    """Extend [start, end] outward until both ends land on a real word boundary.

    A context window cut mid-word lands on a bare continuation subtoken (e.g. a
    WordPiece "##xyz" with no preceding piece in the window) that the tokenizer's
    decoder can't merge back into its word, leaking the raw marker into the text.
    subtoken_map (subtoken index -> word index) is defined the same way regardless
    of tokenizer family, so this snapping works for WordPiece, SentencePiece and
    byte-BPE alike.
    """
    word_start = subtoken_map[start]
    while start > 0 and subtoken_map[start - 1] == word_start:
        start -= 1
    word_end = subtoken_map[end]
    while end < len(subtoken_map) - 1 and subtoken_map[end + 1] == word_end:
        end += 1
    return start, end


def build_chunks(spans_with_labels, num_tokens, context, max_gap, subtoken_map):
    """Window each span, then merge windows that are close enough to read as one
    continuous passage. Windows further apart than max_gap stay separate chunks,
    so a distant antecedent doesn't force rendering everything in between."""
    windows = []
    for item in spans_with_labels:  # item = (span, label, is_elided)
        span = item[0]
        start = max(0, span[0] - context)
        end = min(num_tokens - 1, span[1] + context)
        start, end = snap_to_word_boundary(subtoken_map, start, end)
        windows.append([start, end, [item]])
    windows.sort(key=lambda w: w[0])

    merged = [windows[0]]
    for w in windows[1:]:
        last = merged[-1]
        if w[0] - last[1] <= max_gap:
            last[1] = max(last[1], w[1])
            last[2].extend(w[2])
        else:
            merged.append(w)
    return merged


def render_chunk(tokenizer, tokens, start, end, spans_with_labels, lead_ellipsis=False, trail_ellipsis=False):
    """One decoded text block with all its spans found and highlighted inside it.

    Can't just join subtokens with spaces: this repo's experiments use WordPiece,
    SentencePiece and byte-level BPE tokenizers, each with its own word-boundary
    convention, so only the matching tokenizer's own decoder gives back correct
    text. Offsets are found by searching for each span's own decoded text near
    where its decoded prefix ends, since decoding isn't always exactly
    prefix-stable at the boundary (e.g. trailing-space trimming).
    """
    window = tokens[start:end + 1]
    text = tokenizer.convert_tokens_to_string(window)
    ents = []
    for span, label, is_elided in spans_with_labels:
        local_start, local_end = span[0] - start, span[1] - start
        prefix = tokenizer.convert_tokens_to_string(window[:local_start])
        span_only = tokenizer.convert_tokens_to_string(window[local_start:local_end + 1])
        char_start = text.find(span_only, max(0, len(prefix) - 2))
        if char_start == -1:
            char_start = text.find(span_only)
        if char_start == -1:
            continue  # couldn't locate it; skip rather than mis-highlight
        ents.append({'start': char_start, 'end': char_start + len(span_only), 'label': label, 'elided': is_elided})

    # Pro-drop (elided) mentions get wrapped in "(...)", same convention as dump_readable_text.py.
    # Ascending order with a running shift counter: each insertion pushes every later
    # ent's (not-yet-visited) offsets forward by the same amount, so they stay correct
    # when their turn comes, while already-visited (earlier) ents are untouched.
    ents.sort(key=lambda e: e['start'])
    shift = 0
    for ent in ents:
        ent['start'] += shift
        ent['end'] += shift
        if ent.pop('elided'):
            bracketed = prodrop.elide_bracket(text[ent['start']:ent['end']])
            delta = len(bracketed) - (ent['end'] - ent['start'])
            text = text[:ent['start']] + bracketed + text[ent['end']:]
            ent['end'] += delta
            shift += delta

    if lead_ellipsis:
        text = '… ' + text
        for ent in ents:
            ent['start'] += 2
            ent['end'] += 2
    if trail_ellipsis:
        text = text + ' …'

    ents.sort(key=lambda e: e['start'])
    ex = {'text': text, 'ents': ents, 'title': None}
    return displacy.render(ex, style='ent', manual=True, jupyter=False, options={'colors': COLORS})


def render_passage(tokenizer, tokens, spans_with_labels, context, max_gap, subtoken_map):
    chunks = build_chunks(spans_with_labels, len(tokens), context, max_gap, subtoken_map)
    pieces = []
    n = len(chunks)
    for i, (start, end, labels) in enumerate(chunks):
        if i > 0:
            gap = start - chunks[i - 1][1]
            pieces.append(f'<div class="elision">&#8942; {gap} tokens omitted &#8942;</div>')
        lead = i == 0 and start > 0
        trail = i == n - 1 and end < len(tokens) - 1
        pieces.append(render_chunk(tokenizer, tokens, start, end, labels, lead, trail))
    return '\n'.join(pieces)


def render_ranking(ranking, predicted_span, gold_spans):
    scores = [c['score'] for c in ranking]
    lo, hi = min(scores), max(scores)
    spread = hi - lo if hi > lo else 1.0
    rows = []
    for c in ranking:
        span = tuple(c['span']) if c['span'] else None
        cls = 'is-dummy' if span is None \
            else 'is-predicted' if predicted_span and span == predicted_span \
            else 'is-gold' if span in gold_spans else ''
        width = (c['score'] - lo) / spread * 100
        rows.append(
            f'<div class="bar-row"><div class="bar-track"><div class="bar-fill {cls}" '
            f'style="width:{width:.0f}%"></div></div>'
            f'<span class="bar-text">{html.escape(c["candidate"])} ({c["score"]:.2f})</span></div>'
        )
    return '\n'.join(rows)


def render_case(tokenizer, case, outcome, tokens, context, max_gap, subtoken_map):
    mention_span = tuple(case['mention_span'])
    predicted_span = tuple(case['predicted_antecedent_span']) if case['predicted_antecedent_span'] else None
    gold_spans = [tuple(m) for m in case['gold_antecedent_spans']]
    gold_elided = case.get('gold_antecedent_is_elided') or [False] * len(gold_spans)
    nearest_i = max(range(len(gold_spans)), key=lambda i: gold_spans[i][1]) if gold_spans else None

    spans_with_labels = [(mention_span, 'MENTION', bool(case.get('mention_is_elided')))]
    if predicted_span:
        spans_with_labels.append((predicted_span, 'PREDICTED', bool(case.get('predicted_antecedent_is_elided'))))
    if nearest_i is not None:
        spans_with_labels.append((gold_spans[nearest_i], 'GOLD', bool(gold_elided[nearest_i])))

    caption_bits = []
    if predicted_span:
        caption_bits.append('Predicted &rarr; <b>%s</b> (score %.2f)' %
                             (html.escape(case['predicted_antecedent']), case['predicted_score']))
    if case['gold_antecedents']:
        gold_text = '; '.join(html.escape(g) for g in case['gold_antecedents'])
        caption_bits.append(f'Gold &rarr; <b>{gold_text}</b>')

    parts = [
        f'<div class="case" id="case-{case["id"]}"><h3>#{case["id"]} &nbsp;{html.escape(case["doc_key"])} '
        f'<span class="outcome {outcome}">{outcome}</span></h3>',
    ]
    if caption_bits:
        parts.append('<div class="caption">' + ' &nbsp;|&nbsp; '.join(caption_bits) + '</div>')
    parts.append(render_passage(tokenizer, tokens, spans_with_labels, context, max_gap, subtoken_map))
    if case['ranking']:
        parts.append('<div class="snippet-label">Top candidates (ranked by score)</div>')
        parts.append(render_ranking(case['ranking'], predicted_span, set(gold_spans)))
    parts.append('</div>')
    return '\n'.join(parts)


def visualize(error_json_path, config_name, outcomes=None, limit=10, context=15, max_gap=60, out_path=None):
    with open(error_json_path, encoding='utf-8') as f:
        data = json.load(f)

    conf = util.initialize_config(config_name)
    doc_data = coref_io.load_doc_data(conf['data_dir'], conf['language'], conf['max_segment_len'])
    tokenizer = AutoTokenizer.from_pretrained(conf['bert_tokenizer_name'])

    outcomes = outcomes or list(data['errors'].keys())
    body = []
    for outcome in outcomes:
        for case in data['errors'].get(outcome, [])[:limit]:
            tokens, subtoken_map = doc_data[case['doc_key']]
            body.append(render_case(tokenizer, case, outcome, tokens, context, max_gap, subtoken_map))

    html_doc = f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head>' \
               f'<body><h2>Error analysis: {html.escape(config_name)}</h2>{"".join(body)}</body></html>'

    if out_path is None:
        out_path = error_json_path.rsplit('.json', 1)[0] + '.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_doc)
    print('Wrote %d cases to %s' % (len(body), out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render error_analysis.py JSON output as highlighted HTML.')
    parser.add_argument('error_json', help='path to the JSON produced by error_analysis.py')
    parser.add_argument('config_name', help='same config used to produce the JSON (to locate the dev jsonlines)')
    parser.add_argument('--outcomes', help='comma-separated subset of false_new,false_link,wrong_link')
    parser.add_argument('--limit', type=int, default=10, help='max cases per outcome category')
    parser.add_argument('--context', type=int, default=15, help='tokens of context on each side of a span')
    parser.add_argument('--max-gap', type=int, default=60,
                         help='merge spans into one continuous passage if closer than this many tokens, '
                              'otherwise show them as separate chunks with an omitted-text marker between')
    parser.add_argument('--out', help='output HTML path')
    args = parser.parse_args()

    visualize(
        args.error_json, args.config_name,
        outcomes=args.outcomes.split(',') if args.outcomes else None,
        limit=args.limit,
        context=args.context,
        max_gap=args.max_gap,
        out_path=args.out,
    )
