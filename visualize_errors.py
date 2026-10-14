"""Render error_analysis.py's JSON output as an HTML page (saliency-style highlights).

For each error case, shows the mention and its predicted/gold antecedents
highlighted in their local context (via spaCy's displaCy, in manual mode --
no spaCy pipeline/model is loaded, it's used purely as a span-highlighting
renderer), plus a score bar for the ranked candidate list. Pure post-processing
of the JSON; does not touch the trained model or re-run inference.

Requires: pip install spacy
"""
import argparse
import html
import json
import sys
from os.path import join

try:
    from spacy import displacy
except ImportError:
    sys.exit('This script needs spaCy for rendering highlighted spans. Install with: pip install spacy')

from transformers import AutoTokenizer

import util
from error_analysis import load_doc_tokens

COLORS = {'MENTION': '#8ab4f8', 'PREDICTED': '#f28b82', 'GOLD': '#81c995'}

CSS = """
body { font-family: sans-serif; max-width: 900px; margin: 2em auto; color: #222; }
.case { border: 1px solid #ccc; border-radius: 8px; padding: 1em 1.5em; margin-bottom: 1.5em; }
.case h3 { margin-top: 0; }
.outcome { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; color: #fff; }
.outcome.false_new { background: #f29900; }
.outcome.false_link { background: #f28b82; }
.outcome.wrong_link { background: #d93025; }
.snippet-label { font-weight: bold; margin-top: 0.8em; }
.bar-row { display: flex; align-items: center; gap: 0.5em; margin: 2px 0; font-size: 0.9em; }
.bar-track { background: #eee; width: 300px; height: 14px; border-radius: 3px; overflow: hidden; }
.bar-fill { background: #8ab4f8; height: 100%; }
.bar-fill.is-predicted { background: #f28b82; }
.bar-fill.is-gold { background: #81c995; }
.bar-text { white-space: nowrap; }
"""


def window_snippet(tokenizer, tokens, span, context):
    """Build a local context window and locate the span's character offsets in it.

    Can't just join subtokens with spaces: this repo's experiments use WordPiece,
    SentencePiece and byte-level BPE tokenizers, each with its own convention for
    marking word boundaries, so only the matching tokenizer's own decoder gives
    back correct text. Offsets are found by searching for the span's own decoded
    text near where the decoded prefix ends, since decoding isn't always exactly
    prefix-stable at the boundary (e.g. trailing-space trimming).
    """
    start = max(0, span[0] - context)
    end = min(len(tokens) - 1, span[1] + context)
    window = tokens[start:end + 1]
    local_start, local_end = span[0] - start, span[1] - start

    text = tokenizer.convert_tokens_to_string(window)
    prefix = tokenizer.convert_tokens_to_string(window[:local_start])
    span_only = tokenizer.convert_tokens_to_string(window[local_start:local_end + 1])

    char_start = text.find(span_only, max(0, len(prefix) - 2))
    if char_start == -1:
        char_start = text.find(span_only)
    if char_start == -1:
        return text, 0, len(text)  # fallback: highlight the whole window
    return text, char_start, char_start + len(span_only)


def render_snippet(tokenizer, tokens, span, label, title, context):
    text, start, end = window_snippet(tokenizer, tokens, span, context)
    ex = {'text': text, 'ents': [{'start': start, 'end': end, 'label': label}], 'title': title}
    return displacy.render(ex, style='ent', manual=True, jupyter=False, options={'colors': COLORS})


def render_ranking(ranking, predicted_span, gold_spans):
    scores = [c['score'] for c in ranking]
    lo, hi = min(scores), max(scores)
    spread = hi - lo if hi > lo else 1.0
    rows = []
    for c in ranking:
        span = tuple(c['span'])
        cls = 'is-predicted' if predicted_span and span == tuple(predicted_span) \
            else 'is-gold' if span in gold_spans else ''
        width = (c['score'] - lo) / spread * 100
        rows.append(
            f'<div class="bar-row"><div class="bar-track"><div class="bar-fill {cls}" '
            f'style="width:{width:.0f}%"></div></div>'
            f'<span class="bar-text">{html.escape(c["candidate"])} ({c["score"]:.2f})</span></div>'
        )
    return '\n'.join(rows)


def render_case(tokenizer, case, outcome, tokens, context):
    mention_span = tuple(case['mention_span'])
    predicted_span = case['predicted_antecedent_span']
    gold_spans = {tuple(m) for m in case['gold_antecedent_spans']}

    parts = [
        f'<div class="case"><h3>{html.escape(case["doc_key"])} '
        f'<span class="outcome {outcome}">{outcome}</span></h3>',
        '<div class="snippet-label">Mention</div>',
        render_snippet(tokenizer, tokens, mention_span, 'MENTION', 'mention', context),
    ]
    if predicted_span:
        parts.append('<div class="snippet-label">Predicted link (score %.2f)</div>' % case['predicted_score'])
        parts.append(render_snippet(tokenizer, tokens, tuple(predicted_span), 'PREDICTED', 'predicted antecedent', context))
    if gold_spans:
        parts.append('<div class="snippet-label">Gold link(s)</div>')
        for span in sorted(gold_spans):
            parts.append(render_snippet(tokenizer, tokens, span, 'GOLD', 'gold antecedent', context))
    if case['ranking']:
        parts.append('<div class="snippet-label">Top candidates (ranked by score)</div>')
        parts.append(render_ranking(case['ranking'], predicted_span, gold_spans))
    parts.append('</div>')
    return '\n'.join(parts)


def visualize(error_json_path, config_name, outcomes=None, limit=10, context=10, out_path=None):
    with open(error_json_path, encoding='utf-8') as f:
        data = json.load(f)

    conf = util.initialize_config(config_name)
    doc_tokens = load_doc_tokens(conf['data_dir'], conf['language'], conf['max_segment_len'])
    tokenizer = AutoTokenizer.from_pretrained(conf['bert_tokenizer_name'])

    outcomes = outcomes or list(data['errors'].keys())
    body = []
    for outcome in outcomes:
        for case in data['errors'].get(outcome, [])[:limit]:
            tokens = doc_tokens[case['doc_key']]
            body.append(render_case(tokenizer, case, outcome, tokens, context))

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
    parser.add_argument('--context', type=int, default=10, help='tokens of context on each side of a span')
    parser.add_argument('--out', help='output HTML path')
    args = parser.parse_args()

    visualize(
        args.error_json, args.config_name,
        outcomes=args.outcomes.split(',') if args.outcomes else None,
        limit=args.limit,
        context=args.context,
        out_path=args.out,
    )
