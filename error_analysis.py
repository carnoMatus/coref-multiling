"""Per-mention antecedent-score dump for error analysis (inference only, no training).

For every gold mention in the dev split, shows the model's full ranked list of
candidate antecedents (not just the winning one) and classifies the mention as
correct / false_new (missed link) / false_link (linked but gold says no link) /
wrong_link (linked to the wrong antecedent). Mentions the model predicted that
are not gold mentions at all (pure mention-detection false positives) are out
of scope here; this only explains linking decisions on spans gold agrees are
mentions.
"""
import argparse
import json
import os
from collections import defaultdict
from os.path import join

import torch
from transformers import AutoTokenizer

from run import Runner
from tensorize import Tensorizer
import util
import prodrop


def load_doc_tokens(data_dir, language, max_seg_len):
    """doc_key -> flat subtoken list, read from the same dev jsonlines used for eval."""
    path = join(data_dir, f'{language}-dev.{max_seg_len}.jsonlines')
    tokens = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            tokens[example['doc_key']] = util.flatten(example['sentences'])
    return tokens


def load_doc_subtoken_maps(data_dir, language, max_seg_len):
    """doc_key -> subtoken_map (subtoken index -> word index), needed to cross-reference
    pro-drop/empty-node words (see prodrop.py)."""
    path = join(data_dir, f'{language}-dev.{max_seg_len}.jsonlines')
    maps = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            maps[example['doc_key']] = example['subtoken_map']
    return maps


def span_text(tokenizer, tokens, start, end, subtoken_map=None, word_flags=None):
    """Subtokens need the tokenizer's own decoder, not a naive join: this repo's experiments use
    WordPiece ('##' continuations), SentencePiece ('▁' word starts) and byte-level BPE (RobeCzech),
    and only the matching tokenizer can reverse any of those correctly."""
    text = tokenizer.convert_tokens_to_string(tokens[start:end + 1])
    if subtoken_map and prodrop.span_is_elided(subtoken_map, word_flags, start, end):
        text = f'({text})'
    return text


def gold_cluster_lookup(gold_clusters):
    """(start, end) -> cluster id (0 = not a gold mention); first mention per cluster is non-anaphoric."""
    span_to_cluster = {}
    non_anaphoric = set()
    for cluster_id, cluster in enumerate(gold_clusters, start=1):
        sorted_cluster = sorted(tuple(m) for m in cluster)
        non_anaphoric.add(sorted_cluster[0])
        for mention in sorted_cluster:
            span_to_cluster[mention] = cluster_id
    return span_to_cluster, non_anaphoric


def classify(span, predicted_antecedent, span_to_cluster, non_anaphoric):
    gold_cluster = span_to_cluster.get(span, 0)
    if gold_cluster == 0:
        return None  # not a gold mention; out of scope
    should_link = span not in non_anaphoric
    if predicted_antecedent is None:
        return 'false_new' if should_link else 'correct'
    if not should_link:
        return 'false_link'  # gold says first mention of its cluster; model linked it anyway
    antecedent_cluster = span_to_cluster.get(predicted_antecedent, 0)
    return 'correct' if antecedent_cluster == gold_cluster else 'wrong_link'


def run_error_analysis(config_name, saved_suffix, gpu_id, doc_keys=None, max_errors=20, top_k=10, out_path=None):
    runner = Runner(config_name, gpu_id)
    conf = runner.config
    model = runner.initialize_model(saved_suffix)
    model.to(runner.device)
    model.eval()

    _, examples_dev = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()
    doc_tokens = load_doc_tokens(runner.data.data_dir, runner.data.language, runner.data.max_seg_len)
    doc_subtoken_maps = load_doc_subtoken_maps(runner.data.data_dir, runner.data.language, runner.data.max_seg_len)
    doc_word_flags = prodrop.load_empty_word_flags(conf['conll_pred_path'])
    tokenizer = AutoTokenizer.from_pretrained(conf['bert_tokenizer_name'])

    if doc_keys:
        wanted = set(doc_keys)
        examples_dev = [(k, e) for k, e in examples_dev if k in wanted]

    max_sentences = conf['max_training_sentences'] if 'max_pred_sentences' not in conf else conf['max_pred_sentences']
    tensorizer = Tensorizer(conf)

    errors = defaultdict(list)
    num_correct = 0
    next_id = 1

    for doc_key, tensor_example in examples_dev:
        gold_clusters = stored_info['gold'][doc_key]
        span_to_cluster, non_anaphoric = gold_cluster_lookup(gold_clusters)
        tokens = doc_tokens[doc_key]
        subtoken_map = doc_subtoken_maps[doc_key]
        word_flags = doc_word_flags[doc_key]
        text_of = lambda s, e: span_text(tokenizer, tokens, s, e, subtoken_map, word_flags)
        elided_of = lambda s, e: prodrop.span_is_elided(subtoken_map, word_flags, s, e)

        tensor_example = tensor_example[:7]  # strip gold; inference only
        num_sentences = tensor_example[0].shape[0]
        batch_examples = [tensor_example] if num_sentences <= max_sentences \
            else tensorizer.split_example(*tensor_example)

        word_offset = 0
        for j, example in enumerate(batch_examples):
            example_gpu = [d.to(runner.device) for d in example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
            span_starts = (span_starts + word_offset).tolist()
            span_ends = (span_ends + word_offset).tolist()
            antecedent_idx = antecedent_idx.tolist()
            antecedent_scores = antecedent_scores.tolist()
            word_offset += int(tensor_example[3][j * max_sentences:(j + 1) * max_sentences].sum())

            spans = list(zip(span_starts, span_ends))
            for i, span in enumerate(spans):
                best_idx, best_score = max(enumerate(antecedent_scores[i]), key=lambda p: p[1])
                predicted_antecedent = None if best_idx == 0 else spans[antecedent_idx[i][best_idx - 1]]

                outcome = classify(span, predicted_antecedent, span_to_cluster, non_anaphoric)
                if outcome is None:
                    continue
                if outcome == 'correct':
                    num_correct += 1
                    continue
                if len(errors[outcome]) >= max_errors:
                    continue

                # antecedent_scores[i][0] is a fixed zero baseline for "no antecedent" (model.py
                # hard-codes it via torch.cat([zeros(...), top_pairwise_scores])); it has no span
                # of its own, so it's added separately rather than zipped with antecedent_idx.
                candidates = [(spans[idx], text_of(*spans[idx]), score)
                              for idx, score in zip(antecedent_idx[i], antecedent_scores[i][1:])
                              if score != float('-inf')]
                candidates.append((None, 'Ø (no antecedent)', antecedent_scores[i][0]))
                candidates.sort(key=lambda t: t[2], reverse=True)

                gold_cluster_id = span_to_cluster[span]
                gold_antecedents = [m for m, c in span_to_cluster.items()
                                    if c == gold_cluster_id and m != span and m < span]

                errors[outcome].append({
                    'id': next_id,
                    'doc_key': doc_key,
                    'mention': text_of(*span),
                    'mention_span': list(span),
                    'mention_is_elided': elided_of(*span),
                    'predicted_antecedent': text_of(*predicted_antecedent) if predicted_antecedent else None,
                    'predicted_antecedent_span': list(predicted_antecedent) if predicted_antecedent else None,
                    'predicted_antecedent_is_elided': elided_of(*predicted_antecedent) if predicted_antecedent else None,
                    'predicted_score': best_score,
                    'gold_antecedents': [text_of(*m) for m in gold_antecedents],
                    'gold_antecedent_spans': [list(m) for m in gold_antecedents],
                    'gold_antecedent_is_elided': [elided_of(*m) for m in gold_antecedents],
                    'ranking': [{'candidate': text, 'span': list(cand_span) if cand_span else None, 'score': score}
                                for cand_span, text, score in candidates[:top_k]],
                })
                next_id += 1

    summary = {outcome: len(cases) for outcome, cases in errors.items()}
    summary['correct'] = num_correct
    result = {'summary': summary, 'errors': errors}

    if out_path is None:
        out_path = join(conf['data_dir'], 'analysis', f'error_analysis_{config_name}_{saved_suffix}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('Summary:', summary)
    print('Saved to', out_path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dump per-mention antecedent-score rankings for error analysis.')
    parser.add_argument('config_name')
    parser.add_argument('saved_suffix')
    parser.add_argument('gpu_id', type=int)
    parser.add_argument('--doc-keys', help='comma-separated doc_keys to restrict to')
    parser.add_argument('--max-errors', type=int, default=20, help='max cases to keep per error category')
    parser.add_argument('--top-k', type=int, default=10, help='candidates to keep per ranking')
    parser.add_argument('--out', help='output JSON path')
    args = parser.parse_args()

    run_error_analysis(
        args.config_name, args.saved_suffix, args.gpu_id,
        doc_keys=args.doc_keys.split(',') if args.doc_keys else None,
        max_errors=args.max_errors,
        top_k=args.top_k,
        out_path=args.out,
    )
