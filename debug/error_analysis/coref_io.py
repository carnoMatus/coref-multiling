"""Lightweight .jsonlines helpers shared by the error-analysis tools.

No model/training dependency (no torch, no run.Runner) -- importing this from
a display-only tool like visualize_errors.py or dump_cluster_html.py must
never drag in the GPU/training stack the way importing error_analysis.py
itself would.
"""
import itertools
import json
from os.path import join


def jsonl_path(data_dir, language, max_seg_len, split='dev'):
    return join(data_dir, f'{language}-{split}.{max_seg_len}.jsonlines')


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_doc_data(data_dir, language, max_seg_len, split='dev'):
    """doc_key -> (flat subtoken list, subtoken_map), read once from the .jsonlines."""
    data = {}
    for example in load_jsonl(jsonl_path(data_dir, language, max_seg_len, split)):
        tokens = list(itertools.chain.from_iterable(example['sentences']))
        data[example['doc_key']] = (tokens, example['subtoken_map'])
    return data
