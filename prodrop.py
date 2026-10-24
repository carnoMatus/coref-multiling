"""Shared helper for marking pro-drop (empty-node) tokens as elided.

Czech is pro-drop; CorefUD reconstructs elided subjects as UD empty nodes
(decimal word IDs, e.g. 13.1) with a real surface FORM but no actual position
in the written sentence. preprocess.py inserts that reconstructed form into
the BERT token sequence exactly like any other word (see its `word = node.form`
loop over `udapi_document.nodes_and_empty`), with nothing in the .jsonlines
output distinguishing it -- so telling a real written word from a
reconstructed one requires cross-referencing the raw .conllu.
"""
import re

from udapi.core.document import Document


def load_empty_word_flags(conllu_path):
    """doc_key -> list[bool], one entry per WORD (not subtoken), in the same
    order preprocess.py iterates them (`nodes_and_empty`, per document)."""
    flags = {}
    current_key = None
    for tree in Document(conllu_path).trees:
        if tree.newdoc:
            current_key = tree.newdoc
            flags[current_key] = []
        for node in tree.descendants_and_empty:
            flags[current_key].append(node.is_empty())
    return flags


def span_is_elided(subtoken_map, word_flags, start, end):
    """True if every subtoken in [start, end] traces back to an empty node."""
    return all(word_flags[subtoken_map[i]] for i in range(start, end + 1))


def clean_elided_text(text):
    """Multi-word empty-node reconstructions (e.g. generic "pro_někoho", for someone/
    anyone) join their words with "_" purely as a CoNLL-U one-form-per-line artifact,
    never a real underscore the author wrote. A subword tokenizer can also split that
    underscore into its own token, with its decoder inserting a space on each side, so
    a naive '_' -> ' ' replace leaves doubled/tripled spaces; collapse them instead.
    """
    return re.sub(r'\s*_\s*', ' ', text)
