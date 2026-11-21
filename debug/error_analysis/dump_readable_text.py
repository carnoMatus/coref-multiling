"""Dump a CoNLL-U treebank as plain, annotation-free readable text.

Detokenizes via udapi (respects SpaceAfter, same as the gold data), one line
per sentence, with a header before each document so individual doc_keys
(e.g. the ones named in error_analysis.py's output) can be found with grep.

Czech is pro-drop, so many of the coreference chains traced through these
documents (e.g. the recurring "my"/"vy" generic-address chains) run through
*empty nodes* -- UD's reconstruction of an elided subject that has no surface
form at all. udapi's own sentence writer silently drops these, which makes a
mention that links to an elided pronoun look like it links to nothing. Here
they're inserted at their correct position, in parentheses, so e.g. "Platime"
(we pay, no visible subject) reads as "(my) Platime" -- same convention as
linguistics write-ups for pro-drop.

Usage:
    python debug/error_analysis/dump_readable_text.py cs_pdt-corefud-train.conllu cs_pdt-corefud-dev.conllu --out cs_pdt.txt
"""
import argparse

from udapi.core.document import Document


def sentence_text(tree, use_mwt=True):
    parts = []
    last_mwt_id = 0
    for node in tree.descendants_and_empty:
        mwt = node.multiword_token if not node.is_empty() else None
        if use_mwt and mwt:
            if node.ord > last_mwt_id:
                last_mwt_id = mwt.words[-1].ord
                parts.append(mwt.form)
                if mwt.misc['SpaceAfter'] != 'No':
                    parts.append(' ')
        else:
            # Multi-word empty-node reconstructions (e.g. generic "#Gen" nodes like
            # "pro_někoho", for someone/anyone) join their words with "_" in the CoNLL-U
            # form field purely because CoNLL-U is one-form-per-line; it's never a real
            # underscore the author wrote, so swap it back to a space for display.
            parts.append(f'({node.form.replace("_", " ")})' if node.is_empty() else node.form)
            if node.misc['SpaceAfter'] != 'No':
                parts.append(' ')
    return ''.join(parts).strip()


def dump(conllu_paths, out_path):
    with open(out_path, 'w', encoding='utf-8') as out:
        for path in conllu_paths:
            for tree in Document(path).trees:
                if tree.newdoc:
                    out.write(f'\n=== {tree.newdoc} ===\n')
                out.write(sentence_text(tree) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('conllu', nargs='+', help='one or more .conllu files, in the order to concatenate them')
    parser.add_argument('--out', required=True, help='output plain-text path')
    args = parser.parse_args()

    dump(args.conllu, args.out)
    print('Wrote', args.out)
