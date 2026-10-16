"""Dump a CoNLL-U treebank as plain, annotation-free readable text.

Detokenizes via udapi (respects SpaceAfter, same as the gold data), one line
per sentence, with a header before each document so individual doc_keys
(e.g. the ones named in error_analysis.py's output) can be found with grep.

Usage:
    python dump_readable_text.py cs_pdt-corefud-train.conllu cs_pdt-corefud-dev.conllu --out cs_pdt.txt
"""
import argparse

from udapi.core.document import Document


def dump(conllu_paths, out_path):
    with open(out_path, 'w', encoding='utf-8') as out:
        for path in conllu_paths:
            for tree in Document(path).trees:
                if tree.newdoc:
                    out.write(f'\n=== {tree.newdoc} ===\n')
                out.write(tree.get_sentence() + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('conllu', nargs='+', help='one or more .conllu files, in the order to concatenate them')
    parser.add_argument('--out', required=True, help='output plain-text path')
    args = parser.parse_args()

    dump(args.conllu, args.out)
    print('Wrote', args.out)
