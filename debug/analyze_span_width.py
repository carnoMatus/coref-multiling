import json

PATH = "./data/UD/CorefUD-1.4-public/data/CorefUD_Czech-PDT/cs_pdt-corefud-train.512.jsonlines"

lengths = []
for line in open(PATH):
    doc = json.loads(line)
    for cluster in doc["clusters"]:
        for start, end in cluster:
            lengths.append(end - start + 1)

lengths.sort()
n = len(lengths)
print(f"Total gold mentions: {n}")
for p in [50, 75, 90, 95, 97, 99, 99.5, 100]:
    idx = min(n - 1, int(n * p / 100))
    print(f"  p{p}: {lengths[idx]} subtokens")

print()
for width in [5, 8, 10, 15, 20, 25, 30, 35, 40, 50]:
    covered = sum(1 for l in lengths if l <= width)
    print(f"max_span_width={width:>3}: covers {covered}/{n} = {100*covered/n:.2f}% of gold mentions")
