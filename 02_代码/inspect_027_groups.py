import json
from collections import defaultdict
p=r'C:\Users\PC\Documents\Codex\实验\06_主线项目/027_自适应稀疏注意力与准确率效率前沿\05_运行记录\canonical_main_table.json'
rows=json.load(open(p,encoding='utf-8'))['rows']
d=defaultdict(lambda: defaultdict(list))
for r in rows:
    d[r['dataset']][r['seed']].append(r)
for dataset in d:
    print(dataset, 'seeds', sorted(d[dataset]), 'cats_per_seed', {s:len(v) for s,v in d[dataset].items()})
