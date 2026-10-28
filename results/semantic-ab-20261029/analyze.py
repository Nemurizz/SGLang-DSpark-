import json,statistics
from pathlib import Path
ROOT=Path('/root/autodl-fs/dspark-semantic-ab-20261029')

def q(v):
 v=sorted(x for x in v if x is not None)
 if not v:return None
 return {'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'p95':v[max(0,int(len(v)*.95)-1)],'min':v[0],'max':v[-1]}
allcats=[]; overall=[]
for arm in ['ar','full','dynamic']:
 for c in [1,4,8]:
  d=json.loads((ROOT/f'{arm}_c{c}'/'result.json').read_text())
  overall.append(d['summary'])
  by={}
  for r in d['responses']:
   by.setdefault(r['category'],[]).append(r)
  for cat,rs in by.items():
   metas=[r.get('meta') or {} for r in rs if r['status']=='ok']
   item={'arm':arm,'concurrency':c,'category':cat,'requests':len(rs)}
   for key in ['ttft_ms','tpot_ms','e2e_ms']:
    item[key]=q([r[key] for r in rs if r['status']=='ok'])
   for key in ['spec_accept_rate','spec_accept_length','spec_num_correct_drafts','spec_num_proposed_drafts','spec_verify_ct']:
    item[key]=q([m.get(key) for m in metas if m.get(key) is not None])
   allcats.append(item)
(ROOT/'category_summary.json').write_text(json.dumps(allcats,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'overall_summary.json').write_text(json.dumps(overall,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(allcats,ensure_ascii=False,indent=2))
