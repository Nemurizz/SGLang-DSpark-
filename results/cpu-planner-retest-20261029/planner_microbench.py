import json, statistics, time
from pathlib import Path
import torch
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkScheduleConfig, compute_verify_token_budget, _additive_step_time_tensor
from sglang.srt.speculative.dspark_components.dspark_sps import load_sps_table_from_path, SpsAdditiveCostTable

TABLE='/root/autodl-tmp/results/sps-probes/dspark14-sps-probe-20260830-210115/qwen3-14b-rtxpro6000-additive-sps.json'
OUT=Path('/root/autodl-fs/dspark-cpu-planner-retest-20261029/planner_microbench.json')
table=load_sps_table_from_path(TABLE)
assert isinstance(table,SpsAdditiveCostTable),type(table)
cfg=DSparkScheduleConfig(gamma=7)

def stat(xs):
    xs=sorted(xs); return {'n':len(xs),'mean_us':statistics.mean(xs),'median_us':statistics.median(xs),'p95_us':xs[max(0,int(len(xs)*.95)-1)],'p99_us':xs[max(0,int(len(xs)*.99)-1)],'min_us':xs[0],'max_us':xs[-1]}

out={'table':TABLE,'table_type':type(table).__name__,'gamma':7,'repeats':20000,'warmup':2000,'cells':[]}
for n in [1,4,8,16]:
    g=torch.Generator().manual_seed(20261029+n)
    conf=torch.rand((n,8),generator=g,dtype=torch.float32)*0.5+0.45
    history=torch.cumprod(conf,dim=1)
    cached=_additive_step_time_tensor(table=table,num_requests=n,num_budgets=n*cfg.resolved_max_verify_len()+1)
    unc=[]; cac=[]; b_unc=[]; b_cac=[]
    for _ in range(2000):
        compute_verify_token_budget(history_survival_probs=history,sps_table=table,cfg=cfg)
        compute_verify_token_budget(history_survival_probs=history,sps_table=table,cfg=cfg,cached_additive_step_time=cached)
    for _ in range(20000):
        t=time.perf_counter_ns(); a=compute_verify_token_budget(history_survival_probs=history,sps_table=table,cfg=cfg); unc.append((time.perf_counter_ns()-t)/1000); b_unc.append(a.budget)
        t=time.perf_counter_ns(); b=compute_verify_token_budget(history_survival_probs=history,sps_table=table,cfg=cfg,cached_additive_step_time=cached); cac.append((time.perf_counter_ns()-t)/1000); b_cac.append(b.budget)
    assert b_unc==b_cac
    out['cells'].append({'num_requests':n,'candidate_count':n*8+1,'uncached':stat(unc),'cached':stat(cac),'speedup_median':statistics.median(unc)/statistics.median(cac),'budget':b_unc[0]})
OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
