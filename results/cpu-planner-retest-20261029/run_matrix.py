import concurrent.futures, json, random, statistics, time
from pathlib import Path
import requests

BASE='http://127.0.0.1:30217'
OUT=Path('/root/autodl-fs/dspark-cpu-planner-retest-20261029')
TOKEN_LOW,TOKEN_HIGH=1000,150000
RNG_SEED=20261029
INPUTS=[512,2048,8192]
CONCURRENCIES=[1,4,8,16]
OUTPUTS=[128]
WAVES=2

sess=requests.Session()

def random_ids(n, seed):
    r=random.Random(seed)
    return [r.randrange(TOKEN_LOW,TOKEN_HIGH) for _ in range(n)]

def one_request(ids, out_len, rid):
    payload={'rid':rid,'input_ids':ids,'sampling_params':{'temperature':0.0,'top_k':1,'top_p':1.0,'max_new_tokens':out_len,'ignore_eos':True},'stream':True}
    start=time.perf_counter(); first=None; last=None; meta=None; text=''; err=None
    try:
        with requests.post(BASE+'/generate',json=payload,stream=True,timeout=900) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith('data: '): continue
                raw=line[6:]
                if raw=='[DONE]': continue
                item=json.loads(raw)
                now=time.perf_counter()
                if first is None: first=now
                last=now
                if item.get('text'): text+=item.get('text','')
                if item.get('meta_info') is not None: meta=item['meta_info']
    except Exception as e:
        err=repr(e)
    if last is None: last=time.perf_counter()
    if first is None: first=last
    completion=int((meta or {}).get('completion_tokens') or 0)
    ttft=(first-start)*1000.0
    e2e=(last-start)*1000.0
    tpot=((e2e-ttft)/max(completion-1,1)) if completion else None
    return {'rid':rid,'status':'error' if err else 'ok','error':err,'ttft_ms':ttft,'e2e_ms':e2e,'tpot_ms':tpot,'meta':meta,'text_chars':len(text),'prompt_tokens':(meta or {}).get('prompt_tokens')}

def server_records():
    j=sess.get(BASE+'/server_info',timeout=60).json()
    states=j.get('internal_states',[])
    allrec=[]
    payloads=[]
    for s in states:
        p=s.get('dspark_info_record') or {}
        payloads.append(p)
        allrec.extend(p.get('records') or [])
    return j,payloads,allrec

def summarize_records(records):
    if not records: return {}
    def vals(k): return [float(r[k]) for r in records if r.get(k) is not None]
    out={'record_count':len(records),'forward_ct_min':min(r.get('forward_ct',-1) for r in records),'forward_ct_max':max(r.get('forward_ct',-1) for r in records)}
    for k in ['step_cpu_ms','step_gpu_ms','draft_gpu_ms','target_verify_gpu_ms','budget','num_verify_tokens','verify_tokens_local','verify_tokens_graph_key','predicted_step_ms','predicted_theta']:
        v=vals(k)
        if v: out[k]={'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'p95':sorted(v)[max(0,int(len(v)*.95)-1)],'min':min(v),'max':max(v)}
    paths={}
    for r in records:
        p=(r.get('verify_path') or {}).get('selected_path')
        if p: paths[p]=paths.get(p,0)+1
    out['selected_path_counts']=paths
    verify=[]; full=[]; compact=[]; acc=[]
    for r in records:
        vp=r.get('verify_path') or {}
        if vp.get('full_logical_tokens') is not None: full.append(vp.get('full_logical_tokens'))
        if vp.get('selected_graph_tokens') is not None: compact.append(vp.get('selected_graph_tokens'))
        for q in r.get('reqs') or []:
            verify.append(q.get('verify_len')); acc.append(q.get('acc_len'))
    if verify: out['verify_len']={'n':len(verify),'mean':statistics.mean(verify),'median':statistics.median(verify),'min':min(verify),'max':max(verify)}
    if acc: out['acc_len']={'n':len(acc),'mean':statistics.mean(acc),'median':statistics.median(acc),'min':min(acc),'max':max(acc)}
    if full: out['full_logical_tokens']={'min':min(full),'max':max(full)}
    if compact: out['selected_graph_tokens']={'min':min(compact),'max':max(compact)}
    return out

def run_cell(inp,c,out_len):
    cell=f'p{inp}_c{c}_o{out_len}'
    cell_dir=OUT/'matrix'/cell; cell_dir.mkdir(parents=True,exist_ok=True)
    # one warmup request, excluded from measurement
    warm_rid=f'{cell}-warmup'
    one_request(random_ids(inp,RNG_SEED+hash(cell)%100000),min(8,out_len),warm_rid)
    waves=[]; all_rids=set(); before_records=[]
    try: _,_,before_records=server_records()
    except Exception: pass
    for wave in range(WAVES):
        ids_base=random_ids(inp,RNG_SEED+wave*100003+inp*17+c)
        reqs=[(random_ids(inp,RNG_SEED+wave*100003+inp*17+c*101+i),f'{cell}-w{wave}-r{i}') for i in range(c)]
        all_rids.update(rid for _,rid in reqs)
        t=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            fut=[ex.submit(one_request,ids,out_len,rid) for ids,rid in reqs]
            results=[f.result() for f in fut]
        waves.append({'wave':wave,'wall_ms':(time.perf_counter()-t)*1000.0,'results':results})
    _,payloads,after_records=server_records()
    records=[r for r in after_records if (r.get('reqs') and any(q.get('rid') in all_rids for q in r.get('reqs',[])))]
    summary={}
    rgood=[x for w in waves for x in w['results'] if x['status']=='ok']
    for k in ['ttft_ms','e2e_ms','tpot_ms']:
        v=[x[k] for x in rgood if x.get(k) is not None]
        if v: summary[k]={'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'p95':sorted(v)[max(0,int(len(v)*.95)-1)],'min':min(v),'max':max(v)}
    summary.update(summarize_records(records))
    summary.update({'cell':cell,'input_len':inp,'concurrency':c,'output_len':out_len,'wave_count':WAVES,'ok_requests':len(rgood),'error_requests':len([x for w in waves for x in w['results'] if x['status']!='ok']),'record_rids':sorted(all_rids)})
    doc={'summary':summary,'waves':waves,'records':records,'payload_headers':[{'mode':p.get('mode'),'components':p.get('components'),'simulate_acc_len':p.get('simulate_acc_len'),'gamma':p.get('gamma'),'verify_num_draft_tokens':p.get('verify_num_draft_tokens')} for p in payloads]}
    (cell_dir/'result.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

if __name__=='__main__':
    OUT.joinpath('matrix').mkdir(exist_ok=True)
    summaries=[]
    for inp in INPUTS:
      for c in CONCURRENCIES:
        if inp==8192 and c==16:
            print(json.dumps({'cell':f'p{inp}_c{c}_o128','status':'capacity_skipped','reason':'131072-token server capacity is fully consumed by prompt tokens before decode'}),flush=True)
            continue
        summaries.append(run_cell(inp,c,128))
    (OUT/'matrix_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
