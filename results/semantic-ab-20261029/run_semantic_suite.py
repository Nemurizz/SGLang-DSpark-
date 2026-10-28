import concurrent.futures, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
from statistics import mean, median
import requests
from transformers import AutoTokenizer

ROOT=Path('/root/autodl-tmp/src/sglang-main')
OUT=Path('/root/autodl-fs/dspark-semantic-ab-20261029')
OUT.mkdir(parents=True,exist_ok=True)
BASE='http://127.0.0.1:30317'
MODEL='/root/autodl-tmp/models/Qwen3-14B'
DRAFT='/root/autodl-tmp/models/dspark_qwen3_14b_block7'
SPS='/root/autodl-tmp/results/sps-probes/dspark14-sps-probe-20260830-210115/qwen3-14b-rtxpro6000-additive-sps.json'
INPUT_LEN=512
OUTPUT_LEN=128
ARRIVAL_RATE=2.0
INTERVAL=1.0/ARRIVAL_RATE
CONCURRENCIES=[1,4,8]
TOKENIZER=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)

PROMPTS={
 'code':[
  'Write a Python function merge_intervals(intervals) that merges overlapping half-open intervals. Return clean code, explain the invariant, and give the time complexity.',
  'Find and fix the bug in this Python LRU cache design: it evicts the newest entry instead of the least recently used entry. Provide corrected code and a short test.',
  'Design a streaming Python parser for newline-delimited JSON that tolerates one malformed record, reports its line number, and preserves valid records. Explain memory usage.',
  'Write a SQL query that returns each merchant\'s latest successful payment, handling duplicate timestamps deterministically. Explain the index you would add.'
 ],
 'math':[
  'Solve this probability problem carefully: three independent tests have positive rates 0.8, 0.7, and 0.9. What is the probability exactly two are positive? Show the derivation.',
  'Solve the recurrence a_n = 3 a_{n-1} - 2 a_{n-2}, with a_0=1 and a_1=2. Give a closed form and verify it for n=2 and n=3.',
  'A rectangle has perimeter 40 and integer side lengths. Determine the side lengths that maximize area, prove why, and state the maximum area.',
  'Explain why the harmonic mean is smaller than or equal to the arithmetic mean for positive numbers, and give a short proof for two variables.'
 ],
 'open_qa':[
  'Explain why a radix tree can reduce repeated prefill work in an LLM serving system, and distinguish cache hit rate from end-to-end latency.',
  'Compare optimistic and pessimistic concurrency control in a payment service. Discuss retries, idempotency, and what happens when a transaction conflicts.',
  'Explain the difference between a queueing delay and a GPU execution delay in a batched inference service, and name measurements that separate them.',
  'Describe how a production system should roll out a numerical kernel optimization while protecting correctness, tail latency, and rollback safety.'
 ]
}

def make_ids(text):
    ids=TOKENIZER.encode(text,add_special_tokens=True)
    filler=TOKENIZER.encode(' The request is part of a fixed benchmark context.',add_special_tokens=False)
    while len(ids)<INPUT_LEN:
        ids.extend(filler)
    ids=ids[:INPUT_LEN]
    if len(ids)<INPUT_LEN: ids += [TOKENIZER.eos_token_id or 0]*(INPUT_LEN-len(ids))
    return ids

REQUESTS=[]
for category,items in PROMPTS.items():
    for idx,text in enumerate(items):
        ids=make_ids(text)
        REQUESTS.append({'category':category,'index':idx,'prompt':text,'input_ids':ids,'prompt_sha256':hashlib.sha256(bytes(str(ids),'utf-8')).hexdigest()})

COMMON=[
 '--model-path',MODEL,'--tokenizer-path',MODEL,'--trust-remote-code','--host','127.0.0.1','--port','30317',
 '--mem-fraction-static','0.70','--max-total-tokens','131072','--random-seed','20261029',
 '--attention-backend','fa4','--decode-attention-backend','fa4','--prefill-attention-backend','fa4',
 '--sampling-backend','pytorch','--stream-interval','1','--dtype','bfloat16','--kv-cache-dtype','bfloat16',
 '--bf16-gemm-backend','torch','--page-size','1','--disable-cuda-graph','--disable-radix-cache','--enable-deterministic-inference'
]

def launch_cmd(arm,c):
    cmd=['/root/conda-envs/sglang/bin/python','-m','sglang.launch_server']+COMMON+['--max-running-requests',str(c)]
    if arm!='ar':
        cmd += ['--speculative-algorithm','DSPARK','--speculative-draft-model-path',DRAFT,
                '--speculative-num-steps','1','--speculative-eagle-topk','1','--speculative-num-draft-tokens','8',
                '--speculative-dspark-block-size','7','--speculative-draft-attention-backend','fa4',
                '--speculative-dsa-topk-backend','sgl-kernel','--speculative-dspark-sps-table-path',SPS]
    return cmd

def launch_env(arm):
    e=os.environ.copy(); e['PYTHONPATH']=str(ROOT/'python')
    for k in list(e):
        if k.startswith('SGLANG_DSPARK_') or k.startswith('SGLANG_RAGGED_') or k=='SGLANG_SIMULATE_ACC_LEN': e.pop(k,None)
    if arm=='full':
        e.update({'SGLANG_RAGGED_VERIFY_MODE':'static','SGLANG_SIMULATE_ACC_LEN':'-1',
                  'SGLANG_DSPARK_DEBUG_DUMP':'core,step_cpu_time,step_gpu_time,draft_gpu_time,target_verify_gpu_time,verify_path,reqs',
                  'SGLANG_DSPARK_PREFIX_CACHE':'0','SGLANG_DSPARK_DISABLE_RADIX_CACHE':'1',
                  'SGLANG_DSPARK_RAGGED_VERIFY_GRAPH':'0','SGLANG_DSPARK_COMPACT_SCAN_CUDA':'0'})
    elif arm=='dynamic':
        e.update({'SGLANG_RAGGED_VERIFY_MODE':'compact','SGLANG_SIMULATE_ACC_LEN':'-1',
                  'SGLANG_DSPARK_DEBUG_DUMP':'core,step_cpu_time,step_gpu_time,draft_gpu_time,target_verify_gpu_time,verify_path,reqs',
                  'SGLANG_DSPARK_PREFIX_CACHE':'0','SGLANG_DSPARK_DISABLE_RADIX_CACHE':'1',
                  'SGLANG_DSPARK_RAGGED_VERIFY_GRAPH':'0','SGLANG_DSPARK_COMPACT_SCAN_CUDA':'0',
                  'SGLANG_DSPARK_COMPACT_MIN_TOKEN_SAVINGS_RATIO':'0'})
    return e

def wait_ready(proc,log):
    for _ in range(180):
        try:
            if requests.get(BASE+'/health',timeout=2).status_code==200:return
        except Exception: pass
        if proc.poll() is not None:
            raise RuntimeError(f'server exited {proc.returncode}: {log.read_text(errors="replace")[-5000:]}')
        time.sleep(2)
    raise RuntimeError('server readiness timeout: '+log.read_text(errors='replace')[-5000:])

def one_request(req,arm,c,scheduled):
    rid=f'{arm}-c{c}-{req["category"]}-{req["index"]}'
    sleep=max(0,scheduled-time.perf_counter());
    if sleep: time.sleep(sleep)
    started=time.perf_counter(); first=None; last=None; meta=None; err=None; text=''
    payload={'rid':rid,'input_ids':req['input_ids'],'sampling_params':{'temperature':0.0,'top_k':1,'top_p':1.0,'max_new_tokens':OUTPUT_LEN,'ignore_eos':True},'stream':True}
    try:
        with requests.post(BASE+'/generate',json=payload,stream=True,timeout=900) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith('data: '): continue
                raw=line[6:]
                if raw=='[DONE]': continue
                item=json.loads(raw); now=time.perf_counter()
                if first is None:first=now
                last=now
                text+=item.get('text','')
                if item.get('meta_info') is not None: meta=item['meta_info']
    except Exception as e: err=repr(e)
    if first is None:first=time.perf_counter()
    if last is None:last=first
    completion=int((meta or {}).get('completion_tokens') or OUTPUT_LEN)
    return {'rid':rid,'category':req['category'],'index':req['index'],'prompt_sha256':req['prompt_sha256'],'status':'error' if err else 'ok','error':err,
            'scheduled_delay_ms':(started-scheduled)*1000,'ttft_ms':(first-started)*1000,'e2e_ms':(last-started)*1000,
            'tpot_ms':((last-first)*1000/max(completion-1,1)),'meta':meta,'text_chars':len(text)}

def qstats(xs):
    xs=sorted(x for x in xs if x is not None)
    if not xs:return None
    return {'n':len(xs),'mean':mean(xs),'median':median(xs),'p95':xs[max(0,int(len(xs)*.95)-1)],'min':xs[0],'max':xs[-1]}

def aggregate(responses,records,arm,c):
    out={'arm':arm,'concurrency':c,'arrival_rate':ARRIVAL_RATE,'input_len':INPUT_LEN,'output_len':OUTPUT_LEN,'requests':len(responses)}
    for k in ['ttft_ms','tpot_ms','e2e_ms','scheduled_delay_ms']:out[k]=qstats([r[k] for r in responses if r['status']=='ok'])
    metas=[r.get('meta') or {} for r in responses if r['status']=='ok']
    for k in ['spec_accept_rate','spec_accept_length','spec_num_correct_drafts','spec_num_proposed_drafts','spec_verify_ct']:
        out[k]=qstats([m.get(k) for m in metas if m.get(k) is not None])
    for k in ['planner_cpu_ms','step_cpu_ms','step_gpu_ms','draft_gpu_ms','target_verify_gpu_ms','budget','num_verify_tokens']:
        out[k]=qstats([r.get(k) for r in records if r.get(k) is not None])
    acc=[]
    for r in records:
        for q in r.get('reqs') or []:
            if q.get('rid') in {x['rid'] for x in responses}: acc.append(q.get('acc_len'))
    out['record_accept_len']=qstats(acc)
    paths={}
    for r in records:
        p=(r.get('verify_path') or {}).get('selected_path')
        if p:paths[p]=paths.get(p,0)+1
    out['selected_path_counts']=paths
    return out

def run_one(arm,c):
    label=f'{arm}_c{c}'; d=OUT/label; d.mkdir(parents=True,exist_ok=True); log=d/'server.log'
    with log.open('w') as fh:
        proc=subprocess.Popen(launch_cmd(arm,c),cwd=ROOT,env=launch_env(arm),stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        wait_ready(proc,log)
        responses=[]; start=time.perf_counter()+2.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(REQUESTS)) as ex:
            futs=[ex.submit(one_request,req,arm,c,start+i*INTERVAL) for i,req in enumerate(REQUESTS)]
            responses=[f.result() for f in futs]
        info=requests.get(BASE+'/server_info',timeout=60).json()
        records=[]
        for st in info.get('internal_states',[]):
            records.extend((st.get('dspark_info_record') or {}).get('records') or [])
        rids={r['rid'] for r in responses}
        records=[r for r in records if any(q.get('rid') in rids for q in r.get('reqs') or [])]
        summary=aggregate(responses,records,arm,c)
        doc={'summary':summary,'responses':responses,'records':records,'server_info':{'model_path':info.get('model_path'),'speculative_algorithm':info.get('speculative_algorithm'),'max_running_requests':info.get('max_running_requests'),'max_total_num_tokens':info.get('max_total_num_tokens'),'dtype':info.get('dtype'),'disable_cuda_graph':info.get('disable_cuda_graph')}}
        (d/'result.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    finally:
        try:
            os.killpg(proc.pid,signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try: os.killpg(proc.pid,signal.SIGKILL)
            except Exception: pass

if __name__=='__main__':
    summaries=[]
    for arm in ['ar','full','dynamic']:
        for c in CONCURRENCIES:
            run_one(arm,c)
            summaries.append(json.loads((OUT/f'{arm}_c{c}'/'result.json').read_text())['summary'])
    (OUT/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
