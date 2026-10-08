"""Two checked p2048 requests, host-only guard facts; no performance claim."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import dspark_prefix_ab as ab
from stable_baseline_gate import validate_row
from steady_tpot import integrity, drain_listener
from target_graph_diagnostic import TargetGraphServer, start


async def run(root):
    inherited=json.loads(Path('/root/autodl-fs/dspark-verify-all-ab-20261008-h47/manifest.json').read_text())
    template=inherited['execution_integrity_template']
    args=SimpleNamespace(**template['args'])
    args.observe_work=False
    args.startup_timeout=300
    dataset=json.loads((Path(args.oracle_jsonl).parent/'dataset.json').read_text())
    refs=ab.load_frozen_references(Path(args.oracle_jsonl),dataset)
    integrity(template,args)
    root.mkdir(exist_ok=False)
    args.output=str(root)
    server=TargetGraphServer(args,'full',root,diagnostic=True,candidate=True)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                connector=ab.http_connector(aiohttp)) as session:
            base=await start(server,session)
            for i in range(2):
                if i==1: ab.dump(root/'graph_policy'/'snapshot_graph',{'excluded':True})
                row=await ab.generate(session,base,dataset['p2048_c1']['prompts'][0],128,args.seed)
                row.update(cell='p2048_c1',variant=0,phase='host_guard_probe',ordinal=i)
                ab.append_jsonl(root/'requests.jsonl',row)
                validate_row(row,dataset['p2048_c1'],128,ordinary=False,require_cold=True)
                ab.check_reference(refs[('p2048_c1',0)],row)
                server.check()
            print((root/'graph_policy'/'graph_final.json').read_text(),flush=True)
    finally:
        server.stop()
        await drain_listener(args.port)
        integrity(template,args)


if __name__=='__main__':
    import sys
    asyncio.run(asyncio.wait_for(run(Path(sys.argv[1])),timeout=600))
