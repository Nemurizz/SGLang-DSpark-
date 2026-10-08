"""One fixed descriptive concurrency A/B; NEVER qualify stable gain.

h59/h60 failed qualification retained. No retry, no filtering, no graph change.
Still stop on token/five-acceptance difference, illegal access or p95 regression.
"""
import argparse
import asyncio
from pathlib import Path
import exact_graph_continuous_ab as continuous
import exact_graph_concurrent_ab as base


PROTOCOL={**continuous.PROTOCOL,
    'exploratory_descriptive_only':True,
    'performance_qualification':'DISABLED even if every observed ratio improves',
    'warmup_and_noise_failures':'recorded, NOT converted to admitted; descriptive probing only',
    'predecessor':'h59 short-wave warmup FAILED; h60 continuous c4 warmup FAILED (0.991% mean spread)',
    'no_more_protocol_retries_after_this_run':True,
    'wall_budget_seconds':3600,
}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inherit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    continuous.PROTOCOL=PROTOCOL
    base.PROTOCOL=PROTOCOL
    base.arm=continuous.arm
    raise SystemExit(asyncio.run(base.run(parser.parse_args())))
