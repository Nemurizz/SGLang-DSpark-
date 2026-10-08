"""Default-off, process-local fixed verification POLICY experiment.

Not an adaptive full-budget shortcut. Omit the SPS table explicitly, then reuse
the existing uniform compact layout. Do not edit kernels/untracked dependencies,
skip the confidence head/relay, or alter acceptance/KV commit. Graphs/DP/TP>1 and
dynamic caps are deliberately unsupported. Rollback is simply process exit.
"""
import functools
import json
import os
from pathlib import Path

POLICY = 'fixed-verify-all-compact-no-host-budget-v1'


def validate_options(args, environment):
    expected = {'tp_size': 1, 'pp_size': 1, 'dp_size': 1,
        'enable_dp_attention': False, 'disable_cuda_graph': True,
        'enable_deterministic_inference': True, 'page_size': 1,
        'speculative_algorithm': 'DSPARK', 'speculative_num_draft_tokens': 8,
        'speculative_dspark_block_size': 7, 'speculative_dspark_sps_table_path': None}
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise ValueError('Unsupported fixed verification option: ' + key)
    required = {'SGLANG_DSPARK_FIXED_VERIFY_ALL_EXPERIMENT': '1',
        'SGLANG_RAGGED_VERIFY_MODE': 'compact', 'SGLANG_DSPARK_RAGGED_VERIFY_GRAPH': '0',
        'SGLANG_DSPARK_DYNAMIC_DRAFT_LENGTH': '0', 'SGLANG_DSPARK_COMPACT_SCAN_CUDA': '0'}
    if any(environment.get(key) != value for key, value in required.items()):
        raise ValueError('Explicit fixed verification opt-in / graph-off guards missing')
    # Collection/probe hooks must not gain a different meaning under this policy.
    if any(value not in ('', '0') for key, value in environment.items()
           if key.startswith('SGLANG_DSPARK_') and
           any(part in key for part in ('COLLECT', 'DUMP', 'TRACE', 'FORCED_BUDGET'))):
        raise ValueError('Fixed verification is incompatible with collection/probe switches')


def apply_to_planner(planner):
    if (not planner._is_verify_all or not planner.is_compact_mode or
            planner._dp_tier_gather_enabled or planner.gamma != 7 or
            planner.verify_num_draft_tokens != 8 or
            planner.model_runner.decode_cuda_graph_runner is not None or
            planner._schedule_cfg.max_verify_len != 0):
        raise ValueError('Planner is not the supported existing uniform eager policy')
    if planner._uniform_layout_cache:
        raise ValueError('Policy must precede any layout construction')
    # No host confidence resolution/CPU sort is needed: schedule_layout already
    # ignores its budget in the explicitly configured uniform verify-all branch.
    planner._budget_planner = None
    planner._dynamic_graph_tier = False
    if planner.schedules_verify_budget or planner.confidence_budget_prepare() is not None:
        raise ValueError('Unused host-budget callback remained registered')


def install(planner_class, config, environment):
    if environment.get('SGLANG_DSPARK_FIXED_VERIFY_ALL_EXPERIMENT') != '1':
        raise ValueError('Fixed verify-all experiment is OFF by default')
    marker = getattr(planner_class, '_dspark_fixed_verify_policy', None)
    if marker is not None:
        if marker['policy'] != POLICY or marker['profile_id'] != config['profile_id']:
            raise ValueError('Conflicting verification policy')
        return marker
    original = planner_class.__init__
    marker = {'policy': POLICY, 'profile_id': config['profile_id'], 'planner_instances': 0}

    @functools.wraps(original)
    def initialize(self, **options):
        validate_options(options['server_args'], environment)
        original(self, **options)
        apply_to_planner(self)
        marker['planner_instances'] += 1
        folder = Path(config['policy_folder'])
        folder.mkdir(parents=True, exist_ok=True)
        data = {**marker, 'gamma': self.gamma, 'verify_width': self.verify_num_draft_tokens,
            'uniform_layout_cache_only': True, 'host_budget_enabled': False,
            'confidence_head_preserved': self.carries_confidence,
            'confidence_relay_preserved': True, 'graphs': 'OFF', 'tp': 1, 'dp': False,
            'sps_table_omitted_explicitly': True, 'source_files_unchanged': True}
        with (folder / f'planner-{marker["planner_instances"]}.json').open('x') as stream:
            json.dump(data, stream, indent=2)

    planner_class.__init__ = initialize
    planner_class._dspark_fixed_verify_policy = marker
    return marker


def make_hook(config):
    from sglang.srt.speculative.dspark_components.dspark_planner import DSparkVerifyPlanner
    from fixed_fa4_policy_hook import make_hook as fixed_policy_hook

    if config.get('verify_policy') != POLICY:
        raise ValueError('Explicit verification policy identity required')
    install(DSparkVerifyPlanner, config, dict(os.environ))
    return fixed_policy_hook(config)
