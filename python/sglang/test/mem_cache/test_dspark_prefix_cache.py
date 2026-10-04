from types import SimpleNamespace

from sglang.srt.mem_cache.dspark_prefix_cache import DsparkPrefixCache


class _Node:
    def __init__(self, key, parent=None):
        self.key = tuple(key)
        self.parent = parent


def _cache(max_tokens=1024):
    cache = DsparkPrefixCache(SimpleNamespace(size=max_tokens * 4))
    cache.active = True
    cache.enabled = True
    cache.max_tokens = max_tokens
    return cache


def test_ready_prefix_and_partial_readiness():
    root = _Node(())
    parent = _Node(range(4), root)
    leaf = _Node(range(4, 8), parent)
    cache = _cache()

    cache.mark_inserted_path(leaf, ready=True, fingerprint="v1")
    assert cache.paired_prefix_length(leaf, "v1") == 8
    assert cache.paired_prefix_length(leaf, "v2") == 0

    cache.on_node_split(_Node(range(2), root), leaf, 2)
    # The old leaf state was split into a two-token parent and suffix.
    assert cache.stats()["tracked_tokens"] <= 8


def test_target_only_and_reset_are_safe():
    root = _Node(())
    leaf = _Node(range(6), root)
    cache = _cache()
    cache.mark_inserted_path(leaf, ready=False)
    assert cache.paired_prefix_length(leaf) == 0
    cache.mark_inserted_path(leaf, ready=True)
    assert cache.paired_prefix_length(leaf) == 6
    cache.reset()
    assert cache.paired_prefix_length(leaf) == 0
    assert cache.stats()["tracked_tokens"] == 0


def test_budget_evicts_metadata_only():
    root = _Node(())
    a = _Node(range(4), root)
    b = _Node(range(4, 8), root)
    cache = _cache(max_tokens=4)
    cache.mark_inserted_path(a, ready=True)
    cache.mark_inserted_path(b, ready=True)
    assert cache.stats()["tracked_tokens"] <= 4
    assert cache.stats()["metadata_evictions"] >= 1


def test_fingerprint_changes_with_draft_configuration():
    a = SimpleNamespace(
        speculative_draft_model_path="draft-a",
        speculative_draft_model_revision="main",
        speculative_draft_kv_cache_dtype=None,
        speculative_dspark_block_size=7,
        speculative_num_draft_tokens=8,
        tp_size=1,
        pp_size=1,
        kv_cache_dtype="auto",
    )
    b = SimpleNamespace(**{**a.__dict__, "speculative_dspark_block_size": 8})
    assert DsparkPrefixCache.make_fingerprint(a) != DsparkPrefixCache.make_fingerprint(b)


def test_suffix_mark_does_not_promote_target_only_prefix():
    root = _Node(())
    parent = _Node(range(4), root)
    leaf = _Node(range(4, 8), parent)
    cache = _cache()
    cache.mark_inserted_path(parent, ready=True, fingerprint="v1")
    cache.mark_inserted_path(leaf, ready=True, fingerprint="v1", ready_from=4)
    assert cache.paired_prefix_length(leaf, "v1") == 8

    cold = _cache()
    cold.mark_inserted_path(leaf, ready=True, fingerprint="v1", ready_from=4)
    assert cold.paired_prefix_length(leaf, "v1") == 0


def test_fingerprint_mismatch_invalidates_sidecar():
    root = _Node(())
    leaf = _Node(range(4), root)
    cache = _cache()
    cache.mark_inserted_path(leaf, ready=True, fingerprint="v1")
    assert cache.paired_prefix_length(leaf, "v1") == 4
    assert cache.paired_prefix_length(leaf, "v2") == 0
    assert cache.stats()["tracked_tokens"] == 0
