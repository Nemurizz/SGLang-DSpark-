"""DSpark target/draft paired prefix-cache metadata.

The target and DSpark draft pools use the same logical token locations, but a
target radix-cache entry only proves that target KV exists. This module keeps
the additional readiness state needed before a DSpark request may reuse that
prefix. It intentionally does not allocate or copy KV tensors; the draft KV
writer populates the existing draft pool and calls mark_inserted_path when the
request is ready to be cached.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from sglang.srt.environ import envs



@dataclass
class DsparkNodeState:
    """Readiness of the draft KV corresponding to one radix node."""

    ready_prefix_len: int = 0
    fingerprint: str = "default"
    last_access_ns: int = 0


@dataclass(frozen=True)
class DsparkRecoveryPlan:
    """How much of a target hit must be re-prefilled to recover draft KV.

    The target radix tree can be longer than the paired sidecar. In that
    case the target-only suffix is intentionally re-prefilled: target hidden
    states are needed to materialize the missing draft KV. The already paired
    prefix remains cached, so this is suffix recovery rather than a full
    prompt prefill.
    """

    target_hit_len: int
    paired_hit_len: int
    recovery_start: int
    recovery_tokens: int
    mode: str


class DsparkPrefixCache:
    """Sidecar metadata for DSpark paired target/draft prefix reuse.

    The sidecar is metadata-only. The target radix node owns physical token
    locations and the draft model writes its KV to the same locations. A node
    is usable only when every segment on the path from root to the matched node
    has a complete, compatible draft KV state.
    """

    def __init__(self, token_to_kv_pool_allocator: Any = None):
        self.enabled = bool(envs.SGLANG_DSPARK_PREFIX_CACHE.get())
        self.shadow = bool(envs.SGLANG_DSPARK_PREFIX_CACHE_SHADOW.get())
        self.active = self.enabled or self.shadow
        self.budget_fraction = max(
            0.0,
            min(1.0, float(envs.SGLANG_DSPARK_PREFIX_CACHE_BUDGET_FRACTION.get())),
        )
        self.capacity_tokens = self._resolve_capacity(token_to_kv_pool_allocator)
        self.max_tokens = (
            int(self.capacity_tokens * self.budget_fraction)
            if self.capacity_tokens > 0
            else 0
        )
        if self.capacity_tokens > 0:
            self.max_tokens = max(1, self.max_tokens)

        self._node_resolver = None
        self._current_fingerprint: str | None = None
        self._states: dict[Any, DsparkNodeState] = {}
        self._lru: OrderedDict[Any, None] = OrderedDict()
        self._tracked_tokens = 0
        self._stats = {
            "match_calls": 0,
            "target_hit_tokens": 0,
            "paired_hit_tokens": 0,
            "ready_tokens": 0,
            "metadata_evictions": 0,
            "partial_recovery_calls": 0,
            "partial_recovery_tokens": 0,
            "full_recovery_calls": 0,
            "full_recovery_tokens": 0,
        }

    @staticmethod
    def _resolve_capacity(allocator: Any) -> int:
        if allocator is None:
            return 0
        for name in ("size", "size_full", "max_total_num_tokens"):
            value = getattr(allocator, name, None)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    value = None
            if value is not None:
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    pass
        return 0

    def set_node_resolver(self, resolver) -> None:
        """Attach a resolver for unified-cache NodeId handles."""

        self._node_resolver = resolver

    def resolve_node(self, node: Any) -> Any:
        if node is None:
            return None
        resolver = getattr(self, "_node_resolver", None)
        if resolver is not None:
            try:
                return resolver(node)
            except (KeyError, TypeError):
                return node
        return node

    @staticmethod
    def make_fingerprint(spec: Any = None, serving: Any = None) -> str:
        """Build a stable compatibility key for target/draft KV reuse."""

        override = envs.SGLANG_DSPARK_PREFIX_CACHE_FINGERPRINT.get()
        if override:
            return str(override)
        if spec is None:
            return "default"
        fields = (
            getattr(spec, "speculative_draft_model_path", None),
            getattr(spec, "speculative_draft_model_revision", None),
            getattr(spec, "speculative_draft_kv_cache_dtype", None),
            getattr(spec, "speculative_dspark_block_size", None),
            getattr(spec, "speculative_num_draft_tokens", None),
            getattr(spec, "tp_size", None),
            getattr(spec, "pp_size", None),
            getattr(spec, "kv_cache_dtype", None),
            getattr(serving, "weight_version", None),
        )
        payload = repr(fields).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

    def is_usable(self) -> bool:
        return self.active

    def should_reuse(self) -> bool:
        """Whether matching may change scheduling (shadow mode is read-only)."""

        return self.enabled

    def reset(self) -> None:
        self._states.clear()
        self._lru.clear()
        self._tracked_tokens = 0
        self._current_fingerprint = None

    def _sync_fingerprint(self, fingerprint: str) -> None:
        fingerprint = str(fingerprint)
        if (
            self._current_fingerprint is not None
            and self._current_fingerprint != fingerprint
        ):
            # A draft/target weight or layout change must not retain stale
            # readiness metadata. Physical target KV remains owned by radix.
            self.reset()
        self._current_fingerprint = fingerprint

    def invalidate_all(self) -> None:
        self.reset()

    def mark_inserted_path(
        self,
        last_node: Any,
        *,
        ready: bool,
        fingerprint: str = "default",
        ready_from: int = 0,
    ) -> None:
        """Mark only newly materialized suffix nodes as draft-ready.

        ``ready_from`` is the request's protected prefix boundary. Existing
        target-only nodes before that boundary must not be promoted to paired
        state merely because a later suffix was materialized. A node that
        straddles the boundary is left unchanged; the normal radix insert/match
        path will split it at the boundary on the next insertion.
        """

        if not self.active or not ready or last_node is None:
            return
        self._sync_fingerprint(fingerprint)
        ready_from = max(0, int(ready_from))
        node = self.resolve_node(last_node)
        path = []
        while node is not None and getattr(node, "parent", None) is not None:
            path.append(node)
            node = node.parent
        offset = 0
        for node in reversed(path):
            key_len = len(getattr(node, "key", ()))
            node_start, node_end = offset, offset + key_len
            if node_end > ready_from and node_start >= ready_from and key_len > 0:
                self._set_state(node, key_len, fingerprint)
            offset = node_end
        self._enforce_budget()

    def on_node_split(self, new_parent: Any, child: Any, split_len: int) -> None:
        """Split readiness exactly like RadixCache._split_node."""

        if not self.active:
            return
        old = self._states.pop(child, None)
        self._lru.pop(child, None)
        if old is None:
            return
        self._tracked_tokens = max(
            0, self._tracked_tokens - int(old.ready_prefix_len)
        )
        old_len = max(0, int(old.ready_prefix_len))
        parent_ready = min(old_len, int(split_len))
        child_ready = max(0, old_len - int(split_len))
        if parent_ready:
            self._set_state(new_parent, parent_ready, old.fingerprint)
        if child_ready:
            self._set_state(child, child_ready, old.fingerprint)

    def on_node_deleted(self, node: Any) -> None:
        if not self.active:
            return
        state = self._states.pop(node, None)
        self._lru.pop(node, None)
        if state is not None:
            self._tracked_tokens = max(
                0, self._tracked_tokens - int(state.ready_prefix_len)
            )

    def _set_state(self, node: Any, ready_len: int, fingerprint: str) -> None:
        ready_len = max(
            0, min(int(ready_len), len(getattr(node, "key", ())))
        )
        old = self._states.get(node)
        if old is not None:
            self._tracked_tokens -= int(old.ready_prefix_len)
        state = DsparkNodeState(
            ready_prefix_len=ready_len,
            fingerprint=str(fingerprint),
            last_access_ns=time.monotonic_ns(),
        )
        if ready_len:
            self._states[node] = state
            self._lru.pop(node, None)
            self._lru[node] = None
            self._tracked_tokens += ready_len
        else:
            self._states.pop(node, None)
            self._lru.pop(node, None)

    def _touch(self, node: Any) -> None:
        state = self._states.get(node)
        if state is None:
            return
        state.last_access_ns = time.monotonic_ns()
        self._lru.pop(node, None)
        self._lru[node] = None

    def _enforce_budget(self) -> None:
        if self.max_tokens <= 0:
            return
        while self._tracked_tokens > self.max_tokens and self._lru:
            node, _ = self._lru.popitem(last=False)
            state = self._states.pop(node, None)
            if state is not None:
                self._tracked_tokens = max(
                    0, self._tracked_tokens - int(state.ready_prefix_len)
                )
                self._stats["metadata_evictions"] += 1

    def paired_prefix_length(
        self, last_node: Any, fingerprint: str = "default"
    ) -> int:
        """Return the longest contiguous paired prefix ending at last_node."""

        self._stats["match_calls"] += 1
        if last_node is None:
            return 0
        self._sync_fingerprint(fingerprint)

        path = []
        node = self.resolve_node(last_node)
        while node is not None and getattr(node, "parent", None) is not None:
            path.append(node)
            node = node.parent

        paired_len = 0
        for node in reversed(path):
            state = self._states.get(node)
            if state is None or state.fingerprint != str(fingerprint):
                break
            node_len = len(getattr(node, "key", ()))
            if state.ready_prefix_len < node_len:
                paired_len += int(state.ready_prefix_len)
                break
            paired_len += node_len
            self._touch(node)
        return paired_len

    def plan_recovery(
        self,
        last_node: Any,
        *,
        target_hit_len: int,
        fingerprint: str = "default",
    ) -> DsparkRecoveryPlan:
        """Build an explicit suffix-recovery plan from a target cache hit.

        This does not copy target KV into the draft pool. The missing suffix
        is recovered by the normal DSpark target prefill, which produces target
        hidden states and injects draft KV. Keeping this plan explicit lets the
        scheduler distinguish partial recovery from a cold prefill and avoids
        re-prefilling the paired prefix.
        """

        target_hit_len = max(0, int(target_hit_len))
        paired_hit_len = self.paired_prefix_length(last_node, fingerprint)
        paired_hit_len = min(target_hit_len, max(0, int(paired_hit_len)))
        recovery_tokens = target_hit_len - paired_hit_len
        if paired_hit_len == target_hit_len:
            mode = "paired"
        elif paired_hit_len > 0:
            mode = "partial"
            self._stats["partial_recovery_calls"] += 1
            self._stats["partial_recovery_tokens"] += recovery_tokens
        else:
            mode = "full"
            if target_hit_len:
                self._stats["full_recovery_calls"] += 1
                self._stats["full_recovery_tokens"] += target_hit_len
        return DsparkRecoveryPlan(
            target_hit_len=target_hit_len,
            paired_hit_len=paired_hit_len,
            recovery_start=paired_hit_len,
            recovery_tokens=recovery_tokens,
            mode=mode,
        )

    def observe_match(self, target_tokens: int, paired_tokens: int) -> None:
        self._stats["target_hit_tokens"] += max(0, int(target_tokens))
        self._stats["paired_hit_tokens"] += max(0, int(paired_tokens))

    def stats(self) -> dict[str, int]:
        return {
            **self._stats,
            "tracked_tokens": int(self._tracked_tokens),
            "ready_tokens": int(self._tracked_tokens),
            "max_tokens": int(self.max_tokens),
        }
