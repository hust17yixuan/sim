"""DSpark AscendStore draft companion pages and joint prefix identity.

[Role]
  Price the DSpark prefix cache: the joint target/draft/auxiliary identity,
  the validated registered draft-kernel pages, and the per-rank versus
  TP-aggregate storage/transfer contract.  Target auxiliary scratch buffers
  are *not* persistent companion pages.

[Invariants]
  * ``DSparkPrefixCache`` registers ``draft_caches`` only, on the final PP
    stage.  Auxiliary layer IDs enter the model identity hash, but target
    hidden-state scratch is a separate transient workspace.
  * The identity is one SHA-256 over both complete model identities (name,
    revision, code revision, activation dtype, KV dtype, quantization and the
    full structural config), the ordered aux IDs, draft width, TP/PP, block
    size and prefix-match unit.  A BF16 -> FP32 change changes the digest.
  * A registered page must cover one logical block with a contiguous kernel,
    carry at least ``page_tokens * row_bytes`` physical bytes, keep scaled
    kernels adjacent and cover every declared draft group.
  * Restore is all-or-nothing per full block and uses the shared
    ``RestoreExtent`` validator: partial blocks, hits beyond the initialized
    prompt and missing block hashes are typed rejections.
  * Per-rank payload and TP-aggregate object/metadata counts are reported
    separately and never mixed in one number.
  * All dimensions are strict integers; booleans, fractional floats and
    negatives reject.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.dspark_prefix_registration import (
    DSparkPrefixRegistration, DSparkRegisteredPage)
from mindie_sim.nr_s7_validate import (
    RestoreExtent, dtype_elem_bytes, strict_bool, strict_dtype, strict_int,
    strict_text, typed_sequence,
)
from mindie_sim.schema.common import ModelingError


def _int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="dspark")


def _bool(name: str, value: Any) -> bool:
    return strict_bool(name, value, reason="dspark")


def _text(name: str, value: Any) -> str:
    return strict_text(name, value, reason="dspark")


@typed_inputs
@dataclass(frozen=True)
class ModelIdentity:
    """The source's complete model identity inputs for one model."""

    name: str
    activation_dtype: str
    kv_dtype: str
    revision: str = ""
    code_revision: str = ""
    quantization: str = ""
    config: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _text("model_name", self.name)
        strict_dtype("activation_dtype", self.activation_dtype,
                     reason="dspark")
        strict_dtype("kv_dtype", self.kv_dtype, reason="dspark")
        for name in ("revision", "code_revision", "quantization"):
            value = getattr(self, name)
            if type(value) is not str:
                raise ModelingError(f"dspark_{name}_invalid")
        if isinstance(self.config, (str, bytes)) or not isinstance(
                self.config, tuple):
            raise ModelingError("dspark_model_config_invalid")
        for item in self.config:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ModelingError("dspark_model_config_entry_invalid")
            key, value = item
            _text("config_key", key)
            if type(value) is not str:
                raise ModelingError(f"dspark_config_value_invalid:{key}")

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.name, "revision": self.revision,
                "code_revision": self.code_revision,
                "dtype": self.activation_dtype, "kv_dtype": self.kv_dtype,
                "quantization": self.quantization,
                "config": dict(self.config)}

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


@typed_inputs
@dataclass(frozen=True)
class DSparkPrefixKeys:
    """Joint target/draft/aux identity and per-TP object keys."""

    identity: str
    pp_size: int
    tp_size: int
    target_hash: str = ""
    draft_hash: str = ""
    aux_layers: tuple[int, ...] = ()
    block_size: int = 128
    draft_width: int = 0
    prefix_match_unit: int = 128
    dcp_size: int = 1
    dcp_in_identity: bool = False

    def __post_init__(self) -> None:
        _text("identity", self.identity)
        if len(self.identity) != 64 or any(
                char not in "0123456789abcdef" for char in self.identity):
            raise ModelingError("dspark_prefix_identity_not_sha256")
        _int("pp_size", self.pp_size, minimum=1)
        _int("tp_size", self.tp_size, minimum=1)
        _int("block_size", self.block_size, minimum=1)
        _int("draft_width", self.draft_width, minimum=0)
        _int("prefix_match_unit", self.prefix_match_unit, minimum=1)
        _int("dcp_size", self.dcp_size, minimum=1)
        _bool("dcp_in_identity", self.dcp_in_identity)
        for name in ("target_hash", "draft_hash"):
            value = getattr(self, name)
            if type(value) is not str or (value and len(value) != 64):
                raise ModelingError(f"dspark_{name}_invalid")
        if not isinstance(self.aux_layers, tuple) or not all(
                type(value) is int and value >= 0 for value in self.aux_layers):
            raise ModelingError("dspark_aux_layers_invalid")
        if self.aux_layers:
            if list(self.aux_layers) != sorted(set(self.aux_layers)):
                raise ModelingError("dspark_aux_layers_not_canonical")

    def make_key(self, block_hash: str, tp_rank: int) -> str:
        _text("block_hash", block_hash)
        _int("tp_rank", tp_rank, minimum=0)
        if tp_rank >= self.tp_size:
            raise ModelingError(f"dspark_tp_rank_out_of_range:{tp_rank}")
        return (f"dspark-prefix-v1@{self.identity}@pp{self.pp_size - 1}"
                f"@tp{tp_rank}@{block_hash}")

    def make_hit_check_keys(self, block_hash: str) -> list[str]:
        return [self.make_key(block_hash, rank) for rank in range(self.tp_size)]

    def dcp_compatible(self, other: "DSparkPrefixKeys") -> bool:
        """Separate DCP compatibility check when DCP is not in the digest.

        Upstream ``DSparkPrefixKeys.from_config`` hashes ``(tp_size, pp_size)``
        only.  A caller that runs more than one decode-context size must either
        set ``include_dcp_in_key`` at key construction or validate this field
        explicitly before reusing a remote object.
        """
        if not isinstance(other, DSparkPrefixKeys):
            raise ModelingError("dspark_prefix_keys_invalid")
        if self.dcp_in_identity or other.dcp_in_identity:
            return self.identity == other.identity
        return self.dcp_size == other.dcp_size

    def to_dict(self) -> dict[str, Any]:
        return {"identity": self.identity, "pp_size": self.pp_size,
                "tp_size": self.tp_size, "target_hash": self.target_hash,
                "draft_hash": self.draft_hash,
                "aux_layers": list(self.aux_layers),
                "block_size": self.block_size,
                "draft_width": self.draft_width,
                "prefix_match_unit": self.prefix_match_unit,
                "dcp_size": self.dcp_size,
                "dcp_in_identity": self.dcp_in_identity}


@typed_inputs
def dspark_prefix_keys(*, target: ModelIdentity, draft: ModelIdentity,
                       aux_layers: Sequence[int], tp_size: int, pp_size: int,
                       block_size: int, prefix_match_unit: int,
                       draft_width: int, dcp_size: int = 1,
                       include_dcp_in_key: bool = False) -> DSparkPrefixKeys:
    """Derive the joint identity from both complete model identities.

    Upstream hashes ``(tp_size, pp_size)`` but not the decode-context size.
    ``include_dcp_in_key=True`` extends the digest; otherwise callers must use
    ``DSparkPrefixKeys.dcp_compatible`` before sharing a remote object.
    """
    if not isinstance(target, ModelIdentity) or not isinstance(
            draft, ModelIdentity):
        raise ModelingError("dspark_prefix_model_identity_invalid")
    _int("tp_size", tp_size, minimum=1)
    _int("pp_size", pp_size, minimum=1)
    _int("block_size", block_size, minimum=1)
    _int("prefix_match_unit", prefix_match_unit, minimum=1)
    _int("draft_width", draft_width, minimum=0)
    _int("dcp_size", dcp_size, minimum=1)
    _bool("include_dcp_in_key", include_dcp_in_key)
    if isinstance(aux_layers, (str, bytes)) or not isinstance(aux_layers, Sequence):
        raise ModelingError("dspark_aux_layers_invalid")
    aux = tuple(aux_layers)
    if not aux or not all(type(value) is int and value >= 0 for value in aux):
        raise ModelingError("dspark_aux_layers_invalid")
    if list(aux) != sorted(set(aux)):
        raise ModelingError("dspark_aux_layers_not_canonical")
    identity = {
        "target": target.to_dict(), "draft": draft.to_dict(),
        "aux_layers": list(aux), "draft_width": draft_width,
        "tp_size": tp_size, "pp_size": pp_size, "block_size": block_size,
        "prefix_match_unit": prefix_match_unit,
    }
    if include_dcp_in_key:
        identity["dcp_size"] = dcp_size
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return DSparkPrefixKeys(
        digest, pp_size, tp_size, target.digest(), draft.digest(), aux,
        block_size, draft_width, prefix_match_unit, dcp_size,
        include_dcp_in_key)


@typed_inputs
@dataclass(frozen=True)
class DSparkPrefixPage:
    """One registered draft-kernel page; mirrors the reference checks."""

    name: str
    group_id: int
    shape0: int
    shape1: int
    numel: int
    element_size: int
    stride0_elements: int
    row_bytes: int = 0

    def __post_init__(self) -> None:
        _text("page_name", self.name)
        for name in ("group_id", "shape0", "shape1", "numel", "element_size",
                     "stride0_elements", "row_bytes"):
            _int(name, getattr(self, name), minimum=0)
        _int("row_bytes", self.row_bytes, minimum=1)
        _int("shape0", self.shape0, minimum=1)
        _int("shape1", self.shape1, minimum=1)
        _int("numel", self.numel, minimum=1)
        _int("element_size", self.element_size, minimum=1)
        _int("stride0_elements", self.stride0_elements, minimum=1)

    @property
    def kernel_bytes(self) -> int:
        return self.numel * self.element_size

    @property
    def kernel_stride_bytes(self) -> int:
        return self.stride0_elements * self.element_size

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "group_id": self.group_id,
                "shape0": self.shape0, "shape1": self.shape1,
                "numel": self.numel, "element_size": self.element_size,
                "stride0_elements": self.stride0_elements,
                "kernel_bytes": self.kernel_bytes,
                "kernel_stride_bytes": self.kernel_stride_bytes,
                "row_bytes": self.row_bytes}


@typed_inputs
def dspark_prefix_page_bytes(*, pages: Sequence[DSparkPrefixPage],
                             num_blocks: int, block_size: int,
                             hash_block_size: int,
                             group_ids: Sequence[int],
                             block_sizes: Sequence[int],
                             draft_group_ids: Sequence[int] | None = None,
                             require_hash_equals_block: bool = False) -> dict[str, Any]:
    """Validate AscendStore draft companion pages and return exact geometry.

    Every accepted page becomes an immutable registration record.  The returned
    mapping carries those records and their coverage inventory alongside the
    scalar totals, so a later plan cannot be rebuilt from forgeable scalars.
    """
    _int("num_blocks", num_blocks, minimum=1)
    _int("block_size", block_size, minimum=1)
    _int("hash_block_size", hash_block_size, minimum=1)
    _bool("require_hash_equals_block", require_hash_equals_block)
    if block_size % hash_block_size:
        raise ModelingError("dspark_prefix_block_hash_alignment_invalid")
    if require_hash_equals_block and block_size != hash_block_size:
        raise ModelingError("dspark_prefix_hash_block_must_equal_logical_block")
    if isinstance(pages, (str, bytes)) or not isinstance(pages, Sequence) or not pages:
        raise ModelingError("dspark_prefix_pages_empty")
    records = tuple(pages)
    if not all(isinstance(page, DSparkPrefixPage) for page in records):
        raise ModelingError("dspark_prefix_page_invalid")
    for name, value in (("groups", group_ids), ("block_sizes", block_sizes),
                        ("draft_groups", draft_group_ids)):
        if value is None and name == "draft_groups":
            continue
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ModelingError(f"dspark_prefix_{name}_invalid")
    groups = tuple(group_ids)
    if not groups or not all(type(value) is int and value >= 0
                             for value in groups):
        raise ModelingError("dspark_prefix_layer_groups_invalid")
    if len(set(groups)) != len(groups):
        raise ModelingError("dspark_prefix_layer_groups_duplicate")
    groups = tuple(sorted(groups))
    page_groups = {page.group_id for page in records}
    if set(groups) != page_groups:
        raise ModelingError("dspark_prefix_layer_group_set_mismatch")
    if draft_group_ids is not None:
        declared = tuple(draft_group_ids)
        if (not declared or not all(type(value) is int and value >= 0
                                    for value in declared)):
            raise ModelingError("dspark_prefix_draft_groups_invalid")
        if not set(declared) <= set(groups):
            raise ModelingError("dspark_prefix_draft_group_uncovered")
    if not block_sizes or not all(type(value) is int and value >= 1
                                  for value in block_sizes):
        raise ModelingError("dspark_prefix_block_sizes_invalid")
    total_size = total_stride = 0
    details: list[dict[str, Any]] = []
    registered: list[DSparkRegisteredPage] = []
    for page in records:
        if not 0 <= page.group_id < len(block_sizes):
            raise ModelingError("dspark_prefix_group_id_out_of_range")
        if block_sizes[page.group_id] != block_size:
            raise ModelingError("dspark_prefix_groups_must_share_block_size")
        if page.shape0 % num_blocks:
            raise ModelingError("dspark_prefix_shape0_not_divisible_by_blocks")
        scale = page.shape0 // num_blocks
        if scale * page.shape1 != block_size:
            raise ModelingError("dspark_prefix_kernel_pages_do_not_cover_block")
        if page.row_bytes and page.kernel_bytes < page.shape1 * page.row_bytes:
            raise ModelingError(
                "dspark_prefix_page_capacity_too_small:"
                f"{page.kernel_bytes}<{page.shape1 * page.row_bytes}")
        kernel_bytes = page.kernel_bytes
        kernel_stride = page.kernel_stride_bytes
        if kernel_stride < kernel_bytes:
            raise ModelingError("dspark_prefix_kernel_stride_smaller_than_page")
        if scale > 1 and kernel_stride != kernel_bytes:
            raise ModelingError("dspark_prefix_scaled_pages_not_adjacent")
        size = kernel_bytes * scale
        stride = kernel_stride * scale
        total_size += size
        total_stride += stride
        region_size = (num_blocks - 1) * stride + size
        details.append({"name": page.name, "scale": scale, "size": size,
                        "stride": stride, "region_size": region_size})
        registered.append(DSparkRegisteredPage(
            page=page, scale=scale, size=size, stride=stride,
            region_size=region_size))
    registration = DSparkPrefixRegistration(
        pages=tuple(registered), num_blocks=num_blocks, block_size=block_size,
        hash_block_size=hash_block_size, page_bytes=total_size,
        stride_bytes=total_stride, objects_per_key=len(records),
        groups=groups, block_sizes=tuple(block_sizes),
        draft_groups=tuple(sorted(page_groups)),
        loaded_pages=records,
        require_hash_equals_block=require_hash_equals_block)
    return {"num_blocks": num_blocks, "block_size": block_size,
            "hash_block_size": hash_block_size,
            "page_bytes": total_size, "stride_bytes": total_stride,
            "pages": details, "objects_per_key": len(records),
            "draft_groups": sorted(page_groups),
            "block_sizes": list(block_sizes),
            "require_hash_equals_block": require_hash_equals_block,
            "registration": registration}


@typed_inputs
@dataclass(frozen=True)
class DSparkPrefixPlan:
    """Immutable draft companion storage/transfer plan for one token extent.

    The plan carries the registered page geometry (``num_blocks``, draft group
    coverage, shared block sizes and the hash-equals-block invariant) so a
    later restore cannot be reconstructed from five mutable scalars.
    """

    keys: DSparkPrefixKeys
    page_bytes: int
    page_stride_bytes: int
    block_size: int
    hash_block_size: int
    objects_per_key: int
    aux_layer_ids: tuple[int, ...]
    target_hidden_size: int
    target_dtype: str
    tokens: int
    prompt_tokens: int
    block_hash_count: int
    num_blocks: int
    draft_groups: tuple[int, ...]
    block_sizes: tuple[int, ...]
    require_hash_equals_block: bool = False
    metadata_bytes_per_object: int = 0
    aux_scratch_tokens: int = 0
    exact: bool = True
    registration: DSparkPrefixRegistration | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.keys, DSparkPrefixKeys):
            raise ModelingError("dspark_prefix_keys_invalid")
        if not isinstance(self.registration, DSparkPrefixRegistration):
            raise ModelingError("dspark_prefix_registration_missing")
        for name in ("page_bytes", "page_stride_bytes", "block_size",
                     "hash_block_size", "objects_per_key", "target_hidden_size",
                     "num_blocks"):
            _int(name, getattr(self, name), minimum=1)
        if self.page_stride_bytes < self.page_bytes:
            raise ModelingError("dspark_prefix_page_stride_too_small")
        for name in ("tokens", "prompt_tokens", "block_hash_count",
                     "metadata_bytes_per_object", "aux_scratch_tokens"):
            _int(name, getattr(self, name), minimum=0)
        strict_dtype("target_dtype", self.target_dtype, reason="dspark")
        _bool("exact", self.exact)
        _bool("require_hash_equals_block", self.require_hash_equals_block)
        if not isinstance(self.aux_layer_ids, tuple) or not all(
                type(value) is int and value >= 0
                for value in self.aux_layer_ids):
            raise ModelingError("dspark_aux_layers_invalid")
        if not self.aux_layer_ids or list(self.aux_layer_ids) != sorted(
                set(self.aux_layer_ids)):
            raise ModelingError("dspark_aux_layers_not_canonical")
        if self.aux_layer_ids != self.keys.aux_layers:
            raise ModelingError(
                "dspark_prefix_plan_aux_layers_mismatch_key")
        if self.keys.block_size != self.block_size:
            raise ModelingError("dspark_prefix_key_block_size_mismatch")
        if self.require_hash_equals_block and (
                self.block_size != self.hash_block_size):
            raise ModelingError(
                "dspark_prefix_hash_block_must_equal_logical_block")
        if not isinstance(self.draft_groups, tuple) or not self.draft_groups \
                or not all(type(value) is int and value >= 0
                           for value in self.draft_groups):
            raise ModelingError("dspark_prefix_draft_groups_invalid")
        if list(self.draft_groups) != sorted(set(self.draft_groups)):
            raise ModelingError("dspark_prefix_draft_groups_not_canonical")
        if not isinstance(self.block_sizes, tuple) or not self.block_sizes \
                or not all(type(value) is int and value == self.block_size
                           for value in self.block_sizes):
            raise ModelingError("dspark_prefix_block_sizes_invalid")
        if max(self.draft_groups) >= len(self.block_sizes):
            raise ModelingError("dspark_prefix_draft_group_out_of_range")
        registered = self.registration
        claims = registered.claims()
        for name, claimed in (
                ("num_blocks", self.num_blocks),
                ("block_size", self.block_size),
                ("hash_block_size", self.hash_block_size),
                ("page_bytes", self.page_bytes),
                ("stride_bytes", self.page_stride_bytes),
                ("objects_per_key", self.objects_per_key),
                ("draft_groups", tuple(self.draft_groups)),
                ("block_sizes", tuple(self.block_sizes)),
                ("require_hash_equals_block",
                 self.require_hash_equals_block)):
            if claimed != claims[name]:
                raise ModelingError(
                    f"dspark_prefix_geometry_mutated:{name}:"
                    f"{claimed!r}!={claims[name]!r}")
        if not registered.require_hash_equals_block and \
                self.require_hash_equals_block:
            raise ModelingError(
                "dspark_prefix_hash_contract_not_registered")
        # Extent/contract errors take precedence over the registered block
        # count: a restore beyond the initialized prompt is not a pool-size
        # message.
        self.extent.validated_blocks()
        if self.blocks_stored > registered.num_blocks:
            raise ModelingError(
                "dspark_prefix_extent_exceeds_registered_blocks:"
                f"{self.blocks_stored}>{registered.num_blocks}")

    @property
    def extent(self) -> RestoreExtent:
        return RestoreExtent(
            prefix_hit_tokens=self.tokens, prompt_tokens=self.prompt_tokens,
            block_size=self.block_size, hash_block_size=self.hash_block_size,
            block_hash_count=self.block_hash_count, exact=self.exact)

    @property
    def blocks_stored(self) -> int:
        return self.tokens // self.block_size

    @property
    def stored_tokens(self) -> int:
        return self.blocks_stored * self.block_size

    def validate_block_table(self, blocks_by_group: Mapping[int, Sequence[int]],
                             indices: Sequence[int]) -> None:
        """Mirror source ``_copy``: every group's extent and physical block id."""
        if not isinstance(blocks_by_group, Mapping):
            raise ModelingError("dspark_prefix_block_table_invalid")
        if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
            raise ModelingError("dspark_prefix_restore_indices_invalid")
        for index in indices:
            if type(index) is not int or not 0 <= index < self.blocks_stored:
                raise ModelingError("dspark_prefix_block_table_index_out_of_range")
        for group in self.draft_groups:
            blocks = blocks_by_group.get(group)
            if blocks is None or isinstance(blocks, (str, bytes)) \
                    or not isinstance(blocks, Sequence):
                raise ModelingError(
                    f"dspark_prefix_block_table_group_missing:{group}")
            if len(blocks) < self.blocks_stored:
                raise ModelingError(
                    "dspark_prefix_block_table_too_short:"
                    f"{len(blocks)}<{self.blocks_stored}")
            for index in indices:
                value = blocks[index]
                if type(value) is not int or not 0 <= value < self.num_blocks:
                    raise ModelingError(
                        "dspark_prefix_block_id_out_of_range:"
                        f"{value!r}@{index}")
                if value >= self.registration.num_blocks:
                    raise ModelingError(
                        "dspark_prefix_block_id_outside_registered_region:"
                        f"{value}>{self.registration.num_blocks - 1}")

    @property
    def companion_page_bytes(self) -> float:
        """Draft-only registered page; target aux scratch is not included."""
        return float(self.page_bytes)

    @property
    def aux_scratch_bytes(self) -> float:
        if not self.aux_scratch_tokens:
            return 0.0
        return (float(len(self.aux_layer_ids) * self.aux_scratch_tokens
                      * self.target_hidden_size)
                * dtype_elem_bytes("target_dtype", self.target_dtype,
                                   reason="dspark"))

    @property
    def per_rank_storage_bytes(self) -> float:
        return (float(self.blocks_stored) * self.companion_page_bytes
                + float(self.object_count) * self.metadata_bytes_per_object)

    @property
    def aggregate_storage_bytes(self) -> float:
        return self.per_rank_storage_bytes * self.keys.tp_size

    @property
    def per_rank_transfer_bytes(self) -> float:
        return self.per_rank_storage_bytes

    @property
    def aggregate_transfer_bytes(self) -> float:
        return self.aggregate_storage_bytes

    @property
    def draft_kv_bytes(self) -> float:
        return float(self.blocks_stored) * self.companion_page_bytes

    @property
    def object_count(self) -> int:
        return self.blocks_stored

    @property
    def aggregate_object_count(self) -> int:
        return self.object_count * self.keys.tp_size

    def to_dict(self) -> dict[str, Any]:
        return {"identity": self.keys.identity,
                "aux_layers": list(self.aux_layer_ids),
                "scope": "per_rank",
                "draft_page_bytes": float(self.page_bytes),
                "companion_page_bytes": self.companion_page_bytes,
                "aux_scratch_bytes": self.aux_scratch_bytes,
                "aux_scratch_persistent": False,
                "blocks_stored": self.blocks_stored,
                "stored_tokens": self.stored_tokens,
                "draft_kv_bytes": self.draft_kv_bytes,
                "per_rank_storage_bytes": self.per_rank_storage_bytes,
                "aggregate_storage_bytes": self.aggregate_storage_bytes,
                "per_rank_transfer_bytes": self.per_rank_transfer_bytes,
                "aggregate_transfer_bytes": self.aggregate_transfer_bytes,
                "object_count": self.object_count,
                "aggregate_object_count": self.aggregate_object_count,
                "num_blocks": self.num_blocks,
                "draft_groups": list(self.draft_groups),
                "block_sizes": list(self.block_sizes),
                "require_hash_equals_block": self.require_hash_equals_block,
                "registration": self.registration.to_dict(),
                "keys": self.keys.to_dict()}


@typed_inputs(errors={"aux_layer_ids": "dspark_aux_layers_invalid"})
def dspark_prefix_plan(*, keys: DSparkPrefixKeys, page_geometry: dict[str, Any],
                       aux_layer_ids: Sequence[int], target_hidden_size: int,
                       target_dtype: str, tokens: int, prompt_tokens: int,
                       block_hash_count: int,
                       metadata_bytes_per_object: int = 0,
                       aux_scratch_tokens: int = 0,
                       exact: bool = True) -> DSparkPrefixPlan:
    """Price draft pages plus an explicitly separate aux scratch workspace.

    The immutable registration evidence built by ``dspark_prefix_page_bytes``
    is revalidated as a whole.  ``num_blocks``, ``page_bytes``, the hash
    contract and every region extent are compared against those accepted
    ``_Page`` records, so mutating one scalar cannot shrink the region or relax
    the registration's required invariant.
    """
    if not isinstance(page_geometry, dict):
        raise ModelingError("dspark_prefix_geometry_invalid")
    registration = page_geometry.get("registration")
    if not isinstance(registration, DSparkPrefixRegistration):
        raise ModelingError("dspark_prefix_geometry_registration_missing")
    required = ("page_bytes", "stride_bytes", "block_size", "hash_block_size",
                "objects_per_key", "num_blocks", "draft_groups",
                "block_sizes", "require_hash_equals_block")
    try:
        values = {name: page_geometry[name] for name in required}
    except KeyError as error:
        raise ModelingError(
            f"dspark_prefix_geometry_missing:{error.args[0]}") from error
    claims = registration.claims()
    for name in ("page_bytes", "stride_bytes", "block_size", "hash_block_size",
                 "objects_per_key", "num_blocks"):
        _int(f"geometry_{name}", values[name], minimum=1)
    _bool("geometry_require_hash_equals_block",
          values["require_hash_equals_block"])
    for name, minimum in (("draft_groups", 0), ("block_sizes", 1)):
        values[name] = typed_sequence(name, values[name], int, nonempty=True)
        for value in values[name]:
            _int(name, value, minimum=minimum)
    for name in ("page_bytes", "stride_bytes", "block_size", "hash_block_size",
                 "objects_per_key", "num_blocks"):
        if values[name] != claims[name]:
            raise ModelingError(
                f"dspark_prefix_geometry_mutated:{name}:"
                f"{values[name]!r}!={claims[name]!r}")
    if tuple(values["draft_groups"]) != claims["draft_groups"]:
        raise ModelingError("dspark_prefix_geometry_mutated:draft_groups")
    if tuple(values["block_sizes"]) != claims["block_sizes"]:
        raise ModelingError("dspark_prefix_geometry_mutated:block_sizes")
    if bool(values["require_hash_equals_block"]) != \
            claims["require_hash_equals_block"]:
        raise ModelingError(
            "dspark_prefix_hash_contract_cannot_be_relaxed")
    if isinstance(aux_layer_ids, (str, bytes)) or not isinstance(aux_layer_ids, Sequence):
        raise ModelingError("dspark_aux_layers_invalid")
    aux = tuple(aux_layer_ids)
    if not all(type(value) is int and value >= 0 for value in aux):
        raise ModelingError("dspark_aux_layers_invalid")
    return DSparkPrefixPlan(
        keys=keys, page_bytes=claims["page_bytes"],
        page_stride_bytes=claims["stride_bytes"],
        block_size=claims["block_size"],
        hash_block_size=claims["hash_block_size"],
        objects_per_key=claims["objects_per_key"], aux_layer_ids=aux,
        target_hidden_size=target_hidden_size, target_dtype=target_dtype,
        tokens=tokens, prompt_tokens=prompt_tokens,
        block_hash_count=block_hash_count, num_blocks=claims["num_blocks"],
        draft_groups=tuple(claims["draft_groups"]),
        block_sizes=tuple(claims["block_sizes"]),
        require_hash_equals_block=claims["require_hash_equals_block"],
        metadata_bytes_per_object=metadata_bytes_per_object,
        aux_scratch_tokens=aux_scratch_tokens, exact=exact,
        registration=registration)


@typed_inputs
def aux_feature_bytes(*, layer_count: int, tokens: int, hidden_size: int,
                      dtype: str) -> float:
    """Auxiliary target hidden-state payload for one transient token extent."""
    _int("layer_count", layer_count, minimum=0)
    _int("tokens", tokens, minimum=0)
    _int("hidden_size", hidden_size, minimum=1)
    return (float(layer_count * tokens * hidden_size)
            * dtype_elem_bytes("dtype", dtype, reason="dspark"))


__all__ = ["DSparkPrefixKeys", "DSparkPrefixPage", "DSparkPrefixPlan",
           "DSparkPrefixRegistration", "DSparkRegisteredPage",
           "ModelIdentity", "aux_feature_bytes", "dspark_prefix_keys",
           "dspark_prefix_page_bytes", "dspark_prefix_plan"]
