"""[Role] Layer-3/4 model & framework piecewise assembly.
[Invariants] Deterministic subgraph evaluation; bounded cache; explicit state threading.
[Timeouts & Locks] Pure in-memory arithmetic; no locks or blocking waits.
[Side-Effect Policy] Side-effect free; immutable records, no file/network/device I/O.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from fractions import Fraction
import math
from typing import Any, Iterable, Mapping, Sequence
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan, ModelSpec, RequestSpec
from mindie_sim.schema.subgraphs import (
    ALL_ATTENTION_KINDS, BoundOperator, LATENT_ATTENTION_KINDS,
    LINEAR_ATTENTION_KINDS, OperatorTemplate, STANDARD_ATTENTION_KINDS,
    StaticSignature, WINDOW_SPARSE_ATTENTION_KINDS,
)
from mindie_sim.prediction.piecewise.compiler import (CompiledSubgraph, ImplementationPolicy, OperatorPiecewiseCompiler,
    PipeSupply)
from mindie_sim.prediction.piecewise.composition import (ContentionEnvelope, OperatorSegment, SerialResult, advance_state,
    contention_envelope, evaluate_serial_split, operator_segment, p2p_transfer_cost)
from mindie_sim.prediction.piecewise.geometry import AnalysisDomain, RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.subgraphs.builders_attention import gemma4_model_spec, _gemma_config, _gemma_global
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, InstantiatedModel, LayerInstances, instantiate_model
from mindie_sim.prediction.subgraphs.registry import TemplateRegistry
from mindie_sim.prediction.subgraphs.state import (CompressionState, MoeRoutingState, ResidualSlotState,
    SharedKvContract, StateView)
_PIPELINE_DOMAIN = AnalysisDomain(max_tokens=1_048_576, max_q=1_048_576, max_s_kv=1_048_576, max_seq=1_048_576, max_moe_rows=8_388_608,
    max_query_tiles=262_144, max_kv_tiles=262_144, max_index_rows=134_217_728, max_work=10**15)
_SOFTMAX_MACRO_ATTENTION = STANDARD_ATTENTION_KINDS | LATENT_ATTENTION_KINDS
_ACTIVE_KINDS = frozenset(
    ALL_ATTENTION_KINDS
    | {"kv_cache", "kv_compressor", "indexer_prolog", "lightning_indexer",
       "moe_router", "moe_token_dispatcher", "routed_experts", "spec_decode"})
_ATTENTION = frozenset({"attention", "mla"})
_FAST_FIELDS = ( "attention", "ffn", "architecture", "compress_ratio", "indexer", "window",
    "num_heads", "num_kv_heads", "head_dim", "intermediate_size", "num_experts",
    "top_k", "moe_intermediate_size", "num_shared_experts", "shared_intermediate_size",
    "shared_expert_gate", "linear_heads", "linear_head_dim", "linear_key_heads",
    "linear_key_head_dim", "linear_value_heads", "conv_kernel", "qk_norm",
    "q_output_gate", "mla_variant", "mla_output_gate", "expert_latent_size",
    "o_groups", "o_lora_rank", "residual_block_size", "ffn_activation", "cache_kind", "compressor_kind", )
def _frac(value: Any) -> Fraction: return value if isinstance(value, Fraction) else Fraction(str(value))
def _ms(value: Any) -> float: return float(value)


class _RationalMs(Fraction):
    """Exact rational duration whose ndigits ``round()`` returns ``float``."""

    def __add__(self, other: Any) -> Any:
        res = super().__add__(other)
        return _RationalMs(res) if isinstance(res, Fraction) else res

    def __sub__(self, other: Any) -> Any:
        res = super().__sub__(other)
        return _RationalMs(res) if isinstance(res, Fraction) else res

    def __round__(self, ndigits: int | None = None) -> Any:
        return super().__round__() if ndigits is None else round(float(self), ndigits)

# Layer ids in ``theta`` are state/port bindings, not static work parameters.
# They must not split the structural template cache: a repeated layer with a
# different KV/index owner has the same physical cost once its state port is
# ready.  Readiness is already part of ``_state_digest``.
_LAYER_PORT_KEYS = frozenset({
    "owner_layer", "kv_owner", "require_kv", "require_index",
    "require_selection", "compression_owner", "index_owner",
    "selection_owner", "index_k_owner", "topk_owner", "layer", "layer_id",
})

def _static_theta_key(theta: Any) -> tuple:
    """Theta groups minus layer/owner port bindings.

    The remaining groups are exactly the static work, quant, layout, tile and
    parallel parameters that can change a compiled cost.
    """
    extras = tuple(sorted((key, value) for key, value in theta.extras
                          if key not in _LAYER_PORT_KEYS))
    return (theta.role, theta.matrix, theta.heads, theta.experts,
            theta.quant, theta.parallel, theta.tiles, extras)

def _instance_compile_key(instance: Any) -> tuple:
    """Layer-independent structural identity for one bound subgraph."""
    return (instance.template.key(),
            tuple((op.template.key(), _static_theta_key(op.theta))
                  for op in instance.operators),
            instance.owner_layer is not None,
            instance.metadata)

def _features_with_hardware_ranks(features: FeatureConfig, supply: PipeSupply) -> FeatureConfig:
    """Copy the device node size onto feature metadata before instantiation.

    Builders freeze ``ranks_per_node`` into collective theta.  The historical
    default is 8, which splits a 16-rank single-node group into two nodes.
    An explicit metadata value wins.  The supply field is the machine profile
    (``PipeSupply.invocations``), not a measured span.
    """
    meta = dict(features.metadata)
    if int(meta.get("ranks_per_node", 0) or 0) > 0:
        return features
    declared = int(dict(supply.invocations).get("ranks_per_node", 0) or 0)
    if declared < 1:
        return features
    meta["ranks_per_node"] = declared
    return replace(features, metadata=tuple(sorted(meta.items())))
def pcp_rank_geometry(geometry: WorkloadGeometry, pcp: int,
                      rank: int | None = None) -> WorkloadGeometry:
    """Local rank geometry for a declared PCP prefill schedule.

    PCP shards the current query chunks, keeps the older context replicated,
    and chooses the true worst-cost rank from the public split_pcp_pairs
    contract.  Each owned early/late range becomes an independent local query
    chunk with its absolute prefix context; no blind division by cp is used.
    """
    pcp = max(1, int(pcp))
    if pcp <= 1 or geometry.phase_name != "prefill":
        return geometry
    from mindie_sim.catalog.context_parallel import (
        pcp_rank_chunk_ranges, split_pcp_pairs,
    )
    costs = [0] * pcp
    for request in geometry.requests:
        split = split_pcp_pairs(request.q_len, request.context, pcp)
        for r, amount in enumerate(split.per_rank):
            costs[r] += amount
    if rank is None:
        rank = max(range(pcp), key=costs.__getitem__)
    if not 0 <= int(rank) < pcp:
        raise ModelingError(f"pcp_rank_invalid:{rank}:{pcp}")
    rank = int(rank)
    requests = []
    for owner, request in enumerate(geometry.requests):
        last_selected = (max(end for _start, end in request.query_ranges) - 1
                         if request.query_ranges else request.q_len - 1)
        for start, end in pcp_rank_chunk_ranges(request.q_len, pcp, rank):
            if end > start:
                requests.append(replace(
                    request, q_len=end - start,
                    context=request.context + start,
                    query_ranges=(), phase=request.phase,
                    owner_index=owner,
                    parent_query_ranges=tuple(request.query_ranges),
                    emit=(start <= last_selected < end),
                    emit_position=(request.context + last_selected
                                   if start <= last_selected < end else -1)))
    if not requests:
        return geometry
    return replace(geometry, requests=tuple(requests),
                   local_batch=geometry.batch)


@dataclass(frozen=True)
class OperatorRow:
    layer: int; kind: str; role: str; cost: Any
    padded_tokens: int; active_tokens: int; geometry: str = "active"
    @property
    def duration_ms(self) -> Fraction:
        return self.cost.duration_ms
    @property
    def native_call(self) -> dict | None:
        return getattr(self.cost, "native_call", None)
    @property
    def native_occurrence(self) -> dict | None:
        return getattr(self.cost, "native_occurrence", None)
    @property
    def phase(self) -> str:
        return str((self.native_occurrence or {}).get("phase") or "")
    def to_dict(self) -> dict:
        return {"layer": self.layer, "kind": self.kind, "role": self.role,
                "name": self.cost.name, "family": self.cost.family, "duration_ms": str(self.cost.duration_ms),
                "dominant_pipe": self.cost.dominant_pipe, "padded_tokens": self.padded_tokens,
                "active_tokens": self.active_tokens, "geometry": self.geometry, "fused": self.cost.family == "fused",
                "phase": self.phase, "native_call": self.native_call,
                "native_occurrence": self.native_occurrence}
@dataclass(frozen=True)
class SubgraphPrediction:
    instance_id: str; kind: str; variant: str; role: str
    duration_ms: Fraction; rows: tuple[OperatorRow, ...]
    fusion_results: tuple[Any, ...] = ()
    #: External state ports of the actual instance, not of a reused
    #: representative layer.  They carry owner/readiness/representation.
    state_in: tuple[Any, ...] = ()
    state_out: tuple[Any, ...] = ()
    instance: Any = None
    states: tuple[Any, ...] = ()
    geometries: tuple[Any, ...] = ()
    def to_dict(self) -> dict:
        return {"instance_id": self.instance_id, "kind": self.kind, "variant": self.variant, "role": self.role,
                "duration_ms": str(self.duration_ms), "operators": [row.to_dict() for row in self.rows],
                "fusions": [item.to_dict() for item in self.fusion_results],
                "state_in": [port.to_dict() for port in self.state_in],
                "state_out": [port.to_dict() for port in self.state_out]}
@dataclass(frozen=True)
class LayerPrediction:
    layer: int
    stage: int
    duration_ms: Fraction
    subgraphs: tuple[SubgraphPrediction, ...]
    state_in: StateView
    state_out: StateView
    group: int = -1
    reused: bool = False
    contention: ContentionEnvelope | None = None
    operator_duration_ms: Fraction = Fraction(0)
    @property
    def execution_duration_ms(self) -> float | None:
        """Actual makespan of submitted events, never a summed isolated plan."""
        times = [float(event.time) for event in self.execution_submission
                 if getattr(event, "time", None) is not None]
        if not times:
            return None
        return max(times) - min(times)
    @property
    def rows(self) -> tuple[OperatorRow, ...]:
        return tuple(row for item in self.subgraphs for row in item.rows)
    def to_dict(self) -> dict:
        return {"layer": self.layer, "stage": self.stage, "duration_ms": str(self.duration_ms),
                "operator_duration_ms": str(self.operator_duration_ms), "group": self.group, "reused": self.reused,
                "subgraphs": [item.to_dict() for item in self.subgraphs],
                "contention": self.contention.to_dict() if self.contention else None}
def selected_geometry(active: WorkloadGeometry,
                      emit_mask: Sequence[bool] | None = None) -> WorkloadGeometry | None:
    """One selected-logit row per completing logical request.

    PCP rank chunks are one logical request.  Only the chunk owning the final
    logical query token emits; emit_mask is indexed by logical request index.
    Ordinary requests keep the historical one-row behavior.
    """
    requests = []
    owned = any(request.owner_index >= 0 for request in active.requests)
    for index, request in enumerate(active.requests):
        if request.q_len <= 0:
            continue
        if owned and request.owner_index >= 0:
            if not request.emit:
                continue
            logical = request.owner_index
            if emit_mask is not None and (logical >= len(emit_mask)
                                          or not emit_mask[logical]):
                continue
        else:
            logical = index
            if emit_mask is not None and index >= len(emit_mask):
                continue
            if emit_mask is not None and not emit_mask[index]:
                continue
        if request.phase == "verify":
            q_len, context = request.q_len, request.context
        elif owned and request.owner_index >= 0 and request.emit_position >= 0:
            q_len, context = 1, request.emit_position
        else:
            q_len, context = 1, request.context + request.q_len - 1
        requests.append(RequestGeometry(
            q_len, context, phase=request.phase, replica=request.replica,
            draft_len=request.draft_len, owner_index=logical, emit=True))
    if not requests:
        return None
    return WorkloadGeometry(
        tuple(requests), media_tokens=active.media_tokens,
        phase=active.phase_name, local_batch=len(requests))


_SPEC_PHASES = frozenset({"verify", "draft", "draft_prefill"})


def physical_selected_geometry(active: WorkloadGeometry) -> WorkloadGeometry | None:
    """One terminal logits row per scheduled request, independent of emit.

    The genuine R31 non-spec runner (model_runner_v1.py L1634-1643) uses
    ``np.ones(num_reqs)`` and ``query_start_loc[1:num_reqs+1]-1``: every
    scheduled partial-prefill request is physically sampled, then invalid
    emissions are discarded later (L1480-85, L3038+).  This geometry is the
    physical sample/model-logit scope, not the emitted/committed token count.

    Spec/draft/verify and padded distributed geometry keep their original
    source path and return ``None``.
    """
    if active.phase_name in _SPEC_PHASES:
        return None
    if any(request.phase in _SPEC_PHASES for request in active.requests):
        return None
    if (int(active.physical_tokens) > int(active.tokens)
            or int(active.graph_bucket) > int(active.tokens)):
        return None
    requests = []
    for index, request in enumerate(active.requests):
        if request.q_len <= 0:
            continue
        if request.phase in _SPEC_PHASES:
            return None
        # Non-owned requests use the scheduled index as their logical key,
        # matching selected_geometry; owned chunks keep their owner index.
        owner_index = request.owner_index if request.owner_index >= 0 else index
        requests.append(RequestGeometry(
            1, request.context + request.q_len - 1, phase=request.phase,
            replica=request.replica, draft_len=request.draft_len,
            owner_index=owner_index, emit=request.emit))
    if not requests:
        return None
    return WorkloadGeometry(
        tuple(requests), media_tokens=active.media_tokens,
        phase=active.phase_name, local_batch=len(requests))


def producing_selected_geometry(active: WorkloadGeometry,
                               selected: WorkloadGeometry | None) -> WorkloadGeometry | None:
    """The producing token rows plus the legitimate selected logit rows.

    The selected geometry stores one completing row per request, so it no
    longer carries the producing tensor length.  Rebuild the active producing
    requests with an explicit ``query_ranges`` selection of the same terminal
    row.  The result keeps ``tokens``/``effective_tokens`` as the producing
    rows and ``selected_tokens``/``producing_lm_head_rows`` as the legitimate
    selected rows, without inventing either count.
    """
    if selected is None:
        return None
    terminals = {terminal.owner_index: terminal
                 for terminal in selected.requests}
    requests = []
    for index, base in enumerate(active.requests):
        key = base.owner_index if base.owner_index >= 0 else index
        terminal = terminals.pop(key, None)
        if terminal is None:
            # The producing tensor still includes this request's scheduled
            # rows; a zero-length range contributes no selected logit row.
            requests.append(replace(base, query_ranges=((0, 0),)))
            continue
        if str(terminal.phase or "") == "verify":
            start, end = 0, int(base.q_len)
        else:
            start = int(terminal.context) - int(base.context)
            end = start + 1
        if start < 0 or end > int(base.q_len) or end <= start:
            return None
        requests.append(replace(base, query_ranges=((start, end),)))
    if terminals or not requests:
        return None
    return replace(active, requests=tuple(requests),
                   local_batch=len(requests))


@dataclass(frozen=True)
class ModelPrediction:
    layers: tuple[LayerPrediction, ...]
    globals: tuple[SubgraphPrediction, ...]
    total_ms: Fraction
    state_out: StateView
    active: WorkloadGeometry
    padded: WorkloadGeometry | None
    supply: PipeSupply
    evaluation_count: int = 0
    template_groups: int = 0
    execution_submission: tuple[Any, ...] = ()
    execution_final_event: Any = None
    submitted_event_count: int = 0
    submitted_operators: tuple[Any, ...] = ()
    compiled_count: int = 0
    cache_high_water: int = 0
    materialized_rows: bool = True
    @property
    def execution_duration_ms(self) -> float | None:
        """Actual makespan of submitted events, never a summed isolated plan."""
        times = [float(event.time) for event in self.execution_submission
                 if getattr(event, "time", None) is not None]
        if not times:
            return None
        return max(times) - min(times)
    @property
    def rows(self) -> tuple[OperatorRow, ...]:
        if not self.materialized_rows:
            return ()
        return (tuple(row for layer in self.layers for row in layer.rows)
                + tuple(row for item in self.globals for row in item.rows))
    def bottlenecks(self, limit: int = 8) -> tuple[tuple[str, float], ...]:
        families, pipes = defaultdict(Fraction), defaultdict(Fraction)
        for row in self.rows:
            families[row.cost.family] += row.cost.duration_ms
            for pipe, value in row.cost.times:
                pipes[pipe] += value
        merged = {f"family:{key}": value for key, value in families.items()}
        merged.update({f"pipe:{key}": value for key, value in pipes.items()})
        ordered = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
        return tuple((key, _ms(value)) for key, value in ordered[:limit])
    def to_dict(self) -> dict:
        return {"total_ms": str(self.total_ms), "active_tokens": self.active.tokens,
                "padded_tokens": (self.padded.effective_tokens if self.padded else self.active.tokens),
                "state": self.state_out.to_dict(), "layers": [item.to_dict() for item in self.layers],
                "globals": [item.to_dict() for item in self.globals], "bottlenecks": dict(self.bottlenecks()),
                "evaluation_count": self.evaluation_count, "template_groups": self.template_groups,
                "submitted_event_count": self.submitted_event_count,
                "execution_duration_ms": self.execution_duration_ms,
                "submitted_operator_count": len(self.submitted_operators),
                "compiled_count": self.compiled_count,
                "cache_high_water": self.cache_high_water,
                "execution_final_event": (None if self.execution_final_event is None
                                          else {"name": self.execution_final_event.name,
                                                "sequence": self.execution_final_event.sequence})}
@dataclass(frozen=True)
class LayerTemplateGroup:
    group_id: int; key: tuple; indices: tuple[int, ...]; representative: int
@dataclass(frozen=True)
class _CachedLayer:
    duration_ms: Fraction; subgraphs: tuple[SubgraphPrediction, ...]
    contention: ContentionEnvelope | None; operator_duration_ms: Fraction
class ModelPipeline:
    """A compiled model walk with layer-template aggregation and state ports."""
    def __init__(self, model: ModelSpec, features: FeatureConfig, supply: PipeSupply,
                 policy: ImplementationPolicy | None = None, *, registry: TemplateRegistry | None = None,
                 domain: AnalysisDomain | None = None):
        if not isinstance(supply, PipeSupply):
            raise ModelingError("model_pipeline_requires_pipe_supply")
        if policy is None:
            policy = ImplementationPolicy(cores=features.cores)
        features = _features_with_hardware_ranks(features, supply)
        self.model, self.features, self.supply, self.policy = model, features, supply, policy
        self.instantiated: InstantiatedModel = instantiate_model(model, features, registry=registry)
        self.compiler = OperatorPiecewiseCompiler(supply, self.policy, domain or _PIPELINE_DOMAIN)
        self._compiled: dict[tuple, CompiledSubgraph] = {}
        self._compiled_bindings: dict[tuple, CompiledSubgraph] = {}
        self._cache: dict[tuple, _CachedLayer] = {}
        self._cache_high_water = 0
        self._kda_beta_ops: dict[tuple, Any] = {}
        self.groups = self._build_groups()
        self._group_of = {layer: group.group_id for group in self.groups for layer in group.indices}
    def _subgraph(self, instance: Any) -> CompiledSubgraph:
        # The structural template/theta is the compile key.  The actual layer
        # and owner occurrence are bindings: two same-template layers share
        # the compiled expressions but execute their own instance ports/state.
        key = instance.performance_key()
        if key not in self._compiled:
            self._compiled[key] = self.compiler.compile(
                instance, feature_metadata=self.features.metadata)
        representative = self._compiled[key]
        binding = (key, instance.layer, instance.owner_layer,
                   instance.instance_id)
        if binding not in self._compiled_bindings:
            self._compiled_bindings[binding] = (
                representative if representative.instance is instance
                else representative.bind(instance))
        return self._compiled_bindings[binding]
    def _layer_key(self, layer: LayerInstances) -> tuple:
        plan = self.model.layer_plans[layer.layer]
        structural = tuple(getattr(plan, name, None) for name in _FAST_FIELDS)
        slots = tuple((slot.kind, slot.variant, slot.role) for slot in layer.slots)
        # Owner vs consumer is structural; the concrete owner layer id is a
        # state port and must not create one template group per layer.
        owner = ("owner" if plan.owns_long_kv or plan.kv_owner_layer_id == plan.index
                 else "consumer")
        return (structural, slots, layer.stage, owner,
                tuple(_instance_compile_key(item) for item in layer.subgraphs))
    def _build_groups(self) -> tuple[LayerTemplateGroup, ...]:
        buckets: dict[tuple, list[int]] = {}
        for layer in self.instantiated.layers:
            buckets.setdefault(self._layer_key(layer), []).append(layer.layer)
        return tuple(LayerTemplateGroup(index, key, tuple(indices), indices[0])
                     for index, (key, indices) in enumerate(buckets.items()))
    @property
    def globals(self) -> tuple[Any, ...]:
        return self.instantiated.globals
    @property
    def draft_subgraph(self) -> Any | None:
        return next((item for item in self.globals if item.kind == "spec_decode"), None)
    def shared_kv_contracts(self) -> tuple[SharedKvContract, ...]:
        owners: dict[int, list[int]] = defaultdict(list)
        tp = max(1, self.features.tp)
        kv_bytes = dtype_bytes(self.features.kv_dtype)
        row_bytes = {}
        for plan in self.model.layer_plans:
            if plan.attention == "mla" or plan.cache_kind == "window_compressed":
                elements = plan.cache_row_elements
            else:
                heads = int(plan.num_kv_heads or self.model.num_kv_heads)
                elements = max(1, math.ceil(heads / tp)) * 2 * int( plan.head_dim or self.model.head_dim)
            row_bytes[plan.index] = int(elements * kv_bytes)
        stage_of = self.instantiated.stage_of
        for plan in self.model.layer_plans:
            owner = int(plan.kv_owner_layer_id)
            if 0 <= owner < plan.index and plan.attention in {"gqa", "mla"}:
                owners[owner].append(plan.index)
        contracts = []
        for owner, consumers in sorted(owners.items()):
            cross = any(stage_of[consumer] != stage_of[owner] for consumer in consumers)
            contracts.append(SharedKvContract( owner, tuple(consumers), row_bytes[owner], cross_stage=cross,
                transfer_bytes=row_bytes[owner] if cross else 0))
        return tuple(contracts)
    @staticmethod
    def _selected_rows(active: WorkloadGeometry, emit_mask: Sequence[bool] | None = None ) -> WorkloadGeometry | None:
        return selected_geometry(active, emit_mask)

    def _o0_physical_source(self) -> bool:
        """Whether the compiled O0 row is the declared R31 non-spec runner."""
        if int(self.features.pp or 1) != 1:
            return False
        for item in self.globals:
            if item.kind != "lm_head_sampler":
                continue
            compiled = self._subgraph(item)
            return any(
                op.name == "hidden_select"
                and str(op.operator.theta.find("row_binding", "")) == "selected"
                and str(op.operator.theta.find("source_runner_class", ""))
                == "NPUModelRunner"
                for op in compiled.operators)
        return False

    def _geometry_for(self, kind: str, operator: Any, active: WorkloadGeometry, padded: WorkloadGeometry | None,
                      selected: WorkloadGeometry | None = None, dsa_local: WorkloadGeometry | None = None,
                      dsa_local_padded: WorkloadGeometry | None = None ) -> tuple[WorkloadGeometry, str]:
        if kind == "lm_head_sampler":
            chosen = selected or active
            if operator.name == "hidden_select":
                if (self.features.final_norm_scope == "all"
                        and selected is not None):
                    producing = producing_selected_geometry(active, selected)
                    if producing is not None:
                        return producing, "producing_selected"
                return chosen, "selected"
            if operator.name == "final_norm" and self.features.final_norm_scope == "all":
                return active, "active"
            return chosen, "selected"
        if dsa_local is not None:
            # Shared DSA-CP projection: local query/attention/indexer, full
            # replicated KV/index cache.  Same helper is used by z._aggregate.
            from mindie_sim.catalog.context_parallel import project_dsa_cp_geometry
            projected = project_dsa_cp_geometry(
                kind, str(operator.name), str(operator.family), active, padded,
                dsa_local, dsa_local_padded)
            if projected is not None:
                return projected
        if padded is None:
            return active, "active"
        if kind in _SOFTMAX_MACRO_ATTENTION:
            return ((active, "active") if operator.family in _ATTENTION else (padded, "padded"))
        if kind in WINDOW_SPARSE_ATTENTION_KINDS | LINEAR_ATTENTION_KINDS:
            return ((active, "active") if operator.family in {"attention", "mla", "recurrent", "kv_access"}
                    else (padded, "padded"))
        if kind in _ACTIVE_KINDS:
            return active, "active"
        return padded, "padded"
    def _prediction(self, compiled: CompiledSubgraph, result: SerialResult,
                    active: WorkloadGeometry, padded: WorkloadGeometry | None,
                    selected: WorkloadGeometry | None = None,
                    dsa_local: WorkloadGeometry | None = None,
                    dsa_local_padded: WorkloadGeometry | None = None ) -> SubgraphPrediction:
        rows = []
        geometries = []
        for cost in result.costs:
            if cost.family == "fused":
                fusion = next((item for item in result.fusion_results if f"fused:{item.name}" == cost.name), None)
                first = (compiled.operator(fusion.semantic[0].name) if fusion else compiled.operators[0])
                used, geometry = self._geometry_for( compiled.instance.kind, first, active, padded, selected,
                    dsa_local, dsa_local_padded)
            else:
                used, geometry = self._geometry_for( compiled.instance.kind, compiled.operator(cost.name),
                    active, padded, selected, dsa_local, dsa_local_padded)
            rows.append(OperatorRow( compiled.instance.layer, compiled.instance.kind,
                str(dict(compiled.instance.metadata).get("role", "")), cost,
                used.effective_tokens, active.tokens, geometry))
            geometries.append(used)
        return SubgraphPrediction( compiled.instance.instance_id, compiled.instance.kind, compiled.instance.variant,
            str(dict(compiled.instance.metadata).get("role", "")),
            result.total_ms, tuple(rows), result.fusion_results,
            compiled.instance.state_in, compiled.instance.state_out,
            compiled.instance, result.states, tuple(geometries))
    def _run(self, compiled: Iterable[CompiledSubgraph], state: StateView,
             active: WorkloadGeometry, padded: WorkloadGeometry | None,
             fusion: bool, selected: WorkloadGeometry | None = None,
             skip: Sequence[str] = (),
             dsa_local: WorkloadGeometry | None = None,
             dsa_local_padded: WorkloadGeometry | None = None ) -> tuple[tuple[SubgraphPrediction, ...], StateView, Fraction]:
        current, predictions, duration = state, [], Fraction(0)
        self._last_contention: ContentionEnvelope | None = None
        a1_chain: list[OperatorSegment] = []
        idx_chain: list[OperatorSegment] = []
        a1_ms, idx_ms = Fraction(0), Fraction(0)
        for item in compiled:
            names = (tuple(contract.name for contract in item.instance.template.fusions) if fusion else ())
            result = evaluate_serial_split( item, active, current,
                geometry_for=lambda operator, kind=item.instance.kind:
                    self._geometry_for(kind, operator, active, padded, selected,
                                       dsa_local, dsa_local_padded)[0], fusion=names, skip=skip)
            if result.contention is not None:
                self._last_contention = result.contention
            if (self.features.multistream_attention_indexer
                    and item.instance.kind in LATENT_ATTENTION_KINDS):
                prolog_names = {"q_down", "q_norm", "q_up", "q_direct", "kv_down", "kv_norm", "rotary", "kv_store", "fused:mla_prolog"}
                prolog = [c for c in result.costs if c.name in prolog_names]
                for c in prolog:
                    a1_chain.append(operator_segment(c, self.supply))
                    a1_ms += c.duration_ms
            elif self.features.multistream_attention_indexer and item.instance.kind in {"indexer_prolog", "lightning_indexer"}:
                idx_chain.extend(operator_segment(c, self.supply) for c in result.costs)
                idx_ms += result.total_ms
            predictions.append(self._prediction(item, result, active, padded, selected,
                                                dsa_local, dsa_local_padded))
            current, duration = result.state, duration + result.total_ms
        if a1_chain and idx_chain:
            self._last_contention = contention_envelope(
                {}, self.supply, chains={"attention": a1_chain, "indexer": idx_chain})
            duration += self._last_contention.makespan_ms - (a1_ms + idx_ms)
        return tuple(predictions), current, duration
    def _evaluate_layer(self, layer: LayerInstances, state: StateView,
                        active: WorkloadGeometry, padded: WorkloadGeometry | None,
                        fusion: bool, selected: WorkloadGeometry | None = None,
                        skip: Sequence[str] = (),
                        dsa_local: WorkloadGeometry | None = None,
                        dsa_local_padded: WorkloadGeometry | None = None ) -> _CachedLayer:
        compiled = tuple(self._subgraph(item) for item in layer.subgraphs)
        indices = [index for index, item in enumerate(compiled) if item.instance.kind in {"moe_token_dispatcher", "routed_experts", "shared_experts"}]
        can_fork = (self.features.multistream_shared_expert and any(item.instance.kind == "shared_experts" for item in compiled)
                    and any(item.instance.kind in {"moe_token_dispatcher", "routed_experts"} for item in compiled))
        if not can_fork:
            predictions, _, duration = self._run(compiled, state, active, padded, fusion, selected, skip=skip,
                                                 dsa_local=dsa_local, dsa_local_padded=dsa_local_padded)
            return _CachedLayer(duration, predictions, self._last_contention, duration)
        first, last = min(indices), max(indices)
        prefix, suffix = compiled[:first], compiled[last + 1:]
        selected = compiled[first:last + 1]
        def is_join(item) -> bool:
            return (item.instance.kind == "shared_experts" and str(dict(item.instance.metadata).get("role", "")) == "join")
        joins = tuple(item for item in selected if is_join(item))
        shared = tuple(item for item in selected if item.instance.kind == "shared_experts" and not is_join(item))
        routed = tuple(item for item in selected if item.instance.kind in {"moe_token_dispatcher", "routed_experts"})
        branches = {}
        if shared:
            branches["shared"] = shared
        if routed:
            branches["routed"] = routed
        prefix_rows, current, prefix_ms = self._run(prefix, state, active, padded, fusion, selected, skip=skip,
                                                    dsa_local=dsa_local, dsa_local_padded=dsa_local_padded)
        base = current
        branch_rows, chains, branch_states = [], {}, {}
        for name, items in branches.items():
            branch_state = base
            chains[name] = []
            for item in items:
                result = evaluate_serial_split( item, active, branch_state,
                    geometry_for=lambda operator, kind=item.instance.kind:
                        self._geometry_for(kind, operator, active, padded, selected,
                                           dsa_local, dsa_local_padded)[0],
                    fusion=(tuple(contract.name for contract in item.instance.template.fusions) if fusion else ()),
                    skip=skip)
                chains[name].extend(operator_segment(cost, self.supply) for cost in result.costs)
                branch_rows.append(self._prediction(item, result, active, padded, selected,
                                                    dsa_local, dsa_local_padded))
                branch_state = result.state
            branch_states[name] = branch_state
        envelope = contention_envelope({}, self.supply, chains=chains)
        current = (branch_states.get("routed") or branch_states.get("shared") or base)
        join_rows, join_ms = [], Fraction(0)
        for item in joins:
            result = evaluate_serial_split( item, active, current,
                geometry_for=lambda operator, kind=item.instance.kind:
                    self._geometry_for(kind, operator, active, padded, selected,
                                       dsa_local, dsa_local_padded)[0],
                fusion=(tuple(contract.name for contract in item.instance.template.fusions) if fusion else ()),
                skip=skip)
            join_rows.append(self._prediction(item, result, active, padded, selected,
                                              dsa_local, dsa_local_padded))
            join_ms += result.total_ms
            current = result.state
        suffix_rows, current, suffix_ms = self._run(suffix, current, active, padded, fusion, selected, skip=skip,
                                                    dsa_local=dsa_local, dsa_local_padded=dsa_local_padded)
        duration = prefix_ms + envelope.makespan_ms + join_ms + suffix_ms
        return _CachedLayer(duration, prefix_rows + tuple(branch_rows)
                            + tuple(join_rows) + suffix_rows, envelope, duration)
    def _state_digest(self, state: StateView, layer: LayerInstances) -> tuple:
        plan = self.model.layer_plans[layer.layer]
        index = layer.layer
        compression = tuple(sorted( (key, value.ratio, value.consumed, value.rows, value.remainder)
            for key, value in state.compression if key == index))
        compression_requests = tuple(sorted( ((key[0], str(key[1])), value.ratio, value.consumed, value.rows,
             value.remainder) for key, value in state.compression_requests if key[0] == index))
        routing = tuple(sorted( (key, value.counts, value.top_k, value.valid_tokens, value.local_expert_ids)
            for key, value in state.routing if key == index))
        owner = int(plan.kv_owner_layer_id)
        selection = tuple(sorted(key for key, _ in state.selection_ready if key == index))
        cross_stage = tuple(sorted((key, value) for key, value in state.cross_stage_ready
                                   if key[0] == owner))
        return (compression, compression_requests, routing, (state.residual.block_size, state.residual.retained),
                (state.acceptance.accepted, state.acceptance.committed,
                 state.acceptance.rejected, state.acceptance.commits),
                state.is_kv_ready(owner) if owner >= 0 else None, selection, cross_stage)
    _OWNER_KEYS = ("owner_layer", "require_kv", "kv_owner",
                   "index_owner", "selection_owner")

    @classmethod
    def _instance_owners(cls, instance: Any) -> dict[str, int]:
        """The actual instance's declared read/write owner per operator."""
        owners: dict[str, int] = {}
        for operator in instance.operators:
            for key in cls._OWNER_KEYS:
                raw = operator.theta.find(key, None)
                if raw is None:
                    continue
                try:
                    owner = int(raw)
                except (TypeError, ValueError):
                    continue
                if owner >= 0:
                    owners[operator.name] = owner
                break
        return owners

    @staticmethod
    def _relabel_owner(value: Any, owner: int) -> Any:
        """Rebind nested caller ``owner_layer`` facts to the actual owner.

        Native occurrences may carry the owner both at the top level and in
        ``typed.attrs``/``prepare_args.attrs`` (for example ScatterPaCache).
        Static work/call identity stays; only this consumer fact changes.
        """
        if isinstance(value, Mapping):
            return {
                key: (owner if key == "owner_layer"
                      and not isinstance(item, bool)
                      and isinstance(item, int) else
                      ModelPipeline._relabel_owner(item, owner))
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(ModelPipeline._relabel_owner(item, owner)
                         for item in value)
        if isinstance(value, list):
            return [ModelPipeline._relabel_owner(item, owner)
                    for item in value]
        return value

    @classmethod
    def _rebind_cost(cls, cost: Any, layer: int, owner: int | None = None) -> Any:
        """Rebind an evaluated cost's occurrence to the actual instance.

        Work, times, materialisation and the selected call identity are
        static template results and stay reused.  The layer and nested caller
        owner facts are relocated to the actual layer/owner.
        """
        occurrence = getattr(cost, "native_occurrence", None)
        if not isinstance(occurrence, Mapping):
            return cost
        rebound = dict(occurrence)
        changed = False
        if int(rebound.get("layer", layer)) != layer:
            rebound["layer"] = layer
            changed = True
        if owner is not None:
            relabelled = cls._relabel_owner(rebound, owner)
            if relabelled != rebound:
                rebound = relabelled
                changed = True
        return replace(cost, native_occurrence=rebound) if changed else cost

    @classmethod
    def _rebind_fusion(cls, fusion: Any, layer: int,
                       owners: Mapping[str, int],
                       fallback_owner: int | None = None) -> Any:
        semantic = tuple(
            cls._rebind_cost(cost, layer, owners.get(cost.name))
            for cost in fusion.semantic)
        fused_owner = owners.get(fusion.semantic[0].name) if fusion.semantic else None
        if fused_owner is None:
            fused_owner = fallback_owner
        values = tuple(
            replace(value, layer=layer)
            for value in getattr(fusion, "values", ()))
        return replace(
            fusion, semantic=semantic, values=values,
            fused=cls._rebind_cost(fusion.fused, layer, fused_owner),
        )

    @classmethod
    def _rebind_subgraph(cls, item: SubgraphPrediction,
                         instance: Any, layer: int) -> SubgraphPrediction:
        owners = cls._instance_owners(instance)
        rows = tuple(
            replace(row, layer=layer,
                    cost=cls._rebind_cost(row.cost, layer,
                                          owners.get(row.cost.name)))
            for row in item.rows)
        fusions = tuple(
            cls._rebind_fusion(fusion, layer, owners, instance.owner_layer)
            for fusion in item.fusion_results)
        return replace(
            item, instance_id=instance.instance_id, rows=rows,
            fusion_results=fusions,
            state_in=instance.state_in, state_out=instance.state_out,
            instance=instance)

    @classmethod
    def _rebase(cls, subgraphs: tuple[SubgraphPrediction, ...],
                instances: Sequence[Any], layer: int) -> tuple[SubgraphPrediction, ...]:
        """Relocate cached template work onto this layer's real instances.

        The cached durations/work are reused unchanged; instance id, external
        state ports, row layer and native/fused occurrence layer are bound to
        the layer that actually consumes the result.  A cached row that has no
        matching actual instance is a binding gap, not a reason to keep the
        representative layer's identity.
        """
        remaining = list(instances)
        rebound: list[SubgraphPrediction] = []
        for item in subgraphs:
            match = next(
                (instance for instance in remaining
                 if (instance.kind == item.kind
                     and instance.variant == item.variant
                     and str(dict(instance.metadata).get("role", ""))
                     == item.role)),
                None)
            if match is None:
                match = next(
                    (instance for instance in remaining
                     if instance.kind == item.kind
                     and instance.variant == item.variant),
                    None)
            if match is None:
                raise ModelingError(
                    "layer_reuse_instance_binding_missing:"
                    f"{layer}:{item.kind}:{item.variant}:{item.role}")
            remaining.remove(match)
            rebound.append(cls._rebind_subgraph(item, match, layer))
        return tuple(rebound)

    def _advance_layer_state(self, layer: LayerInstances, state: StateView, active: WorkloadGeometry,
                             request_ids: Sequence[Any] | None = None,
                             commit_tokens: Sequence[int] | None = None) -> StateView:
        for instance in layer.subgraphs:
            if instance.kind == "kv_compressor":
                # The layer plan owns the true compression ratio.  A K1
                # placeholder for a zero-compressor/no-owner layer carries
                # ratio 0 and must not invent a ratio-1 cursor; genuine K1
                # keeps its source ratio 1/2/4/128 and advances as before.
                ratio = int(instance.operator("compress").theta.find("ratio", 0) or 0)
                if ratio <= 0:
                    continue
                ratio = max(1, ratio)
                records = ((
                    request_ids[(request.owner_index if request.owner_index >= 0 else index)]
                    if request_ids is not None and
                    (request.owner_index if request.owner_index >= 0 else index) < len(request_ids)
                    else (request.owner_index if request.owner_index >= 0 else index),
                    request.context,
                    int(commit_tokens[(request.owner_index if request.owner_index >= 0 else index)])
                    if commit_tokens is not None and
                    (request.owner_index if request.owner_index >= 0 else index) < len(commit_tokens)
                    else request.q_len)
                    for index, request in enumerate(active.requests))
                state, _ = state.update_compression(layer.layer, ratio, records)
                committed = [record for (owner, _), record in state.compression_requests
                             if owner == layer.layer]
                if committed:
                    # The owner cursor is the furthest committed request, so a
                    # rolled-back speculative tail cannot leave it ahead.
                    cursor = max(committed, key=lambda record: (record.consumed, record.rows))
                    values = dict(state.compression)
                    values[layer.layer] = cursor
                    state = replace(state, compression=tuple(sorted(values.items())))
            else:
                state = advance_state(self._subgraph(instance), state, active)
        return state

    def _kda_external_beta_cost(self, layer: LayerInstances, active: WorkloadGeometry):
        """Separate beta sigmoid when the declared option leaves it outside the scan.

        ``use_beta_sigmoid`` true folds that producer into the decode kernel.
        False keeps ``linear_beta_sigmoid``: one elementwise launch per KDA
        layer, ``heads`` elements per token and 5 ops per element, matching
        ``Builder.vector`` in ``recurrent.linear``. Undeclared stays unchanged.
        """
        declared = dict(self.features.metadata).get("kda_use_beta_sigmoid", None)
        if declared is None or int(declared) != 0 or active.phase_name != "decode":
            return None
        plan = self.model.layer_plans[layer.layer]
        if plan.attention != "kda" or not any(item.kind == "kda_linear_attn" for item in layer.subgraphs):
            return None
        heads = max(1, math.ceil(int(plan.linear_value_heads or plan.linear_heads or 1)
                                 / max(1, self.features.tp)))
        act_bytes = dtype_bytes(self.features.activation_dtype or "bf16")
        return self._kda_beta_operator(heads, act_bytes).evaluate(active, readiness=False)

    def _kda_beta_operator(self, heads: int, act_bytes: int):
        key = (heads, act_bytes, self.features.token_tile, self.features.cores)
        found = self._kda_beta_ops.get(key)
        if found is not None:
            return found
        template = OperatorTemplate("linear_beta_sigmoid", "elementwise", "sigmoid",
                                    static_slots=("elements",))
        theta = StaticSignature(
            "sigmoid",
            matrix=(("elements", heads),),
            quant=(("act_bytes", act_bytes),),
            tiles=(("mt", self.features.token_tile), ("cores", self.features.cores)),
            extras=(("ops_per_element", 5),
                    ("read_bytes", heads * act_bytes),
                    ("write_bytes", heads * 4)))
        compiled = self.compiler.compile_operator(
            BoundOperator(template, theta), kind="kda_linear_attn",
            feature_metadata=self.features.metadata)
        self._kda_beta_ops[key] = compiled
        return compiled

    def diagnostics(self, prediction: "ModelPrediction",
                    limit: int = 64) -> dict[str, Any]:
        """Bounded occurrence/resource plan for a completed forward.

        This is an observation of the same compiled values submitted above; it
        does not re-evaluate, retain per-request StateViews, or create a second
        execution path.  Stage order is the actual ``stage_work`` order.
        """
        work: dict[str, float] = {}
        elapsed: dict[str, float] = {}
        native_slots = 0
        owners: set[int] = set()
        stage_order: list[dict[str, Any]] = []
        for row in prediction.rows:
            for pipe, amount in row.cost.work:
                work[pipe] = work.get(pipe, 0.0) + float(amount)
            for pipe, amount in row.cost.times:
                elapsed[pipe] = elapsed.get(pipe, 0.0) + float(amount)
            occurrence = getattr(row.cost, "native_occurrence", None) or {}
            layer = occurrence.get("layer")
            if layer is not None:
                owners.add(int(layer))
            if getattr(row.cost, "native_program", None) is not None:
                native_slots += 1
            if len(stage_order) < max(0, int(limit)):
                stage_order.append({
                    "layer": row.layer, "kind": row.kind, "role": row.role,
                    "operator": row.cost.name,
                    "stages": len(row.cost.stage_work),
                    "stage_work": [len(stage) for stage in row.cost.stage_work],
                    "native": getattr(row.cost, "native_program", None) is not None,
                })
        state = prediction.state_out
        return {
            "compiled": self.compile_stats(),
            "rows": len(prediction.rows),
            "native_slots": native_slots,
            "owners": tuple(sorted(owners)),
            "resource_work": dict(sorted(work.items())),
            "resource_elapsed_ms": dict(sorted(elapsed.items())),
            "state_last_use": {
                "iteration": int(state.iteration),
                "kv_ready": len(state.kv_ready),
                "index_ready": len(state.index_ready),
                "selection_ready": len(state.selection_ready),
                "accepted_tokens": len(state.accepted_tokens),
            },
            "occurrence_stage_order": tuple(stage_order),
        }

    def compile_stats(self) -> dict[str, Any]:
        """Bounded static-compile and per-forward dynamic-cache diagnostics."""
        return {
            "compiled_subgraphs": len(self._compiled),
            "bound_subgraphs": len(self._compiled_bindings),
            "compiled_by_kind": dict(sorted(
                ((compiled.instance.kind,
                  sum(1 for item in self._compiled.values()
                      if item.instance.kind == compiled.instance.kind))
                 for compiled in self._compiled.values()),
                key=lambda item: item[0])),
            "template_groups": len(self.groups),
            "dynamic_cache_high_water": self._cache_high_water,
        }

    def forward(self, geometry: WorkloadGeometry, state: StateView | None = None, *,
                padded_geometry: WorkloadGeometry | None = None, fusion: bool = True, aggregate: bool = True,
                include_draft: bool = False, request_ids: Sequence[Any] | None = None,
                emit_mask: Sequence[bool] | None = None,
                acceptance: tuple[int, int] | None = None,
                commit_tokens: Sequence[int] | None = None,
                execution: Any | None = None, replica: int = 0,
                stage: int = 0, memory: Any | None = None,
                retain_rows: bool = True) -> ModelPrediction:
        self._cache.clear()
        submission_events: list[Any] = []
        submission_operators: list[Any] = []
        submission_ready = None
        def submit_rows(predictions: Sequence[SubgraphPrediction],
                        row_stage: int | None = None) -> None:
            nonlocal submission_ready
            if execution is None:
                return
            from .composition import submit_cost
            for item in predictions:
                compiled = (self._subgraph(item.instance)
                            if item.instance is not None else None)
                for row_index, row in enumerate(item.rows):
                    operator = None
                    if compiled is not None:
                        try:
                            operator = compiled.operator(row.cost.name)
                        except ModelingError:
                            operator = None
                    submitted = submit_cost(
                        row.cost, execution,
                        name=f"{item.instance_id}:{row.cost.name}",
                        ready=submission_ready, replica=replica,
                        stage=(stage if row_stage is None else row_stage),
                        supply=self.supply, operator=operator,
                        geometry=(item.geometries[row_index]
                                  if row_index < len(item.geometries) else active),
                        state=(item.states[row_index]
                               if row_index < len(item.states) else None),
                        memory=memory)
                    submission_operators.append(submitted)
                    submission_events.extend(submitted.events)
                    if submitted.events:
                        submission_ready = submitted.events[-1]
        current, active, padded = state or StateView(), geometry, padded_geometry
        if (self.features.pcp > 1
                and self.features.cp_plan in {"pcp", "pcp_dcp"}):
            active = pcp_rank_geometry(active, self.features.pcp)
            if padded is not None:
                padded = pcp_rank_geometry(padded, self.features.pcp)
        dsa_local = None
        dsa_local_padded = None
        if (self.features.cp_plan == "dsa_cp" and self.features.tp > 1
                and active.phase_name == "prefill"):
            from mindie_sim.catalog.context_parallel import dsa_cp_rank_geometry
            dsa_local = dsa_cp_rank_geometry(active, self.features.tp)
            if padded is not None:
                dsa_local_padded = dsa_cp_rank_geometry(padded, self.features.tp)
        if current.residual.retained or current.index_ready or current.selection_ready:
            current = replace(current, residual=ResidualSlotState(current.residual.block_size, ()),
                              index_ready=(), selection_ready=())
        if self.features.ep > 1:
            routing = dict(active.routing)
            cohort_tokens = active.for_subgraph("routed_experts").tokens
            for layer in self.instantiated.layers:
                if layer.layer in routing or not any(item.kind == "routed_experts" for item in layer.subgraphs):
                    continue
                experts, top_k, local = _layer_moe(self.model, layer.layer, self.features.ep)
                total_rows = cohort_tokens * max(1, top_k)
                base, extra = divmod(total_rows, experts)
                uniform = tuple(base + (1 if index < extra else 0) for index in range(experts))[::self.features.ep][:local]
                routing[layer.layer] = uniform + (0,) * (local - len(uniform))
            active = replace(active, routing=tuple(sorted(routing.items())))
        selected = self._selected_rows(active, emit_mask)
        physical_source = self._o0_physical_source()
        physical_selected = (physical_selected_geometry(active)
                             if physical_source else None)
        o0_selected = physical_selected or selected
        dsa_skip = (("dsa_cp_output_all_gather",)
                    if (self.features.cp_plan == "dsa_cp"
                        and active.phase_name != "prefill") else ())
        rows_in, current, total = self._run( tuple(self._subgraph(item) for item in self.globals
                  if item.kind == "vocab_parallel_embedding" or (item.kind == "mm_encoder_projector"
                                           and active.media_tokens > 0)), current, active, padded, fusion,
                  dsa_local=dsa_local, dsa_local_padded=dsa_local_padded)
        submit_rows(rows_in)
        layers, evaluations = [], 0
        for layer in self.instantiated.layers:
            block = int(self.model.layer_plans[layer.layer].residual_block_size or 0)
            if block and current.residual.block_size != block:
                current = replace(current, residual=ResidualSlotState( block, current.residual.retained))
            key = (self._group_of[layer.layer], geometry_key(active),
                   geometry_key(padded), self._state_digest(current, layer),
                   tuple((k, tuple(v)) for k, v in active.routing if k == layer.layer), bool(fusion), bool(aggregate))
            cached = self._cache.get(key); reused = cached is not None
            if cached is None:
                cached = self._evaluate_layer(layer, current, active, padded, fusion,
                                              selected, skip=dsa_skip,
                                              dsa_local=dsa_local,
                                              dsa_local_padded=dsa_local_padded)
                if aggregate:
                    self._cache[key] = cached
                    self._cache_high_water = max(
                        self._cache_high_water, len(self._cache))
                evaluations += 1
            next_state = self._advance_layer_state(layer, current, active, request_ids, commit_tokens)
            beta = self._kda_external_beta_cost(layer, active)
            duration, operator_duration = cached.duration_ms, cached.operator_duration_ms
            subgraphs = self._rebase(cached.subgraphs, layer.subgraphs, layer.layer)
            if beta is not None:
                row = OperatorRow(layer.layer, "kda_linear_attn", "sigmoid",
                                  beta, active.effective_tokens, active.tokens,
                                  "active")
                beta_subgraph = SubgraphPrediction(
                    f"kda_linear_attn:linear_beta_sigmoid:L{layer.layer}",
                    "kda_linear_attn", "beta", "sigmoid",
                    beta.duration_ms, (row,))
                items = list(subgraphs)
                at = next((index for index, item in enumerate(items)
                           if item.kind == "kda_linear_attn"), len(items))
                items.insert(at, beta_subgraph)
                subgraphs = tuple(items)
                duration += beta.duration_ms
                operator_duration += beta.duration_ms
            layers.append(LayerPrediction( layer.layer, layer.stage, duration,
                subgraphs, current, next_state, self._group_of[layer.layer], reused,
                cached.contention, operator_duration))
            current, total = next_state, total + duration
        if self.features.pp > 1:
            for index in range(len(layers) - 1):
                if layers[index].stage >= layers[index + 1].stage:
                    continue
                transferred = p2p_transfer_cost( layers[index].layer, active.tokens, self.model.hidden_size,
                    dtype_bytes(self.features.activation_format), self.supply)
                row = OperatorRow(layers[index].layer, "PP", "transfer",
                                  transferred, active.tokens, active.tokens, "active")
                subgraph = SubgraphPrediction( f"PP:stage_transfer:L{layers[index].layer}", "PP", "p2p",
                    "transfer", transferred.duration_ms, (row,))
                layers[index] = replace( layers[index],
                    duration_ms=layers[index].duration_ms + transferred.duration_ms,
                    operator_duration_ms=(layers[index].operator_duration_ms + transferred.duration_ms),
                    subgraphs=layers[index].subgraphs + (subgraph,))
                total += transferred.duration_ms
        if physical_selected is not None:
            # The R31 non-spec runner physically samples every scheduled
            # request; invalid emissions are discarded later, not skipped
            # from the computation.
            skip = ()
        else:
            skip = (("hidden_select", "lm_head", "logits_gather", "sampling")
                    if selected is None else ())
            if selected is None and self.features.final_norm_scope == "selected":
                skip += ("final_norm",)
            if physical_source and physical_selected is None:
                # Spec/draft/verify or padded distributed keeps its original
                # source path; do not bind this non-spec producer row.
                skip += ("hidden_select",)
        # Submit the complete per-layer plan after beta/PP rows are attached.
        # A reused static template still executes its actual occurrence work.
        for layer_prediction in layers:
            submit_rows(layer_prediction.subgraphs, layer_prediction.stage)
        rows_out, current, out_ms = self._run( tuple(self._subgraph(item) for item in self.globals
                  if item.kind == "lm_head_sampler" or (include_draft and item.kind == "spec_decode")),
            current, active, padded, fusion, o0_selected, skip,
            dsa_local=dsa_local, dsa_local_padded=dsa_local_padded)
        submit_rows(rows_out, (layers[-1].stage if layers else stage))
        if acceptance is not None:
            current = current.commit(int(acceptance[0]), verified=int(acceptance[1]))
        return ModelPrediction(
            tuple(layers), rows_in + rows_out, _RationalMs(total + out_ms),
            current, active, padded, self.supply, evaluations,
            len(self.groups), tuple(submission_events), submission_ready,
            len(submission_events), tuple(submission_operators),
            len(self._compiled), self._cache_high_water,
            retain_rows)
def geometry_key(geometry: WorkloadGeometry | None) -> tuple:
    if geometry is None:
        return ()
    return (geometry.tokens, geometry.effective_tokens, geometry.graph_bucket, geometry.cohort_batch,
            geometry.physical_tokens, geometry.media_tokens, geometry.phase_name,
            tuple((request.q_len, request.context, request.phase,
                   request.query_ranges, request.replica, request.draft_len) for request in geometry.requests),
            tuple((layer, tuple(counts)) for layer, counts in geometry.routing))
def assemble_model(model: ModelSpec | Mapping[str, Any], *, supply: PipeSupply, features: FeatureConfig | None = None,
                   policy: ImplementationPolicy | None = None, registry: TemplateRegistry | None = None,
                   domain: AnalysisDomain | None = None, machine: Any | None = None) -> ModelPipeline:
    """Zero-parameter entry: config -> templates -> compiled model walk."""
    if isinstance(model, Mapping):
        family = str(model.get("model_type", "")).lower()
        spec = (gemma4_model_spec(model) if family.startswith("gemma") else ModelSpec.from_mapping(model))
    elif isinstance(model, ModelSpec):
        spec = model
    else: raise ModelingError("model_pipeline_requires_model_spec_or_mapping")
    if features is None:
        features = FeatureConfig.from_model(
            spec, machine=machine,
            **({"cores": policy.cores} if policy is not None else {}))
    return ModelPipeline(spec, features, supply, policy, registry=registry, domain=domain)
@dataclass(frozen=True)
class FrameworkConfig:
    supply: PipeSupply; policy: ImplementationPolicy = field(default_factory=ImplementationPolicy)
    graph_buckets: tuple[int, ...] = (); max_batch: int = 8; max_prefill_tokens: int = 0
    mtp_draft_len: int = 0; mtp_accept: int = 0; routing: tuple[tuple[int, tuple[int, ...]], ...] = ()
    graph_replay_ms: Fraction = Fraction(0); max_steps: int = 4096; max_retained_steps: int = 256
    def __post_init__(self) -> None:
        if self.max_batch < 1 or self.max_steps < 1 or self.max_prefill_tokens < 0:
            raise ModelingError("framework_scheduler_geometry_invalid")
        if not 0 <= self.mtp_accept <= self.mtp_draft_len:
            raise ModelingError("framework_mtp_accept_invalid")
        if self.graph_replay_ms < 0:
            raise ModelingError("framework_replay_invalid")
        object.__setattr__(self, "graph_buckets", tuple(sorted({int(value) for value in self.graph_buckets})))
@dataclass
class _RunRequest:
    spec: RequestSpec
    prefilled: int = 0
    generated: int = 0
    first_token_at: float | None = None
    last_token_at: float | None = None
    completion_at: float | None = None
    replica: int = 0
    itl_sum: float = 0.0
    itl_count: int = 0
@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    ttft_ms: float
    tpot_ms: float
    itl_ms: float
    completion_ms: float
    prompt_tokens: int
    output_tokens: int
@dataclass(frozen=True)
class FrameworkPrediction:
    body_duration_ms: float
    ttft_ms: float
    tpot_ms: float
    itl_ms: float
    requests: tuple[RequestMetrics, ...]
    steps: tuple[dict, ...]
    bottlenecks: tuple[tuple[str, float], ...]
    padding: dict
    sync: dict
    mtp: dict | None
    stop_reason: str = "drained"
    scheduler_feedback: tuple[Any, ...] = ()
    def to_dict(self) -> dict:
        return {"body_duration_ms": self.body_duration_ms, "ttft_ms": self.ttft_ms, "tpot_ms": self.tpot_ms,
                "itl_ms": self.itl_ms, "requests": [item.__dict__ for item in self.requests],
                "steps": list(self.steps), "bottlenecks": dict(self.bottlenecks),
                "stop_reason": self.stop_reason,
                "scheduler_feedback": list(self.scheduler_feedback),
                "padding": dict(self.padding), "sync": dict(self.sync), "mtp": dict(self.mtp) if self.mtp else None}
def _coerce_requests(requests: Iterable[Any]) -> tuple[RequestSpec, ...]:
    result = tuple(value if isinstance(value, RequestSpec) else RequestSpec.from_mapping(value, index)
                   for index, value in enumerate(requests))
    if not result:
        raise ModelingError("framework_requests_empty")
    return result
def _bucket_for(tokens: int, buckets: Sequence[int]) -> int:
    for value in buckets:
        if tokens <= value:
            return value
    return buckets[-1] if buckets else 0
def _layer_moe(model: Any, layer_index: int, ep: int) -> tuple[int, int, int]:
    """Expert cardinality, top-k and per-rank shard for one layer.

    A layer plan that declares ``num_experts`` or ``top_k`` replaces the model
    default.  The shard length is the even expert-parallel split of that count.
    """
    plan = model.layer_plans[layer_index] if 0 <= layer_index < len(model.layer_plans) else None
    experts = int((getattr(plan, "num_experts", 0) or 0) or model.num_experts or 1)
    top_k = int((getattr(plan, "top_k", 0) or 0) or model.top_k or 1)
    local = max(1, math.ceil(experts / max(1, int(ep))))
    return experts, top_k, local

def _padding(active: WorkloadGeometry, buckets: Sequence[int]) -> WorkloadGeometry | None:
    if not buckets:
        return None
    bucket = _bucket_for(active.tokens, buckets)
    if bucket <= active.tokens:
        return None
    # Padding changes the physical bucket only.  Routing, cohort and phase
    # stay on the geometry the step already declared.
    return replace(active, graph_bucket=bucket)
def _ep_skew(pipeline: ModelPipeline, active: WorkloadGeometry, config: FrameworkConfig) -> tuple[float, float]:
    """Per-rank complete layer envelopes, then one join maximum per layer (rank-permutation invariant)."""
    if pipeline.features.ep <= 1:
        return 0.0, 0.0
    cohort_tokens = active.for_subgraph("routed_experts").tokens
    layers = pipeline.instantiated.layers
    kv = tuple((layer.layer, "kv_cache") for layer in layers)
    index = tuple((layer.layer, "index") for layer in layers)
    selection = tuple((layer.layer, "selection") for layer in layers)
    compression = tuple((plan.index, CompressionState.start(int(plan.compress_ratio), int(plan.compress_ratio)))
                        for plan in pipeline.model.layer_plans if int(plan.compress_ratio or 0) > 1)
    seed = StateView(kv_ready=kv, index_ready=index, selection_ready=selection, compression=compression)
    layer_scores: list[list[Fraction]] = []
    for layer in layers:
        if not any(item.kind == "routed_experts" for item in layer.subgraphs):
            continue
        experts, top_k, local = _layer_moe(pipeline.model, layer.layer, pipeline.features.ep)
        base, extra = divmod(cohort_tokens * max(1, top_k), experts)
        uniform = tuple(base + (1 if index < extra else 0) for index in range(experts))
        counts = next((tuple(values) for key, values in config.routing if key == layer.layer), None) or uniform
        row = []
        for rank in range(pipeline.features.ep):
            shard = counts[rank * local:(rank + 1) * local]
            geometry = replace(active, routing=((layer.layer, shard),))
            row.append(pipeline._evaluate_layer(layer, seed, geometry, None, False).duration_ms)
        layer_scores.append(row)
    critical = sum((max(row) for row in layer_scores), Fraction(0))
    rank0 = sum((row[0] for row in layer_scores), Fraction(0))
    skew = sum((max(row) - min(row) for row in layer_scores), Fraction(0))
    return _ms(critical - rank0), _ms(skew)
def _draft_cost(pipeline: ModelPipeline, geometry: WorkloadGeometry, state: StateView,
                *, proposals: int | None = None,
                execution: Any | None = None) -> tuple[Fraction, Fraction]:
    draft, draft_ms, commit_ms = pipeline.draft_subgraph, Fraction(0), Fraction(0)
    if draft is None: return draft_ms, commit_ms
    heads = next((op for op in draft.operators if op.name == "draft_heads"), None)
    proposals = max(1, int(proposals or (heads.theta.find("draft_len", 1) if heads else 1) or 1))
    noncausal = bool(int(heads.theta.find("noncausal_block", 0) or 0)) if heads else (draft.template.variant == "dspark")
    body = tuple(item for item in draft.operators if item.name not in {"target_verify", "accept_commit"})
    commit = tuple(item for item in draft.operators if item.name == "accept_commit")
    steps = ((proposals, proposals),) if noncausal else tuple((1, step) for step in range(proposals))
    for q_mult, ctx_off in steps:
        step_geom = WorkloadGeometry(
            tuple(RequestGeometry(max(1, request.q_len) * q_mult, request.context + ctx_off,
                                  phase="draft", replica=request.replica)
                  for request in geometry.requests), phase="draft")
        for operator in body:
            compiled = pipeline.compiler.compile_operator(
                operator, kind="spec_decode",
                feature_metadata=pipeline.features.metadata)
            cost = compiled.evaluate(step_geom, state)
            if execution is None:
                draft_ms += cost.duration_ms
            else:
                from .composition import submit_cost
                before = float(execution.engine.now)
                submitted = submit_cost(
                    cost, execution, name=f"draft:{operator.name}",
                    supply=pipeline.supply)
                execution.engine.advance()
                after = (float(submitted.events[-1].time)
                         if submitted.events and submitted.events[-1].time is not None
                         else float(execution.engine.now))
                draft_ms += Fraction(str(max(0.0, after - before)))
    for operator in commit:
        compiled = pipeline.compiler.compile_operator(
            operator, kind="spec_decode",
            feature_metadata=pipeline.features.metadata)
        cost = compiled.evaluate(geometry, state)
        if execution is None:
            commit_ms += cost.duration_ms
        else:
            from .composition import submit_cost
            before = float(execution.engine.now)
            submitted = submit_cost(
                cost, execution, name=f"commit:{operator.name}",
                supply=pipeline.supply)
            execution.engine.advance()
            after = (float(submitted.events[-1].time)
                     if submitted.events and submitted.events[-1].time is not None
                     else float(execution.engine.now))
            commit_ms += Fraction(str(max(0.0, after - before)))
    return draft_ms, commit_ms
def predict_framework_cohort(model: ModelSpec | ModelPipeline | Mapping[str, Any],
                             requests: Iterable[Any], config: FrameworkConfig, *,
                             case: Any, execution: Any,
                             schedulers: Sequence[Any] | None = None,
                             memory: Any | None = None, lower: Any | None = None,
                             window: Mapping[str, Mapping[str, Any]] | None = None,
                             trace_limit: int = 2,
                             max_model_steps: int | None = None,
                             observer: Any | None = None,
                             ) -> FrameworkPrediction:
    """Adapt existing FrameworkConfig/requests to T14's shared cohort driver.

    The caller owns the selected Case/Execution/Scheduler/Memory and lowerer.
    ``run_cohort`` is the only service frontier; this function does not create
    a second Engine, does not add isolated forward durations as delays and does
    not reject DP>1 when the caller supplies per-replica schedulers.
    """
    from mindie_sim.prediction.api import run_cohort
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.frontend import AttentionPolicy, Frontend
    from mindie_sim.prediction.scheduler import Scheduler
    from mindie_sim.prediction.structure import parse_case

    case = parse_case(case) if isinstance(case, Mapping) else case
    specs = list(_coerce_requests(requests))
    parallel = case.runtime.parallel
    if schedulers is None:
        memory = memory if memory is not None else Memory(case)
        groups = [[] for _ in range(parallel.dp)]
        chains: dict[str, int] = {}
        for index, spec in enumerate(sorted(specs, key=lambda item: (
                item.arrival_ms, item.request_id))):
            replica = (chains.setdefault(spec.chain_id, index % parallel.dp)
                       if spec.chain_id else index % parallel.dp)
            groups[replica].append(spec)
        schedulers = tuple(
            Scheduler(case, group, memory, replica)
            for replica, group in enumerate(groups))
    else:
        schedulers = tuple(schedulers)
        memory = memory if memory is not None else getattr(
            schedulers[0], "memory", None)
    if memory is None:
        raise ModelingError("framework_cohort_memory_required")
    if window:
        for scheduler in schedulers:
            for request_id, values in dict(window).items():
                state = scheduler.by_id.get(request_id)
                if state is None:
                    continue
                if "computed" in values:
                    state.computed = int(values["computed"])
                if "generated" in values:
                    state.generated = int(values["generated"])
                if "release_ms" in values:
                    state.release_ms = max(state.request.arrival_ms,
                                           float(values["release_ms"]))
    lower = lower or Frontend(
        attention_policy=AttentionPolicy(**case.runtime.attention_policy))
    # T14's production entry binds the same T08 ledger and scheduler set onto
    # the caller's Execution. Mirror that public binding here; do not create a
    # second Memory/Execution.
    execution.memory = memory
    execution.memory_ledger = memory
    execution.schedulers = schedulers
    replay_ms = float(config.graph_replay_ms)
    budget_steps = (config.max_steps if max_model_steps is None
                    else max_model_steps)

    class _ReplayScheduler:
        """Put replay before the plan that consumes the current state."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def schedule(self, now, **kwargs):
            plan = self._inner.schedule(now, **kwargs)
            if plan is not None and plan.requests and replay_ms > 0:
                before = float(execution.engine.now)
                execution.engine.submit(
                    f"graph_replay:{plan.iteration}", latency=replay_ms)
                execution.engine.advance(horizon=before + replay_ms)
            return plan

    run_schedulers = (tuple(_ReplayScheduler(item) for item in schedulers)
                      if replay_ms > 0 else tuple(schedulers))
    run = run_cohort(
        case, execution, run_schedulers, memory, lower,
        trace_limit=trace_limit, max_model_steps=budget_steps,
        observer=observer)
    served_end = float(run.get("serving_end", run["window_end"]))
    if run.get("stop_reason") == "drained":
        # A finite safety cap can still drain naturally; complete the same
        # host/client delivery boundary before projecting metrics.  A bounded
        # max-steps window intentionally leaves it pending.
        execution.engine.advance()
        flush = getattr(memory, "flush_delayed_releases", None)
        if flush is not None:
            flush()
        served_end = max(served_end, float(execution.engine.now))
    metrics: list[RequestMetrics] = []
    feedback: list[dict[str, Any]] = []
    for scheduler in schedulers:
        for state in scheduler.states:
            spec = state.request
            feedback.append({
                "request_id": spec.request_id,
                "replica": scheduler.replica,
                "arrival_ms": spec.arrival_ms,
                "generated": state.generated,
                "computed": state.computed,
                "first_token_ms": state.first_token_ms,
                "finish_ms": state.finish_ms,
                "done": state.done,
                "reason": state.reason,
                "host_tokens": state.host_result.tokens,
                "host_last_ms": state.host_result.last_ms,
                "delivered_tokens": state.client_delivery.tokens,
                "delivery_last_ms": state.client_delivery.last_ms,
            })
            if state.first_token_ms is None or state.finish_ms is None:
                continue
            completed_at = (state.client_delivery.last_ms
                            if state.client_delivery.tokens
                            else state.finish_ms)
            metrics.append(RequestMetrics(
                spec.request_id,
                state.first_token_ms - spec.arrival_ms,
                ((completed_at - state.first_token_ms)
                 / max(1, spec.output_tokens - 1)),
                (state.itl_sum / state.itl_count if state.itl_count else 0.0),
                completed_at - spec.arrival_ms,
                spec.prompt_tokens, spec.output_tokens))
    def average(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    return FrameworkPrediction(
        body_duration_ms=served_end,
        ttft_ms=average([item.ttft_ms for item in metrics]),
        tpot_ms=average([item.tpot_ms for item in metrics]),
        itl_ms=average([item.itl_ms for item in metrics]),
        requests=tuple(metrics),
        steps=tuple(run.get("traces", ())),
        bottlenecks=(),
        padding={},
        sync={
            "dp_ranks": parallel.dp,
            "phase": getattr(config, "phase", "serving"),
            "iterations": run.get("iterations", 0),
            "model_steps": run.get("model_steps", 0),
            "assembly": run.get("assembly", {}),
        },
        mtp=None,
        stop_reason=str(run.get("stop_reason", "drained")),
        scheduler_feedback=tuple(feedback),
    )


def predict_framework(model: ModelSpec | ModelPipeline | Mapping[str, Any],
                      requests: Iterable[Any], config: FrameworkConfig, *, features: FeatureConfig | None = None,
                      policy: ImplementationPolicy | None = None,
                      execution: Any | None = None,
                      case: Any | None = None,
                      schedulers: Sequence[Any] | None = None,
                      memory: Any | None = None, lower: Any | None = None,
                      window: Mapping[str, Mapping[str, Any]] | None = None,
                      trace_limit: int = 2,
                      max_model_steps: int | None = None,
                      observer: Any | None = None) -> FrameworkPrediction:
    """Zero-parameter serving entry from structure, ragged requests and config.

    With ``case``/``execution``/``schedulers``/``memory`` it delegates to
    ``predict_framework_cohort`` and T14 ``run_cohort`` (the production
    frontier).  Without a case it keeps the standalone compatibility path for
    callers that only have a ModelPipeline/supply.
    """
    if case is not None:
        if execution is None:
            raise ModelingError("framework_cohort_execution_required")
        return predict_framework_cohort(
            model, requests, config, case=case, execution=execution,
            schedulers=schedulers, memory=memory, lower=lower, window=window,
            trace_limit=trace_limit, max_model_steps=max_model_steps,
            observer=observer)
    pipeline = (model if isinstance(model, ModelPipeline) else
                assemble_model(model, supply=config.supply, features=features, policy=policy or config.policy))
    dp = max(1, pipeline.features.dp)
    if execution is not None and dp != 1:
        raise ModelingError("framework_execution_requires_single_dp")
    work = [_RunRequest(item, prefilled=min(item.prefix_tokens, item.prompt_tokens - 1), replica=index % dp)
            for index, item in enumerate(_coerce_requests(requests))]
    pending = deque(sorted(work, key=lambda item: (item.spec.arrival_ms, -item.spec.priority)))
    queues = [ (deque(), deque()) for _ in range(dp) ]  # running, prefilling per replica
    replica_now = [0.0 for _ in range(dp)]
    replica_states = [StateView() for _ in range(dp)]
    step_index = 0
    idle_rounds = 0
    records, family_ms, pipe_ms = [], defaultdict(float), defaultdict(float)
    padded_rows = attention_rows = moe_rows = padding_waste = decode_cnt = all_cnt = max_act_tok = max_pad_tok = 0
    dp_skew_total = ep_skew_total = decode_body_sum = all_body_sum = 0.0
    mtp_draft = mtp_target = mtp_commit = 0.0
    mtp_steps = mtp_accepted = mtp_verified = 0
    if config.mtp_draft_len and pipeline.draft_subgraph is None:
        raise ModelingError("mtp_requires_draft_subgraph")
    buckets = config.graph_buckets or pipeline.features.graph_buckets
    def localized(rank: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
        rows = []
        for layer, counts in config.routing:
            _experts, _top_k, local = _layer_moe(pipeline.model, int(layer), pipeline.features.ep)
            rows.append((int(layer), tuple(counts[rank * local:(rank + 1) * local])))
        return tuple(rows)
    def take_pending(rank: int, limit: float) -> _RunRequest | None:
        for candidate in pending:
            if candidate.replica == rank and candidate.spec.arrival_ms <= limit + 1e-12:
                pending.remove(candidate)
                return candidate
        return None
    while pending or any(running or prefilling for running, prefilling in queues):
        if step_index >= config.max_steps:
            raise ModelingError("framework_max_steps_exceeded")
        round_runs: list[dict] = []
        round_preds: dict[int, Any] = {}
        for rank in range(dp):
            running, prefilling = queues[rank]
            start = (float(execution.engine.now)
                     if execution is not None else replica_now[rank])
            admitted = take_pending(rank, start)
            while admitted is not None:
                prefilling.append(admitted)
                admitted = take_pending(rank, start)
            items = [(run, "decode", 1) for run in running]
            slots = max(1, config.max_batch) - len(items)
            while prefilling and slots > 0:
                run = prefilling.popleft()
                remaining = run.spec.prompt_tokens - run.prefilled
                chunk = remaining if not config.max_prefill_tokens else min(remaining, config.max_prefill_tokens)
                items.append((run, "prefill", max(1, chunk)))
                slots -= 1
            if not items:
                future = [run.spec.arrival_ms for run in pending
                          if run.replica == rank]
                if future and not running and not prefilling:
                    target = max(start, min(future))
                    replica_now[rank] = target
                    if execution is not None and target > execution.engine.now:
                        execution.engine.submit(
                            f"r{rank}:idle", latency=target - execution.engine.now)
                        execution.engine.advance()
                idle_rounds += 1
                if idle_rounds > 10000:
                    raise ModelingError("framework_idle_progress_exhausted")
                continue
            idle_rounds = 0
            kinds = {kind for _, kind, _ in items}
            phase = next(iter(kinds)) if len(kinds) == 1 else "mixed"
            requests_geom = [RequestGeometry(
                chunk, run.prefilled if kind == "prefill" else run.spec.prompt_tokens + max(0, run.generated - 1),
                phase=kind, replica=rank) for run, kind, chunk in items]
            active = WorkloadGeometry(tuple(requests_geom), phase=phase)
            mtp = (config.mtp_draft_len > 0 and all(kind == "decode" for _, kind, _ in items))
            if mtp:
                declared = pipeline.draft_subgraph.operator("target_verify").theta.find("verify_tokens", 0) or 0
                verify_rows = max(1 + config.mtp_draft_len, int(declared))
                active = WorkloadGeometry(tuple(RequestGeometry(
                    verify_rows, request.context, phase="verify", replica=rank,
                    draft_len=max(1, config.mtp_draft_len)) for request in active.requests),
                    phase="verify")
            routing = localized(rank)
            if routing:
                active = WorkloadGeometry(active.requests, routing=routing, phase=active.phase_name)
            ids = tuple(run.spec.request_id for run, _, _ in items)
            emit = tuple(kind == "decode" or (kind == "prefill" and run.prefilled + chunk >= run.spec.prompt_tokens)
                         for run, kind, chunk in items)
            accepted, commits = 0, None
            if mtp:
                commits = tuple(min(config.mtp_accept + 1, run.spec.output_tokens - run.generated)
                                for run, _, _ in items)
                accepted = sum(min(config.mtp_accept, max(0, commit - 1)) for commit in commits)
            padded = _padding(active, buckets)
            prediction = pipeline.forward(
                active, replica_states[rank], padded_geometry=padded,
                request_ids=ids, emit_mask=emit, commit_tokens=commits,
                acceptance=((accepted, len(items) * config.mtp_draft_len) if mtp else None),
                execution=execution, replica=rank, stage=0)
            if execution is not None:
                execution.engine.advance()
            round_preds[rank] = prediction
            round_runs.append(dict(rank=rank, start=start, items=items, active=active, padded=padded,
                                   phase=phase, mtp=mtp, prediction=prediction, commits=commits))
        if not round_runs:
            continue
        prediction = max(round_preds.values(), key=lambda item: item.total_ms)
        dp_skew = (0.0 if execution is not None else
                   (max(item.total_ms for item in round_preds.values())
                    - min(item.total_ms for item in round_preds.values())))
        reference_ms = (pipeline.forward(prediction.active, StateView()).total_ms
                        if prediction.padded is not None else prediction.total_ms)
        ep_delta, ep_skew = _ep_skew(pipeline, prediction.active, config)
        ep_delta, ep_skew = _frac(ep_delta), _frac(ep_skew)
        draft_ms, commit_ms = (_draft_cost(pipeline, WorkloadGeometry(
                tuple(RequestGeometry(1, request.context, phase="draft", replica=request.replica,
                                      draft_len=max(1, config.mtp_draft_len))
                      for request in prediction.active.requests), phase="draft"),
            replica_states[0], proposals=config.mtp_draft_len,
            execution=execution)
            if any(item["mtp"] for item in round_runs) else (Fraction(0), Fraction(0)))
        overhead = max(Fraction(0), ep_delta) + draft_ms + commit_ms + _frac(config.graph_replay_ms)
        if execution is not None:
            body_local = {}
            starts = {entry["rank"]: entry["start"] for entry in round_runs}
            for rank, item in round_preds.items():
                event = item.execution_final_event
                completed_at = (Fraction(str(event.time))
                                if event is not None and event.time is not None
                                else _frac(execution.engine.now))
                body_local[rank] = max(
                    Fraction(0), completed_at - _frac(starts[rank])) + overhead
        else:
            body_local = {rank: item.total_ms + overhead for rank, item in round_preds.items()}
        body = max(body_local.values())
        for row in prediction.rows:
            family_ms[row.cost.family] += _ms(row.cost.duration_ms)
            for pipe, value in row.cost.times:
                pipe_ms[pipe] += _ms(value)
        if prediction.padded:
            padded_rows += sum(row.padded_tokens for row in prediction.rows if row.geometry == "padded")
            padding_waste += max(0.0, _ms(prediction.total_ms - reference_ms))
        attention_rows += sum(row.active_tokens for row in prediction.rows if row.cost.family in _ATTENTION)
        moe_rows += sum(int(row.cost.get("moe_rows", 0)) for row in prediction.rows if row.cost.family == "grouped_gemm")
        completed, accepted_total, mtp_verified_round = [], 0, 0
        active_tokens = padded_tokens = 0
        ids_all: list[str] = []
        for entry in round_runs:
            rank, start, items = entry["rank"], entry["start"], entry["items"]
            running, _ = queues[rank]
            local_end = start + _ms(body_local[rank])
            active_tokens += entry["active"].tokens
            padded_tokens += entry["padded"].effective_tokens if entry["padded"] else entry["active"].tokens
            ids_all.extend(run.spec.request_id for run, _, _ in items)
            mtp_verified_round += len(items) * config.mtp_draft_len if entry["mtp"] else 0
            for index, (run, kind, chunk) in enumerate(items):
                if kind == "prefill":
                    run.prefilled += chunk
                    if run.prefilled >= run.spec.prompt_tokens:
                        run.first_token_at = run.last_token_at = local_end
                        run.generated = 1
                        if run.spec.output_tokens <= 1:
                            run.completion_at = local_end
                            completed.append(run)
                        else:
                            running.append(run)
                    else:
                        queues[rank][1].appendleft(run)
                else:
                    committed = (entry["commits"][index] if entry["mtp"] else 1)
                    accepted = committed - 1 if entry["mtp"] else 0
                    if run.last_token_at is not None:
                        run.itl_sum += max(0.0, local_end - run.last_token_at)
                        run.itl_count += max(1, committed)
                    accepted_total += accepted
                    run.generated += committed
                    run.last_token_at = local_end
                    if run.generated >= run.spec.output_tokens:
                        run.completion_at = local_end
                        completed.append(run)
            replica_now[rank] = local_end
        for run in completed:
            if run in queues[run.replica][0]:
                queues[run.replica][0].remove(run)
        for rank, item in round_preds.items():
            state = item.state_out
            for run in completed:
                if run.replica == rank:
                    state = state.retire_compression(run.spec.request_id)
            replica_states[rank] = state
        if any(item["mtp"] for item in round_runs):
            mtp_steps += 1
            mtp_accepted += accepted_total
            mtp_verified += mtp_verified_round
            mtp_draft += _ms(draft_ms)
            mtp_target += _ms(prediction.total_ms)
            mtp_commit += _ms(commit_ms)
        dp_skew_total += _ms(dp_skew)
        ep_skew_total += _ms(ep_skew)
        phase = next(iter({item["phase"] for item in round_runs})) if len(
            {item["phase"] for item in round_runs}) == 1 else "mixed"
        body_val = _ms(body)
        all_body_sum += body_val; all_cnt += 1; max_act_tok = max(max_act_tok, active_tokens); max_pad_tok = max(max_pad_tok, padded_tokens)
        if phase == "decode": decode_body_sum += body_val; decode_cnt += 1
        if len(records) < max(1, int(config.max_retained_steps)):
            records.append({"index": int(step_index), "kind": phase, "batch": sum(len(item["items"]) for item in round_runs),
                            "active_tokens": active_tokens, "padded_tokens": padded_tokens,
                            "body_ms": body_val, "sync_wait_ms": _ms(dp_skew + ep_skew),
                            "request_ids": tuple(ids_all)})
        step_index += 1
    metrics = []
    for run in work:
        if run.first_token_at is None or run.completion_at is None:
            raise ModelingError(f"request_not_completed:{run.spec.request_id}")
        metrics.append(RequestMetrics( run.spec.request_id, run.first_token_at - run.spec.arrival_ms,
            (run.completion_at - run.first_token_at) / max(1, run.spec.output_tokens - 1),
            run.itl_sum / run.itl_count if run.itl_count else 0.0, run.completion_at - run.spec.arrival_ms,
            run.spec.prompt_tokens, run.spec.output_tokens))
    body = (decode_body_sum / decode_cnt if decode_cnt else all_body_sum / max(1, all_cnt))
    bottlenecks = sorted({**{f"family:{key}": value for key, value in family_ms.items()},
                          **{f"pipe:{key}": value for key, value in pipe_ms.items()}}.items(),
                         key=lambda item: (-item[1], item[0]))
    padding = {"bucket": max_pad_tok, "active_tokens": max_act_tok, "padded_tokens": max_pad_tok,
               "operator_padded_rows": padded_rows, "attention_active_rows": attention_rows,
               "moe_active_rows": moe_rows, "padding_waste_ms": padding_waste}
    sync = {"dp_ranks": dp, "ep_ranks": pipeline.features.ep,
            "dp_skew_ms": dp_skew_total, "ep_skew_ms": ep_skew_total, "sync_wait_ms": dp_skew_total + ep_skew_total,
            "source": "dp_request_skew+ep_routing_skew"}
    mtp_report = ({"draft_len": config.mtp_draft_len, "accepted": mtp_accepted,
                   "verified": mtp_verified, "draft_ms": mtp_draft, "target_ms": mtp_target,
                   "commit_ms": mtp_commit, "steps": mtp_steps,
                   "total_ms": mtp_draft + mtp_target + mtp_commit,
                   "acceptance_factor": mtp_accepted / max(1, mtp_verified)}
                  if config.mtp_draft_len else None)
    return FrameworkPrediction( body, sum(item.ttft_ms for item in metrics) / len(metrics),
        sum(item.tpot_ms for item in metrics) / len(metrics),
        sum(item.itl_ms for item in metrics) / len(metrics), tuple(metrics),
        tuple(records), tuple(bottlenecks[:12]), padding, sync, mtp_report)


# Backward-compatible private name used by tests and profiler callers.
_pcp_rank_geometry = pcp_rank_geometry
