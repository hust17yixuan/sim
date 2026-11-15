"""Typed canonical subgraph vocabulary and port contracts.

[Role]
  Define the finite, typed semantic subgraph vocabulary used by the
  three-level piecewise compiler.  A ``SubgraphTemplate`` is a reusable
  semantic shape; a ``SubgraphInstance`` binds it to one layer/stage and one
  fully static parameter record (``StaticSignature``).  ``OperatorTemplate``
  describes a physical work producer inside a subgraph, and ``StatePortSpec``
  carries ownership/readiness instead of an untyped owner dictionary.
  ``TensorPortSpec``/``EventPortSpec``/``PortSet`` are the common typed ports
  used by program fragments from model, framework and native adapters.

[Invariants]
  * Template and instance records are immutable; all mappings become tuples.
  * A template key contains semantics, not model names, layer ids, request ids
    or measurement availability.  Layer relocalisation is a port operation.
  * ``StaticSignature`` fixes weight dimensions, head counts, expert geometry,
    quantisation roles, parallel axis sizes and declared tiles.  Sequence
    geometry remains dynamic and lives outside this module.
  * Every operator name is unique inside one template; every fusion member is
    an operator of that template.  Unknown kinds/variants fail loudly.
  * A consumer state port must declare a producer and a readiness contract;
    this module validates records, never device work.
  * Port names are unique inside one ``PortSet`` so a binding has one target.
    Direction, scope and readiness are explicit; a tensor port never infers a
    dtype or layout from another value.

[Timeouts & Locks]
  Pure immutable record construction and bounded validation.  No waits, I/O,
  locks, device probes or weight loading.

[Side-Effect Policy]
  No filesystem/network/global mutation; only new frozen records are returned.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .common import ModelingError

#: The 29 canonical LLM subgraph families under their paper / vLLM names.
#: Unknown mechanisms must remain an explicit coverage hole rather than being
#: folded into one of these entries.  Attention is split by the mechanism the
#: kernel actually executes (MHA/GQA/MQA versus latent MLA/DSA versus
#: window-compressed SWA/CSA/CSA2/HCA versus linear GDN/KDA), not by a numeric
#: family code.
SUBGRAPH_KINDS: tuple[str, ...] = (
    "vocab_parallel_embedding",
    "layernorm",
    "kv_cache",
    "kv_compressor",
    "indexer_prolog",
    "lightning_indexer",
    "mlp",
    "moe_router",
    "moe_token_dispatcher",
    "routed_experts",
    "shared_experts",
    "hyper_connection",
    "attn_residual",
    "per_layer_embedding",
    "lm_head_sampler",
    "spec_decode",
    "mm_encoder_projector",
    "mha_attention",
    "gqa_attention",
    "mqa_attention",
    "attention",
    "mla_attention",
    "dsa_attention",
    "swa_attention",
    "csa_attention",
    "csa2_attention",
    "hca_attention",
    "gdn_linear_attn",
    "kda_linear_attn",
)

SUBGRAPH_KIND_SET = frozenset(SUBGRAPH_KINDS)

#: Standard (softmax) attention kinds; the ``attention`` member is the
#: shape-free fallback used when a plan declares no head geometry.
STANDARD_ATTENTION_KINDS: frozenset[str] = frozenset({
    "mha_attention", "gqa_attention", "mqa_attention", "attention",
})

#: Latent attention kinds: uncompressed MLA and sparse (DSA) latent MLA.
LATENT_ATTENTION_KINDS: frozenset[str] = frozenset({
    "mla_attention", "dsa_attention",
})

#: Window and compressed-sparse attention kinds (SWA plus the compressed
#: branches that merge a window and a compressed-KV result through LSE).
WINDOW_SPARSE_ATTENTION_KINDS: frozenset[str] = frozenset({
    "swa_attention", "csa_attention", "csa2_attention", "hca_attention",
})

#: Compressed branches only (SWA is a window branch without compression).
COMPRESSED_ATTENTION_KINDS: frozenset[str] = frozenset({
    "csa_attention", "csa2_attention", "hca_attention",
})

#: Linear-attention kinds with a recurrent state instead of a KV cache.
LINEAR_ATTENTION_KINDS: frozenset[str] = frozenset({
    "gdn_linear_attn", "kda_linear_attn",
})

#: Kinds that may execute the standard softmax attention macro (including the
#: window-only SWA schedule, which shares that macro and its qkv_norm_rope
#: fusion).  The window/compressed macro is *not* part of this set even when
#: it emits ``swa_attention`` for a compression-free plan.
SOFTMAX_ATTENTION_KINDS: frozenset[str] = STANDARD_ATTENTION_KINDS | frozenset({
    "swa_attention",
})

#: Every kind that executes an attention (or linear-attention) core.
ALL_ATTENTION_KINDS: frozenset[str] = (
    STANDARD_ATTENTION_KINDS | LATENT_ATTENTION_KINDS
    | WINDOW_SPARSE_ATTENTION_KINDS | LINEAR_ATTENTION_KINDS
)

#: Kinds that read a KV cache; a draft/verify stage needs one of these.
KV_READING_ATTENTION_KINDS: frozenset[str] = frozenset({
    "mha_attention", "gqa_attention", "mqa_attention", "attention",
    "mla_attention", "dsa_attention",
    "swa_attention", "csa_attention", "csa2_attention", "hca_attention",
})

#: MoE kinds that form the routed/shared expert execution chain.
MOE_KINDS: frozenset[str] = frozenset({
    "moe_router", "moe_token_dispatcher", "routed_experts", "shared_experts",
})

_PORT_DIRECTIONS = frozenset({"in", "out", "inout"})
_PORT_SCOPES = frozenset({"operator", "layer", "stage", "request", "cohort"})
_PORT_READINESS = frozenset({
    "producer_complete", "same_invocation", "previous_iteration", "external",
})

#: Event ports name the causality a fragment waits for or publishes.
EVENT_PORT_KINDS = frozenset({
    "ready", "complete", "commit", "release", "grant", "record", "wait",
})

#: Fusion membership may never cross these boundaries.
_FUSION_BOUNDARIES = frozenset({"stream", "stage", "phase", "capture"})


def _freeze_value(item: Any, reason: str) -> Any:
    if item is None or isinstance(item, (bool, int, float, str)):
        return item
    if isinstance(item, Mapping):
        raise ModelingError(f"{reason}:mapping_value_invalid")
    if isinstance(item, (tuple, list)):
        return tuple(_freeze_value(sub, reason) for sub in item)
    raise ModelingError(f"{reason}:mutable_value_invalid:{type(item).__name__}")


def _tuple_of_pairs(value: Any, reason: str) -> tuple[tuple[str, Any], ...]:
    """Normalise a mapping or iterable of pairs into sorted immutable pairs."""
    if not value:
        return ()
    if isinstance(value, dict) or isinstance(value, Mapping):
        value = tuple(value.items())
    try:
        pairs = tuple((str(key), _freeze_value(item, reason)) for key, item in value)
    except ModelingError:
        raise
    except (TypeError, ValueError) as exc:
        raise ModelingError(reason) from exc
    seen: set[str] = set()
    for key, _ in pairs:
        if not key or key in seen:
            raise ModelingError(f"{reason}:duplicate_or_empty:{key}")
        seen.add(key)
    return tuple(sorted(pairs))


def _typed_port_tuple(value: Any, cls: type, reason: str) -> tuple:
    if value is None:
        return ()
    if not isinstance(value, (tuple, list)):
        raise ModelingError(reason)
    result = tuple(value)
    if not all(isinstance(port, cls) for port in result):
        raise ModelingError(reason)
    names = [port.name for port in result]
    if len(set(names)) != len(names):
        raise ModelingError(f"{reason}:duplicate_port")
    return result


def _port_tuple(value: Any, reason: str) -> tuple["StatePortSpec", ...]:
    return _typed_port_tuple(value, StatePortSpec, reason)


@dataclass(frozen=True)
class StatePortSpec:
    """One typed state/production port of a subgraph.

    ``owner`` is ``self`` or ``layer:<n>``/``stage:<n>``; ``readiness`` states
    which completion edge a consumer must observe.  A port describes a
    dependency and a representation, not device traffic by itself.
    """

    name: str
    direction: str
    representation: str
    scope: str = "layer"
    owner: str = "self"
    readiness: str = "producer_complete"
    unit: str = ""
    persistent: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("state_port_name_invalid")
        if self.direction not in _PORT_DIRECTIONS:
            raise ModelingError(f"state_port_direction_invalid:{self.direction}")
        if self.scope not in _PORT_SCOPES:
            raise ModelingError(f"state_port_scope_invalid:{self.scope}")
        if self.readiness not in _PORT_READINESS:
            raise ModelingError(f"state_port_readiness_invalid:{self.readiness}")
        if not isinstance(self.representation, str) or not self.representation.strip():
            raise ModelingError(f"state_port_representation_invalid:{self.name}")
        if not isinstance(self.owner, str):
            raise ModelingError(f"state_port_owner_invalid:{self.name}")
        if not isinstance(self.unit, str):
            raise ModelingError(f"state_port_unit_invalid:{self.name}")

    @property
    def is_producer(self) -> bool:
        return self.direction in {"out", "inout"}

    @property
    def is_consumer(self) -> bool:
        return self.direction in {"in", "inout"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "direction": self.direction,
            "representation": self.representation, "scope": self.scope,
            "owner": self.owner, "readiness": self.readiness,
            "unit": self.unit, "persistent": self.persistent,
        }


@dataclass(frozen=True)
class TensorPortSpec:
    """A typed tensor input/output of a fragment or subgraph.

    ``dtype``/``layout_kind`` describe this value's own representation; they
    are never inherited from a model-level activation dtype.  ``optional``
    marks a port that may be absent, not a port whose value may be guessed.
    """

    name: str
    direction: str
    representation: str
    dtype: str = ""
    layout_kind: str = "strided"
    scope: str = "operator"
    owner: str = "self"
    readiness: str = "producer_complete"
    optional: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("tensor_port_name_invalid")
        if self.direction not in _PORT_DIRECTIONS:
            raise ModelingError(f"tensor_port_direction_invalid:{self.direction}")
        if self.scope not in _PORT_SCOPES:
            raise ModelingError(f"tensor_port_scope_invalid:{self.scope}")
        if self.readiness not in _PORT_READINESS:
            raise ModelingError(f"tensor_port_readiness_invalid:{self.name}")
        if not isinstance(self.representation, str) or not self.representation.strip():
            raise ModelingError(f"tensor_port_representation_invalid:{self.name}")
        for field in ("dtype", "layout_kind", "owner"):
            if not isinstance(getattr(self, field), str):
                raise ModelingError(f"tensor_port_{field}_invalid:{self.name}")
        if not self.layout_kind:
            raise ModelingError(f"tensor_port_layout_kind_invalid:{self.name}")
        if not isinstance(self.optional, bool):
            raise ModelingError(f"tensor_port_optional_invalid:{self.name}")

    @property
    def is_producer(self) -> bool:
        return self.direction in {"out", "inout"}

    @property
    def is_consumer(self) -> bool:
        return self.direction in {"in", "inout"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "direction": self.direction,
            "representation": self.representation, "dtype": self.dtype,
            "layout_kind": self.layout_kind, "scope": self.scope,
            "owner": self.owner, "readiness": self.readiness,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class EventPortSpec:
    """A typed event dependency or publication.

    ``generation`` identifies the recorded event instance; a consumer waits on
    that record, not on an arbitrary same-named historical event.
    """

    name: str
    direction: str
    event: str
    scope: str = "stage"
    owner: str = "self"
    readiness: str = "producer_complete"
    generation: int | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("event_port_name_invalid")
        if self.direction not in _PORT_DIRECTIONS:
            raise ModelingError(f"event_port_direction_invalid:{self.direction}")
        if self.event not in EVENT_PORT_KINDS:
            raise ModelingError(f"event_port_kind_unknown:{self.event}")
        if self.scope not in _PORT_SCOPES:
            raise ModelingError(f"event_port_scope_invalid:{self.scope}")
        if self.readiness not in _PORT_READINESS:
            raise ModelingError(f"event_port_readiness_invalid:{self.name}")
        if not isinstance(self.owner, str):
            raise ModelingError(f"event_port_owner_invalid:{self.name}")
        if self.generation is not None and (
                not isinstance(self.generation, int)
                or isinstance(self.generation, bool)
                or self.generation < 0):
            raise ModelingError(f"event_port_generation_invalid:{self.name}")
        if not isinstance(self.optional, bool):
            raise ModelingError(f"event_port_optional_invalid:{self.name}")

    @property
    def is_producer(self) -> bool:
        return self.direction in {"out", "inout"}

    @property
    def is_consumer(self) -> bool:
        return self.direction in {"in", "inout"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "direction": self.direction,
            "event": self.event, "scope": self.scope, "owner": self.owner,
            "readiness": self.readiness, "generation": self.generation,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class PortSet:
    """The common tensor/state/event ports of one fragment.

    Names are unique across all six groups.  Each group is normalised to a
    tuple of the declared port type; a plain string or an untyped mapping is
    rejected instead of being guessed.
    """

    tensor_in: tuple[TensorPortSpec, ...] = ()
    tensor_out: tuple[TensorPortSpec, ...] = ()
    state_in: tuple[StatePortSpec, ...] = ()
    state_out: tuple[StatePortSpec, ...] = ()
    event_in: tuple[EventPortSpec, ...] = ()
    event_out: tuple[EventPortSpec, ...] = ()

    def __post_init__(self) -> None:
        groups = (
            ("tensor_in", TensorPortSpec), ("tensor_out", TensorPortSpec),
            ("state_in", StatePortSpec), ("state_out", StatePortSpec),
            ("event_in", EventPortSpec), ("event_out", EventPortSpec),
        )
        names: dict[str, str] = {}
        for group, cls in groups:
            ports = _typed_port_tuple(getattr(self, group), cls,
                                      f"port_set_{group}_invalid")
            object.__setattr__(self, group, ports)
            for port in ports:
                if port.name in names:
                    raise ModelingError(
                        f"port_set_name_duplicate:{port.name}:"
                        f"{names[port.name]}:{group}")
                names[port.name] = group

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(port.name for group in (
            self.tensor_in, self.tensor_out, self.state_in, self.state_out,
            self.event_in, self.event_out) for port in group)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "tensor_in": [port.to_dict() for port in self.tensor_in],
            "tensor_out": [port.to_dict() for port in self.tensor_out],
            "state_in": [port.to_dict() for port in self.state_in],
            "state_out": [port.to_dict() for port in self.state_out],
            "event_in": [port.to_dict() for port in self.event_in],
            "event_out": [port.to_dict() for port in self.event_out],
        }


@dataclass(frozen=True)
class StaticSignature:
    """Frozen ``Theta`` record for one operator: all shape-free parameters.

    Groups are intentionally explicit rather than one untyped dictionary so a
    family model reads the same fields that instantiation fixed.  Values must
    be immutable scalars (``int``/``float``/``str``/``bool``) or tuples of
    those; a mutable value would make the static contract unsound.
    """

    role: str
    matrix: tuple[tuple[str, Any], ...] = ()
    heads: tuple[tuple[str, Any], ...] = ()
    experts: tuple[tuple[str, Any], ...] = ()
    quant: tuple[tuple[str, Any], ...] = ()
    parallel: tuple[tuple[str, Any], ...] = ()
    tiles: tuple[tuple[str, Any], ...] = ()
    extras: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ModelingError("static_signature_role_invalid")
        for group in ("matrix", "heads", "experts", "quant", "parallel",
                      "tiles", "extras"):
            object.__setattr__(
                self, group, _tuple_of_pairs(getattr(self, group),
                                             f"static_signature_{group}_invalid"))

    def group(self, name: str) -> dict[str, Any]:
        if name not in {"matrix", "heads", "experts", "quant", "parallel",
                        "tiles", "extras"}:
            raise ModelingError(f"static_signature_group_unknown:{name}")
        return dict(getattr(self, name))

    def get(self, group: str, name: str, default: Any = None) -> Any:
        return self.group(group).get(name, default)

    def find(self, name: str, default: Any = None) -> Any:
        """Search all groups in declaration order (first hit wins)."""
        flat = getattr(self, "_flat_cache", None)
        if flat is None:
            flat = {}
            for group in ("extras", "tiles", "parallel", "quant", "experts", "heads", "matrix"):
                flat.update(getattr(self, group))
            object.__setattr__(self, "_flat_cache", flat)
        return flat.get(name, default)

    def require(self, name: str) -> Any:
        marker = object()
        value = self.find(name, marker)
        if value is marker:
            raise ModelingError(f"static_parameter_missing:{self.role}:{name}")
        return value

    def key(self) -> tuple:
        return (
            self.role,
            (
                ("matrix", self.matrix), ("heads", self.heads),
                ("experts", self.experts), ("quant", self.quant),
                ("parallel", self.parallel), ("tiles", self.tiles),
                ("extras", self.extras),
            ),
        )

    def fingerprint(self) -> str:
        payload = json.dumps(_jsonable(self.key()), sort_keys=True,
                             separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            **{group: dict(getattr(self, group)) for group in (
                "matrix", "heads", "experts", "quant", "parallel", "tiles",
                "extras")},
        }

    def relocated(self, **updates: Any) -> "StaticSignature":
        """Return a copy with extras replaced/added; layer ids stay ports."""
        extras = dict(self.extras)
        extras.update(updates)
        return StaticSignature(
            self.role, matrix=self.matrix, heads=self.heads,
            experts=self.experts, quant=self.quant, parallel=self.parallel,
            tiles=self.tiles, extras=_tuple_of_pairs(extras, "static_extras"),
        )


@dataclass(frozen=True)
class OperatorTemplate:
    """A semantic work producer before static parameters are bound."""

    name: str
    family: str
    role: str
    stream: str = "main"
    fusion_group: str = ""
    inputs: tuple[StatePortSpec, ...] = ()
    outputs: tuple[StatePortSpec, ...] = ()
    coverage: str = "structural"
    static_slots: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("operator_template_name_invalid")
        if not isinstance(self.family, str) or not self.family.strip():
            raise ModelingError(f"operator_template_family_invalid:{self.name}")
        if not isinstance(self.role, str) or not self.role.strip():
            raise ModelingError(f"operator_template_role_invalid:{self.name}")
        if not isinstance(self.stream, str) or not self.stream.strip():
            raise ModelingError(f"operator_template_stream_invalid:{self.name}")
        object.__setattr__(self, "inputs",
                           _port_tuple(self.inputs, f"operator_inputs_invalid:{self.name}"))
        object.__setattr__(self, "outputs",
                           _port_tuple(self.outputs, f"operator_outputs_invalid:{self.name}"))
        object.__setattr__(self, "static_slots", tuple(self.static_slots))
        object.__setattr__(self, "notes", tuple(self.notes))

    def key(self) -> tuple:
        return (
            self.name, self.family, self.role, self.stream, self.fusion_group,
            tuple(tuple(port.to_dict().items()) for port in self.inputs),
            tuple(tuple(port.to_dict().items()) for port in self.outputs),
            self.coverage, self.static_slots, self.notes,
        )


@dataclass(frozen=True)
class BoundOperator:
    """An operator template with all static parameters fixed."""

    template: OperatorTemplate
    theta: StaticSignature

    def __post_init__(self) -> None:
        if not isinstance(self.template, OperatorTemplate):
            raise ModelingError("bound_operator_template_invalid")
        if not isinstance(self.theta, StaticSignature):
            raise ModelingError(f"bound_operator_theta_invalid:{self.template.name}")
        for slot in self.template.static_slots:
            self.theta.require(slot)

    @property
    def name(self) -> str:
        return self.template.name

    @property
    def family(self) -> str:
        return self.template.family

    @property
    def role(self) -> str:
        return self.template.role

    @property
    def stream(self) -> str:
        return self.template.stream

    def key(self) -> tuple:
        return self.template.key(), self.theta.key()


@dataclass(frozen=True)
class FusionContract:
    """Declared native-fusion lowering over semantic operators.

    ``removes`` names internal materialisations that disappear from physical
    traffic.  ``keeps`` names externally consumed quantities (residual, scale,
    rstd, KV, routing ledger) that must survive the lowering.  A fusion may
    not cross a stream/stage/phase/capture boundary.
    """

    name: str
    members: tuple[str, ...]
    removes: tuple[str, ...] = ()
    keeps: tuple[str, ...] = ()
    startup: int = 1
    stream: str = "main"
    contract: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("fusion_contract_name_invalid")
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "removes", tuple(self.removes))
        object.__setattr__(self, "keeps", tuple(self.keeps))
        if not self.members or len(set(self.members)) != len(self.members):
            raise ModelingError(f"fusion_members_invalid:{self.name}")
        if set(self.removes) & set(self.keeps):
            raise ModelingError(f"fusion_remove_keeps_conflict:{self.name}")
        if self.startup < 1:
            raise ModelingError(f"fusion_startup_invalid:{self.name}")
        if isinstance(self.contract, Mapping):
            object.__setattr__(self, "contract",
                               tuple(sorted(self.contract.items())))
        else:
            object.__setattr__(self, "contract", tuple(self.contract))
        for key, _ in self.contract:
            if key not in _FUSION_BOUNDARIES:
                raise ModelingError(f"fusion_contract_boundary_unknown:{key}")


@dataclass(frozen=True)
class SubgraphTemplate:
    """A reusable canonical semantic subgraph; no layer or request identity.

    ``title`` is a paper-aligned English presentation label only.  It is deliberately
    excluded from :meth:`key` so renaming a display string cannot change the
    template, instance or performance identity.
    """

    kind: str
    variant: str
    title: str
    operators: tuple[OperatorTemplate, ...]
    state_in: tuple[StatePortSpec, ...] = ()
    state_out: tuple[StatePortSpec, ...] = ()
    children: tuple[str, ...] = ()
    fusions: tuple[FusionContract, ...] = ()
    coverage: str = "structural"
    contract: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in SUBGRAPH_KIND_SET:
            raise ModelingError(f"subgraph_kind_unknown:{self.kind}")
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ModelingError(f"subgraph_variant_invalid:{self.kind}")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ModelingError(f"subgraph_title_invalid:{self.kind}")
        operators = tuple(self.operators)
        if not operators:
            raise ModelingError(f"subgraph_operators_empty:{self.kind}")
        if not all(isinstance(op, OperatorTemplate) for op in operators):
            raise ModelingError(f"subgraph_operator_invalid:{self.kind}")
        names = [op.name for op in operators]
        if len(names) != len(set(names)):
            raise ModelingError(f"subgraph_operator_duplicate:{self.kind}")
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "state_in",
                           _port_tuple(self.state_in, f"subgraph_state_in_invalid:{self.kind}"))
        object.__setattr__(self, "state_out",
                           _port_tuple(self.state_out, f"subgraph_state_out_invalid:{self.kind}"))
        object.__setattr__(self, "children", tuple(self.children))
        for child in self.children:
            if child not in SUBGRAPH_KIND_SET:
                raise ModelingError(f"subgraph_child_unknown:{self.kind}:{child}")
        fusions = tuple(self.fusions)
        for fusion in fusions:
            if not isinstance(fusion, FusionContract):
                raise ModelingError(f"subgraph_fusion_invalid:{self.kind}")
            missing = set(fusion.members) - set(names)
            if missing:
                raise ModelingError(
                    f"subgraph_fusion_member_unknown:{self.kind}:{sorted(missing)}")
        object.__setattr__(self, "fusions", fusions)
        if isinstance(self.contract, Mapping):
            object.__setattr__(self, "contract",
                               tuple(sorted(self.contract.items())))
        else:
            object.__setattr__(self, "contract", tuple(self.contract))

    @property
    def operator_names(self) -> tuple[str, ...]:
        return tuple(op.name for op in self.operators)

    def operator(self, name: str) -> OperatorTemplate:
        for op in self.operators:
            if op.name == name:
                return op
        raise ModelingError(f"subgraph_operator_missing:{self.kind}:{name}")

    def key(self) -> tuple:
        cached = getattr(self, "_key_cache", None)
        if cached is not None:
            return cached
        # Presentation text (``title``) stays out of structural identity.
        val = (
            self.kind, self.variant,
            tuple(op.key() for op in self.operators),
            tuple(tuple(port.to_dict().items()) for port in self.state_in),
            tuple(tuple(port.to_dict().items()) for port in self.state_out),
            self.children,
            tuple((f.name, f.members, f.removes, f.keeps, f.startup, f.stream,
                   f.contract) for f in self.fusions),
            self.coverage, self.contract,
        )
        object.__setattr__(self, "_key_cache", val)
        return val

    def fingerprint(self) -> str:
        cached = getattr(self, "_fp_cache", None)
        if cached is not None:
            return cached
        payload = json.dumps(_jsonable(self.key()), sort_keys=True,
                             separators=(",", ":"), default=str)
        val = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
        object.__setattr__(self, "_fp_cache", val)
        return val


@dataclass(frozen=True)
class SubgraphInstance:
    """One layer/stage binding of a template plus fully static ``Theta``."""

    instance_id: str
    template: SubgraphTemplate
    layer: int
    stage: int
    operators: tuple[BoundOperator, ...]
    state_in: tuple[StatePortSpec, ...] = ()
    state_out: tuple[StatePortSpec, ...] = ()
    owner_layer: int | None = None
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.template, SubgraphTemplate):
            raise ModelingError("subgraph_instance_template_invalid")
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ModelingError("subgraph_instance_id_invalid")
        if not isinstance(self.layer, int) or self.layer < 0:
            raise ModelingError(f"subgraph_instance_layer_invalid:{self.instance_id}")
        if not isinstance(self.stage, int) or self.stage < 0:
            raise ModelingError(f"subgraph_instance_stage_invalid:{self.instance_id}")
        operators = tuple(self.operators)
        if not all(isinstance(op, BoundOperator) for op in operators):
            raise ModelingError(f"subgraph_instance_operator_invalid:{self.instance_id}")
        if tuple(op.name for op in operators) != self.template.operator_names:
            raise ModelingError(
                f"subgraph_instance_operator_order:{self.instance_id}")
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "state_in",
                           _port_tuple(self.state_in,
                                       f"subgraph_instance_state_in_invalid:{self.instance_id}"))
        object.__setattr__(self, "state_out",
                           _port_tuple(self.state_out,
                                       f"subgraph_instance_state_out_invalid:{self.instance_id}"))
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata",
                               tuple(sorted(self.metadata.items())))
        else:
            object.__setattr__(self, "metadata", tuple(self.metadata))

    @property
    def kind(self) -> str:
        return self.template.kind

    @property
    def variant(self) -> str:
        return self.template.variant

    def operator(self, name: str) -> BoundOperator:
        for op in self.operators:
            if op.name == name:
                return op
        raise ModelingError(f"subgraph_instance_operator_missing:{name}")

    def performance_key(self) -> tuple:
        """Cache identity for structural pricing.

        Layer identity, instance id and request identity are deliberately
        absent: relocating a repeated layer must not create a second
        performance entry.  Owner/consumer role remains part of the key.
        """
        return (
            self.template.key(),
            tuple(op.key() for op in self.operators),
            self.owner_layer is not None,
            tuple(self.metadata),
        )

    def fingerprint(self) -> str:
        cached = getattr(self, "_fp_cache", None)
        if cached is not None:
            return cached
        payload = json.dumps(_jsonable(self.performance_key()), sort_keys=True,
                             separators=(",", ":"), default=str)
        val = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
        object.__setattr__(self, "_fp_cache", val)
        return val


@dataclass(frozen=True)
class SubgraphKind:
    """Registry entry for one canonical subgraph family.

    ``title`` carries the paper-aligned English display name; it is vocabulary
    presentation metadata and never participates in structural keying.
    """

    kind: str
    title: str
    variants: tuple[str, ...]
    semantic_operators: tuple[str, ...]
    state_inputs: tuple[str, ...] = ()
    state_outputs: tuple[str, ...] = ()
    coverage: str = "structural"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in SUBGRAPH_KIND_SET:
            raise ModelingError(f"subgraph_kind_unknown:{self.kind}")
        if not self.variants:
            raise ModelingError(f"subgraph_kind_variants_empty:{self.kind}")
        if not self.semantic_operators:
            raise ModelingError(f"subgraph_kind_operators_empty:{self.kind}")


@dataclass(frozen=True)
class CoverageHole:
    """A declared unsupported or partial semantic contract."""

    kind: str
    reason: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in SUBGRAPH_KIND_SET:
            raise ModelingError(f"coverage_hole_kind_unknown:{self.kind}")
        if not self.reason:
            raise ModelingError(f"coverage_hole_reason_empty:{self.kind}")


@dataclass(frozen=True)
class CoverageReport:
    entries: tuple[tuple[str, str], ...] = ()
    holes: tuple[CoverageHole, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(sorted(self.entries)))
        object.__setattr__(self, "holes", tuple(self.holes))

    @property
    def complete(self) -> bool:
        return not self.holes

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": dict(self.entries),
            "holes": [hole.__dict__ for hole in self.holes],
            "complete": self.complete,
        }


def _jsonable(value: Any) -> Any:
    """Stable JSON view that keeps the value's type distinguishable."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict) or isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def validate_kind(kind: str) -> str:
    if kind not in SUBGRAPH_KIND_SET:
        raise ModelingError(f"subgraph_kind_unknown:{kind}")
    return kind
