"""Source-compatible arch22 dense MLA and its geometry-derived host partition.

[Role] Bind q1 BF16 latent MLA and the existing caller head materializations.
[Invariants] The caller ABI is shared with arch35; physical execution is not.
  SplitCore is geometry-derived; an explicit host declaration is an override.
  Retained KV rows are the caller's explicit arch22 tile declaration, not a
  bucket dimension and never active query work.
[Timeouts & Locks] O(batch + host owners), no locks or I/O.
[Side-Effect Policy] Appends immutable ordinary graph operators only.
"""
from mindie_sim.schema.common import ModelingError
from . import family_paged_mla, paged_mla_arch22
from .mechanism import PlannedRecipe

IMPLEMENTATION = paged_mla_arch22.IMPLEMENTATION
BRANCH = 'arch22_bf16_noquant_mla'
PROGRAM_RECIPE = 'mindie_sim.prediction.paged_mla_arch22.build'


def recipes():
    return {IMPLEMENTATION: family_paged_mla.planned_recipe(
        paged_mla_arch22.build)}


def _common_kwargs(kwargs):
    options = dict(kwargs)
    options.update(
        implementation=IMPLEMENTATION,
        branch=BRANCH,
        program_recipe=PROGRAM_RECIPE,
        source_host_dispatch=True)
    return options


def _validate_retained(retained_kv_rows):
    if (type(retained_kv_rows) is not int
            or not 0 <= retained_kv_rows <= paged_mla_arch22.KEY_TILE):
        raise ModelingError('paged_mla_arch22_retained_rows_must_fit_source_s2_tile')


def native_contract(**kwargs):
    """Arch22 source host partitioning is implicit when ``dispatch=None``."""
    _validate_retained(kwargs.get('retained_kv_rows', 0))
    return family_paged_mla.native_contract(**_common_kwargs(kwargs))


def native_plan(**kwargs):
    """Return the arch22 recipe plan for the same declaration."""
    _validate_retained(kwargs.get('retained_kv_rows', 0))
    options = _common_kwargs(kwargs)
    options.setdefault('build', paged_mla_arch22.build)
    return family_paged_mla.native_plan(**options)


def validate(*, dispatch=None, retained_kv_rows=0, native_query_capacity=None, **kwargs):
    family_paged_mla.validate(dispatch=dispatch,
                              native_query_capacity=native_query_capacity, **kwargs)
    _validate_retained(retained_kv_rows)


def lower_attention(b, *, padded_tokens=None, dispatch=None, retained_kv_rows=0,
                    native_query_capacity=None, **kwargs):
    _validate_retained(retained_kv_rows)
    return family_paged_mla.lower_attention(
        b, padded_tokens=padded_tokens, dispatch=dispatch,
        native_query_capacity=native_query_capacity,
        retained_kv_rows=retained_kv_rows,
        implementation=IMPLEMENTATION, branch=BRANCH,
        program_recipe=PROGRAM_RECIPE, source_host_dispatch=True, **kwargs)
