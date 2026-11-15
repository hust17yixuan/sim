"""Packed INT8 latent, BF16 RoPE and FP32 scales in a native MLA prolog.

[Role] Describe the per-tile cache ABI and its explicit analytic producer.
[Invariants] D=656 is 512 INT8 + 64 BF16 + 4 FP32 scales, not 656 INT8 values.
  Allocated cache capacity is not per-call traffic. UK and query outputs stay
  BF16. This profile does not reuse the arch22 BF16-cache timing program.
[Timeouts & Locks] Constant geometry arithmetic; no source/capture reads.
[Side-Effect Policy] Builds immutable graph records only.
"""
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.schema.operators import (
    FieldDisposition, ProgramFragment, SemanticAttr, SemanticRef, StateEffect,
    WorkItem, WorkSet,
)
from mindie_sim.schema.subgraphs import (
    EventPortSpec, PortSet, StatePortSpec, TensorPortSpec,
)

LAYOUT = 'packed_int8_bf16_rope_fp32_scale'
PROFILE = 'mla_prolog_packed_int8_tile128_analytic'
SOURCE = {
    'caller': {'repository': 'vllm-project/vllm-ascend',
               'revision': '8c30e4f491f73cba737a1dacc522894e437cdf3d',
               'path': 'vllm_ascend/attention/sfa_v1.py', 'lines': '1181-1289'},
    'layout': {'path': 'vllm_ascend/attention/utils.py', 'lines': '69-80'},
    'cache_writer': {'repository': 'ops-transformer',
                     'revision': 'a3925fd49cd256ad478d2faf983bf2961b8ae2c8',
                     'path': 'attention/mla_prolog/op_kernel/arch22/kernel_mla_prolog_split_n_arch22.h',
                     'lines': '1454-1556,1628-1645'},
    'capture_binary_identity': 'not_established; explicit_representation_and_analytic_program',
}


def view(latent, rope, group_size=128):
    if latent != 512 or rope != 64 or group_size != 128:
        raise ModelingError('mla_prolog_packed_int8_requires_latent512_rope64_tile128')
    return {'layout': LAYOUT, 'latent_dtype': 'int8', 'rope_dtype': 'bf16',
            'scale_dtype': 'fp32', 'group_size': group_size,
            'payload_bytes': latent, 'rope_bytes': rope * 2,
            'scale_bytes': latent // group_size * 4,
            'row_bytes': latent + rope * 2 + latent // group_size * 4,
            'component_offsets': {'latent': 0, 'rope': latent, 'scales': latent + rope * 2},
            'cache_mode': 'PA_BSND', 'kv_cache_quant_mode': 3,
            'ckvkr_repo_mode': 1, 'quant_scale_repo_mode': 1}


def _packed_plan(*, plan, rows):
    cache = view(plan.kv_lora_rank, plan.qk_rope_head_dim)
    if type(rows) is not int or isinstance(rows, bool) or rows < 0:
        raise ModelingError('mla_prolog_packed_requires_nonnegative_rows')
    width, groups = plan.kv_lora_rank, plan.kv_lora_rank // 128
    # The semantic norm producer is BF16. The native source uses FP32 norm
    # scratch; this analytic producer includes widening, group abs/max, clip,
    # normalize and three-step FP32->INT8 conversion, without instruction tiling.
    vector = rows * (9 * width + 4 * groups)
    read = rows * ((width + plan.qk_rope_head_dim) * 2 + 8) + 4
    return {
        'cache': cache, 'rows': rows, 'vector': vector,
        'read': read, 'write': rows * cache['row_bytes'],
        'hbm': read + rows * cache['row_bytes'],
        'groups': groups,
    }


def packed_contract(*, plan, rows, occurrence=None):
    """Typed packed-cache writer contract with component offsets and scales."""
    if occurrence is not None and not isinstance(occurrence, Occurrence):
        raise ModelingError('mla_prolog_packed_occurrence_invalid')
    spec = _packed_plan(plan=plan, rows=rows)
    cache = spec['cache']
    return ProgramFragment(
        'mla_prolog_packed_cache',
        occurrence=occurrence,
        semantic_refs=(SemanticRef('subgraph', 'mla_attention', role='mla_prolog_cache_store'),),
        attrs=(
            SemanticAttr('implementation', 'mla_prolog_packed_int8_tile128_analytic',
                         source='selected_program'),
            SemanticAttr('layout', cache['layout']),
            SemanticAttr('latent_dtype', cache['latent_dtype']),
            SemanticAttr('rope_dtype', cache['rope_dtype']),
            SemanticAttr('scale_dtype', cache['scale_dtype']),
            SemanticAttr('group_size', cache['group_size'], unit='row'),
            SemanticAttr('payload_bytes', cache['payload_bytes'], unit='byte'),
            SemanticAttr('rope_bytes', cache['rope_bytes'], unit='byte'),
            SemanticAttr('scale_bytes', cache['scale_bytes'], unit='byte'),
            SemanticAttr('row_bytes', cache['row_bytes'], unit='byte'),
            SemanticAttr('component_offsets', cache['component_offsets'], unit='byte'),
            SemanticAttr('cache_mode', cache['cache_mode']),
            SemanticAttr('kv_cache_quant_mode', cache['kv_cache_quant_mode']),
            SemanticAttr('ckvkr_repo_mode', cache['ckvkr_repo_mode']),
            SemanticAttr('quant_scale_repo_mode', cache['quant_scale_repo_mode']),
            SemanticAttr('rows', spec['rows'], unit='row'),
            SemanticAttr('source', SOURCE, source='source_binding'),
            SemanticAttr('unresolved_fields',
                         ('capture_binary_identity', 'native_instruction_tiling'),
                         source='specific_request'),
        ),
        ports=PortSet(
            tensor_in=(
                TensorPortSpec('kv_latent_norm', 'in', 'activation',
                               dtype='bf16', layout_kind='ND', scope='layer'),
                TensorPortSpec('rope_input', 'in', 'activation',
                               dtype='bf16', layout_kind='ND', scope='layer'),
            ),
            tensor_out=(
                TensorPortSpec('packed_cache', 'out', 'packed_latent_cache',
                               dtype='mixed_int8_bf16',
                               layout_kind=cache['cache_mode'], scope='cohort'),
                TensorPortSpec('packed_scale', 'out', 'quant_scale',
                               dtype=cache['scale_dtype'], layout_kind='ND',
                               scope='cohort'),
            ),
            state_out=(
                StatePortSpec('kv_cache_state', 'out', 'paged_latent_kv',
                              scope='cohort', readiness='producer_complete'),
            ),
            event_in=(EventPortSpec('kv_latent_ready', 'in', 'ready'),),
            event_out=(EventPortSpec('done', 'out', 'complete'),),
        ),
        work=WorkSet((
            WorkItem('hbm', spec['hbm'], 'bytes', 'issued', 'selected_packed_writer'),
            WorkItem('mte2', spec['read'], 'bytes', 'issued', 'selected_packed_writer'),
            WorkItem('mte3', spec['write'], 'bytes', 'issued', 'selected_packed_writer'),
            WorkItem('vector', spec['vector'], 'operations', 'issued',
                     'analytic_quantization_and_scatter'),
        )),
        state_effects=(
            StateEffect('kv_cache', 'write', condition='active_rows',
                        arbiter='packed_cache_writer'),
        ),
        field_effects=(
            FieldDisposition('kv_latent.dtype', 'rewrite',
                             source='bf16 normalized latent -> int8 payload'),
            FieldDisposition('rope.dtype', 'preserve'),
            FieldDisposition('scale.dtype', 'rewrite',
                             source='fp32 group scale written in packed row'),
            FieldDisposition('layout', 'rewrite', source='PA_BSND packed row'),
            FieldDisposition('cache.alias', 'preserve',
                             source='cache_output_alias'),
        ),
        notes=('D=656 is int8 payload + bf16 rope + fp32 scales, not one dtype',
               'work is the actual packed writer traffic and quantization ops'),
    )


def store(builder, name, deps, rows, plan):
    spec = _packed_plan(plan=plan, rows=rows)
    cache = spec['cache']
    return builder.work(
        name, deps, (rows, cache['row_bytes']), family='cache_store', dtype='int8',
        read=spec['read'], write=spec['write'], vector=spec['vector'],
        metadata=(('cache_kind', name), ('cache_visible', True), ('shared_owner', False),
                  ('packed_cache', cache), ('source', SOURCE),
                  ('cache_producer', 'analytic_tile128_quantization_and_packed_scatter'),
                  ('slot_index_dtype', 'int64'),
                  ('cache_output_alias', True)))


def validate_view(declaration):
    expected = view(declaration['latent_rank'], declaration['rope_width'])
    if (declaration.get('packed_cache') != expected
            or declaration.get('token_dtype') != 'int8'
            or tuple(declaration.get('projection_weight_dtypes', ())) != ('int8',) * 3
            or declaration.get('absorbed_weight_dtype') != 'bf16'
            or declaration.get('norm_dtype') != 'bf16'
            or declaration.get('cache_dtype') != 'int8'
            or declaration.get('query_norm_dtype') != 'int8'):
        raise ModelingError('mla_prolog_packed_requires_int8_token_projections_bf16_uk_and_tile128_cache')
