"""hexgf — Поле Галуа GF(2^6) на 64 элементах."""
from .hexgf import (
    POLY, POLY_REDUCE, ORDER, SIZE, PRIMITIVE,
    gf_add, gf_sub, gf_mul, gf_pow, gf_inv, gf_div,
    gf_exp, gf_log, gf_mul_via_log, build_exp_log_tables,
    gf_trace, gf_norm, trace_bilinear,
    element_order, is_primitive, primitive_elements, count_primitive,
    cyclotomic_coset_of_exp, cyclotomic_coset_of, all_cyclotomic_cosets,
    minimal_polynomial, poly_eval_gf,
    subfield_elements,
    build_zech_log_table,
    additive_character, additive_character_b, character_sum,
    bch_zeros, bch_generator_degree,
    frobenius, frobenius_orbit,
)
