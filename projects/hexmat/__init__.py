"""hexmat — Линейная алгебра над GF(2) на Q6 = (GF(2))^6."""
from .hexmat import (
    mat_zero, mat_identity, mat_from_cols, mat_from_rows, mat_permutation,
    mat_add, mat_neg, mat_transpose, mat_vec_mul, mat_mul, mat_pow,
    mat_trace, mat_det, is_invertible,
    row_reduce, mat_rank,
    mat_kernel, mat_image, mat_column_space,
    mat_inv,
    mat_hadamard_gf2, symplectic_matrix,
    gl6_order, random_invertible, count_invertible_6x6,
    orthogonal_complement,
    linear_code_from_generator, parity_check_matrix, minimum_distance,
)
