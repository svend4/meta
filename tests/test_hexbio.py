"""Тесты для hexbio — биоинформатика на Q6."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from projects.hexbio.hexbio import (
    codon_to_int, int_to_codon, codon_nucleotides,
    point_mutations, mutation_distance, hamming_distance_bits, synonymous_path,
    translate, stop_codons, synonymous_codons, degeneracy_class,
    is_synonymous_mutation, synonymous_neighbors, nonsynonymous_neighbors,
    third_position_structure, wobble_pairs,
    gc_content, purine_count,
    rscu, codon_adaptation_index_weights,
    amino_acid_distance,
    amino_acids, codon_table_summary,
    synonymous_mutation_fraction, mutation_graph_edges,
    q6_vs_mutation_comparison,
)


# ============================================================
# Кодирование нуклеотидов
# ============================================================

class TestCodonEncoding:
    def test_codon_to_int_uuu(self):
        # U=11, UUU = 11_11_11 = 63
        assert codon_to_int('UUU') == 0b111111

    def test_codon_to_int_aaa(self):
        # A=00, AAA = 00_00_00 = 0
        assert codon_to_int('AAA') == 0

    def test_codon_to_int_aug(self):
        # AUG = 00_11_10 = 0+12+2 = 14
        assert codon_to_int('AUG') == 0b001110

    def test_codon_to_int_gcu(self):
        # GCU = 10_01_11 = 32+4+3 = 39
        assert codon_to_int('GCU') == 0b100111

    def test_codon_to_int_dna(self):
        # DNA: T→U
        assert codon_to_int('ATG') == codon_to_int('AUG')

    def test_codon_to_int_lowercase(self):
        assert codon_to_int('aug') == codon_to_int('AUG')

    def test_int_to_codon_roundtrip(self):
        for n in range(64):
            assert codon_to_int(int_to_codon(n)) == n

    def test_int_to_codon_uuu(self):
        assert int_to_codon(63) == 'UUU'

    def test_int_to_codon_aaa(self):
        assert int_to_codon(0) == 'AAA'

    def test_int_to_codon_aug(self):
        assert int_to_codon(codon_to_int('AUG')) == 'AUG'

    def test_codon_nucleotides(self):
        n = codon_to_int('AUG')
        x, y, z = codon_nucleotides(n)
        assert x == 'A' and y == 'U' and z == 'G'

    def test_codon_to_int_invalid(self):
        with pytest.raises(ValueError):
            codon_to_int('XYZ')

    def test_codon_to_int_wrong_length(self):
        with pytest.raises(ValueError):
            codon_to_int('AU')

    def test_all_64_codons_covered(self):
        codons = {int_to_codon(n) for n in range(64)}
        assert len(codons) == 64


# ============================================================
# Граф мутаций
# ============================================================

class TestMutationGraph:
    def test_point_mutations_count(self):
        # У каждого кодона ровно 9 однонуклеотидных соседей
        for n in range(64):
            assert len(point_mutations(n)) == 9

    def test_point_mutations_aaa(self):
        # AAA мутирует в CAA, GAA, UAA, ACA, AGA, AUA, AAC, AAG, AAU
        n = codon_to_int('AAA')
        neighbors = {int_to_codon(m) for m in point_mutations(n)}
        expected = {'CAA', 'GAA', 'UAA', 'ACA', 'AGA', 'AUA', 'AAC', 'AAG', 'AAU'}
        assert neighbors == expected

    def test_point_mutations_symmetry(self):
        # Если b в мутациях(a), то a в мутациях(b)
        for a in range(64):
            for b in point_mutations(a):
                assert a in point_mutations(b)

    def test_mutation_distance_same(self):
        assert mutation_distance(42, 42) == 0

    def test_mutation_distance_one(self):
        # AAA → CAA: одна позиция различается
        a = codon_to_int('AAA')
        b = codon_to_int('CAA')
        assert mutation_distance(a, b) == 1

    def test_mutation_distance_two(self):
        a = codon_to_int('AAA')
        b = codon_to_int('CCA')
        assert mutation_distance(a, b) == 2

    def test_mutation_distance_three(self):
        a = codon_to_int('AAA')
        b = codon_to_int('UUU')
        assert mutation_distance(a, b) == 3

    def test_hamming_distance_bits_same(self):
        assert hamming_distance_bits(42, 42) == 0

    def test_hamming_distance_bits_one(self):
        # Q6-сосед: ровно 1 бит различается
        assert hamming_distance_bits(0, 1) == 1

    def test_hamming_distance_bits_six(self):
        # AAA=0 и UUU=63: все 6 бит различаются
        assert hamming_distance_bits(0, 63) == 6

    def test_mutation_graph_edges_count(self):
        edges = mutation_graph_edges()
        # 64 вершины × 9 соседей / 2 = 288
        assert len(edges) == 288

    def test_mutation_graph_edges_sorted(self):
        edges = mutation_graph_edges()
        for a, b in edges:
            assert a < b

    def test_q6_vs_mutation(self):
        cmp = q6_vs_mutation_comparison()
        assert cmp['q6_neighbors'] == 6
        assert cmp['mutation_neighbors'] == 9
        assert cmp['q6_edges'] == 192
        assert cmp['mutation_edges'] == 288
        # Общих рёбер: не более min(192, 288)
        assert 0 <= cmp['shared_edges'] <= 192


# ============================================================
# Генетический код
# ============================================================

class TestGeneticCode:
    def test_translate_aug_is_met(self):
        assert translate(codon_to_int('AUG')) == 'M'

    def test_translate_uaa_is_stop(self):
        assert translate(codon_to_int('UAA')) == '*'

    def test_translate_uag_is_stop(self):
        assert translate(codon_to_int('UAG')) == '*'

    def test_translate_uga_is_stop(self):
        assert translate(codon_to_int('UGA')) == '*'

    def test_all_64_codons_have_aa(self):
        for n in range(64):
            aa = translate(n)
            assert aa in 'ACDEFGHIKLMNPQRSTVWY*'

    def test_stop_codons_count(self):
        stops = stop_codons()
        assert len(stops) == 3

    def test_stop_codons_values(self):
        stops = {int_to_codon(c) for c in stop_codons()}
        assert stops == {'UAA', 'UAG', 'UGA'}

    def test_synonymous_codons_met(self):
        # M (Met) кодируется только AUG
        mets = synonymous_codons('M')
        assert len(mets) == 1
        assert int_to_codon(mets[0]) == 'AUG'

    def test_synonymous_codons_leu(self):
        # L (Leu) кодируется 6 кодонами
        leus = synonymous_codons('L')
        assert len(leus) == 6

    def test_synonymous_codons_ser(self):
        # S (Ser) кодируется 6 кодонами
        sers = synonymous_codons('S')
        assert len(sers) == 6

    def test_degeneracy_class_met(self):
        assert degeneracy_class(codon_to_int('AUG')) == 1

    def test_degeneracy_class_leu(self):
        assert degeneracy_class(codon_to_int('CUU')) == 6

    def test_amino_acids_count(self):
        aas = amino_acids()
        assert len(aas) == 20

    def test_codon_table_summary_all_aas(self):
        summary = codon_table_summary()
        assert '*' in summary
        assert len(summary) == 21  # 20 AA + stop


# ============================================================
# Синонимичные/несинонимичные мутации
# ============================================================

class TestSynonymousMutations:
    def test_synonymous_mutation_phe(self):
        # UUU → UUC: оба Phe
        a = codon_to_int('UUU')
        b = codon_to_int('UUC')
        assert is_synonymous_mutation(a, b) is True

    def test_nonsynonymous_mutation_phe_leu(self):
        # UUU → UUA: Phe → Leu
        a = codon_to_int('UUU')
        b = codon_to_int('UUA')
        assert is_synonymous_mutation(a, b) is False

    def test_synonymous_neighbors_count(self):
        # У каждого кодона ≥ 0 синонимичных соседей
        for n in range(64):
            neighbors = synonymous_neighbors(n)
            assert len(neighbors) >= 0

    def test_synonymous_neighbors_subset(self):
        # Синонимичные ⊆ все мутации
        for n in range(64):
            syn = set(synonymous_neighbors(n))
            all_mut = set(point_mutations(n))
            assert syn.issubset(all_mut)

    def test_nonsynonymous_neighbors_subset(self):
        for n in range(64):
            nonsyn = set(nonsynonymous_neighbors(n))
            all_mut = set(point_mutations(n))
            assert nonsyn.issubset(all_mut)

    def test_syn_and_nonsyn_partition(self):
        # Синонимичные + несинонимичные = все мутации
        for n in range(64):
            syn = set(synonymous_neighbors(n))
            nonsyn = set(nonsynonymous_neighbors(n))
            all_mut = set(point_mutations(n))
            assert syn | nonsyn == all_mut
            assert syn & nonsyn == set()

    def test_synonymous_mutation_fraction_range(self):
        frac = synonymous_mutation_fraction()
        assert 0.0 < frac < 1.0

    def test_synonymous_path_same(self):
        n = codon_to_int('GCU')
        path = synonymous_path(n, n)
        assert path == [n]

    def test_synonymous_path_ala(self):
        # GCU, GCC, GCA, GCG — все Ala; должен быть синонимичный путь
        a = codon_to_int('GCU')
        b = codon_to_int('GCG')
        path = synonymous_path(a, b)
        assert path is not None
        assert path[0] == a
        assert path[-1] == b


# ============================================================
# Структура 3-й позиции, wobble
# ============================================================

class TestThirdPosition:
    def test_third_position_structure_keys(self):
        struct = third_position_structure()
        assert len(struct) == 16  # 4×4 = 16 боксов

    def test_third_position_fourfold_ala(self):
        # GC*: GCU/GCC/GCA/GCG — все Ala → fourfold
        struct = third_position_structure()
        assert struct['GC']['fourfold'] is True

    def test_third_position_fourfold_count(self):
        struct = third_position_structure()
        fourfold = sum(1 for v in struct.values() if v['fourfold'])
        # В стандартном генетическом коде: 8 четырёхкратных боксов
        assert fourfold == 8

    def test_wobble_pairs_count(self):
        pairs = wobble_pairs()
        assert len(pairs) > 0

    def test_wobble_pairs_same_aa(self):
        for a, b in wobble_pairs():
            assert translate(a) == translate(b)

    def test_wobble_pairs_same_first_two(self):
        # Wobble-пары отличаются только 3-й позицией
        for a, b in wobble_pairs():
            assert (a >> 2) == (b >> 2)


# ============================================================
# GC-содержание и пурины
# ============================================================

class TestNucleotideContent:
    def test_gc_content_aaa(self):
        assert gc_content(codon_to_int('AAA')) == 0.0

    def test_gc_content_ggg(self):
        n = (_NUC_BIT('G') << 4) | (_NUC_BIT('G') << 2) | _NUC_BIT('G')
        assert gc_content(n) == 1.0

    def test_gc_content_aug(self):
        # AUG: A=пурин/не GC, U=не GC, G=GC → 1/3
        assert abs(gc_content(codon_to_int('AUG')) - 1/3) < 1e-10

    def test_gc_content_range(self):
        for n in range(64):
            assert 0.0 <= gc_content(n) <= 1.0

    def test_purine_count_aaa(self):
        assert purine_count(codon_to_int('AAA')) == 3

    def test_purine_count_uuu(self):
        assert purine_count(codon_to_int('UUU')) == 0

    def test_purine_count_aug(self):
        # AUG: A=пурин, U=пиримидин, G=пурин → 2
        assert purine_count(codon_to_int('AUG')) == 2

    def test_purine_count_range(self):
        for n in range(64):
            assert 0 <= purine_count(n) <= 3


def _NUC_BIT(c):
    return {'A': 0b00, 'C': 0b01, 'G': 0b10, 'U': 0b11}[c]


# ============================================================
# RSCU и CAI
# ============================================================

class TestRSCU:
    def test_rscu_uniform_usage(self):
        # Равномерное использование → RSCU=1 для всех кодонов
        usage = {n: 1 for n in range(64)}
        result = rscu(usage)
        for n in range(64):
            if translate(n) != '*':
                assert abs(result[n] - 1.0) < 1e-10, f"RSCU({n}) = {result[n]}"

    def test_rscu_single_codon_preferred(self):
        # Если только один кодон из синонимов используется
        aa = 'F'  # Phe: UUU, UUC
        codons = synonymous_codons(aa)
        usage = {n: 0 for n in range(64)}
        usage[codons[0]] = 100
        result = rscu(usage)
        # Используемый кодон имеет RSCU = deg (т.к. ожидаемое = 100/2=50, наблюдаемое=100)
        assert result[codons[0]] > result[codons[1]]

    def test_cai_weights_range(self):
        usage = {n: max(1, n) for n in range(64)}
        weights = codon_adaptation_index_weights(usage)
        for n in range(64):
            assert 0.0 <= weights[n] <= 1.0 + 1e-10

    def test_cai_max_weight_one(self):
        # Максимальный вес в каждой группе = 1.0
        usage = {n: max(1, n) for n in range(64)}
        weights = codon_adaptation_index_weights(usage)
        aa_codons: dict = {}
        for n in range(64):
            aa = translate(n)
            aa_codons.setdefault(aa, []).append(n)
        for aa, codons in aa_codons.items():
            max_w = max(weights[c] for c in codons)
            assert abs(max_w - 1.0) < 1e-10 or max_w == 0.0


# ============================================================
# Расстояние Гранхэма
# ============================================================

class TestGranthamDistance:
    def test_same_aa_distance_zero(self):
        assert amino_acid_distance('A', 'A') == 0

    def test_known_distance(self):
        # A-R = 112 (из таблицы Гранхэма)
        assert amino_acid_distance('A', 'R') == 112

    def test_symmetric(self):
        assert amino_acid_distance('A', 'R') == amino_acid_distance('R', 'A')

    def test_il_distance_small(self):
        # I-L = 5 (наиболее похожие)
        assert amino_acid_distance('I', 'L') == 5

    def test_cw_distance_max(self):
        # C-W = 215 (наиболее разные)
        assert amino_acid_distance('C', 'W') == 215
