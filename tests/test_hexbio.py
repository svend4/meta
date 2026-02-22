"""Тесты для hexbio — биоинформатика на Q6."""
import sys
import os
import io
import unittest
from contextlib import redirect_stdout
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
    evolutionary_distance_matrix,
    _cmd_info, _cmd_codon, _cmd_mutation, _cmd_graph, _cmd_wobble,
)


class TestCodonEncoding(unittest.TestCase):
    def test_codon_to_int_uuu(self):
        self.assertEqual(codon_to_int('UUU'), 0b111111)
    def test_codon_to_int_aaa(self):
        self.assertEqual(codon_to_int('AAA'), 0)
    def test_codon_to_int_aug(self):
        self.assertEqual(codon_to_int('AUG'), 0b001110)
    def test_codon_to_int_gcu(self):
        self.assertEqual(codon_to_int('GCU'), 0b100111)
    def test_codon_to_int_dna(self):
        self.assertEqual(codon_to_int('ATG'), codon_to_int('AUG'))
    def test_codon_to_int_lowercase(self):
        self.assertEqual(codon_to_int('aug'), codon_to_int('AUG'))
    def test_int_to_codon_roundtrip(self):
        for n in range(64):
            self.assertEqual(codon_to_int(int_to_codon(n)), n)
    def test_int_to_codon_uuu(self):
        self.assertEqual(int_to_codon(63), 'UUU')
    def test_int_to_codon_aaa(self):
        self.assertEqual(int_to_codon(0), 'AAA')
    def test_int_to_codon_aug(self):
        self.assertEqual(int_to_codon(codon_to_int('AUG')), 'AUG')

    def test_int_to_codon_out_of_range(self):
        """int_to_codon вне диапазона 0-63 вызывает ValueError."""
        with self.assertRaises(ValueError):
            int_to_codon(-1)
        with self.assertRaises(ValueError):
            int_to_codon(64)
    def test_codon_nucleotides(self):
        n = codon_to_int('AUG')
        x, y, z = codon_nucleotides(n)
        self.assertEqual(x, 'A'); self.assertEqual(y, 'U'); self.assertEqual(z, 'G')
    def test_codon_to_int_invalid(self):
        with self.assertRaises(ValueError): codon_to_int('XYZ')
    def test_codon_to_int_wrong_length(self):
        with self.assertRaises(ValueError): codon_to_int('AU')
    def test_all_64_codons_covered(self):
        self.assertEqual(len({int_to_codon(n) for n in range(64)}), 64)


class TestMutationGraph(unittest.TestCase):
    def test_point_mutations_count(self):
        for n in range(64): self.assertEqual(len(point_mutations(n)), 9)
    def test_point_mutations_aaa(self):
        n = codon_to_int('AAA')
        self.assertEqual({int_to_codon(m) for m in point_mutations(n)},
                         {'CAA','GAA','UAA','ACA','AGA','AUA','AAC','AAG','AAU'})
    def test_point_mutations_symmetry(self):
        for a in range(64):
            for b in point_mutations(a): self.assertIn(a, point_mutations(b))
    def test_mutation_distance_same(self):
        self.assertEqual(mutation_distance(42, 42), 0)
    def test_mutation_distance_one(self):
        self.assertEqual(mutation_distance(codon_to_int('AAA'), codon_to_int('CAA')), 1)
    def test_mutation_distance_two(self):
        self.assertEqual(mutation_distance(codon_to_int('AAA'), codon_to_int('CCA')), 2)
    def test_mutation_distance_three(self):
        self.assertEqual(mutation_distance(codon_to_int('AAA'), codon_to_int('UUU')), 3)
    def test_hamming_distance_bits_same(self):
        self.assertEqual(hamming_distance_bits(42, 42), 0)
    def test_hamming_distance_bits_one(self):
        self.assertEqual(hamming_distance_bits(0, 1), 1)
    def test_hamming_distance_bits_six(self):
        self.assertEqual(hamming_distance_bits(0, 63), 6)
    def test_mutation_graph_edges_count(self):
        self.assertEqual(len(mutation_graph_edges()), 288)
    def test_mutation_graph_edges_sorted(self):
        for a, b in mutation_graph_edges(): self.assertLess(a, b)
    def test_q6_vs_mutation(self):
        cmp = q6_vs_mutation_comparison()
        self.assertEqual(cmp['q6_neighbors'], 6)
        self.assertEqual(cmp['mutation_neighbors'], 9)
        self.assertEqual(cmp['q6_edges'], 192)
        self.assertEqual(cmp['mutation_edges'], 288)
        self.assertGreaterEqual(cmp['shared_edges'], 0)
        self.assertLessEqual(cmp['shared_edges'], 192)


class TestGeneticCode(unittest.TestCase):
    def test_translate_aug_is_met(self):
        self.assertEqual(translate(codon_to_int('AUG')), 'M')
    def test_translate_uaa_is_stop(self):
        self.assertEqual(translate(codon_to_int('UAA')), '*')
    def test_translate_uag_is_stop(self):
        self.assertEqual(translate(codon_to_int('UAG')), '*')
    def test_translate_uga_is_stop(self):
        self.assertEqual(translate(codon_to_int('UGA')), '*')
    def test_all_64_codons_have_aa(self):
        for n in range(64): self.assertIn(translate(n), 'ACDEFGHIKLMNPQRSTVWY*')
    def test_stop_codons_count(self):
        self.assertEqual(len(stop_codons()), 3)
    def test_stop_codons_values(self):
        self.assertEqual({int_to_codon(c) for c in stop_codons()}, {'UAA','UAG','UGA'})
    def test_synonymous_codons_met(self):
        mets = synonymous_codons('M')
        self.assertEqual(len(mets), 1); self.assertEqual(int_to_codon(mets[0]), 'AUG')
    def test_synonymous_codons_leu(self):
        self.assertEqual(len(synonymous_codons('L')), 6)
    def test_synonymous_codons_ser(self):
        self.assertEqual(len(synonymous_codons('S')), 6)
    def test_degeneracy_class_met(self):
        self.assertEqual(degeneracy_class(codon_to_int('AUG')), 1)
    def test_degeneracy_class_leu(self):
        self.assertEqual(degeneracy_class(codon_to_int('CUU')), 6)
    def test_amino_acids_count(self):
        self.assertEqual(len(amino_acids()), 20)
    def test_codon_table_summary_all_aas(self):
        summary = codon_table_summary()
        self.assertIn('*', summary); self.assertEqual(len(summary), 21)


class TestSynonymousMutations(unittest.TestCase):
    def test_synonymous_mutation_phe(self):
        self.assertTrue(is_synonymous_mutation(codon_to_int('UUU'), codon_to_int('UUC')))
    def test_nonsynonymous_mutation_phe_leu(self):
        self.assertFalse(is_synonymous_mutation(codon_to_int('UUU'), codon_to_int('UUA')))
    def test_synonymous_neighbors_count(self):
        for n in range(64): self.assertGreaterEqual(len(synonymous_neighbors(n)), 0)
    def test_synonymous_neighbors_subset(self):
        for n in range(64):
            self.assertTrue(set(synonymous_neighbors(n)).issubset(set(point_mutations(n))))
    def test_nonsynonymous_neighbors_subset(self):
        for n in range(64):
            self.assertTrue(set(nonsynonymous_neighbors(n)).issubset(set(point_mutations(n))))
    def test_syn_and_nonsyn_partition(self):
        for n in range(64):
            syn = set(synonymous_neighbors(n)); nonsyn = set(nonsynonymous_neighbors(n))
            all_mut = set(point_mutations(n))
            self.assertEqual(syn | nonsyn, all_mut); self.assertEqual(syn & nonsyn, set())
    def test_synonymous_mutation_fraction_range(self):
        frac = synonymous_mutation_fraction()
        self.assertGreater(frac, 0.0); self.assertLess(frac, 1.0)
    def test_synonymous_path_same(self):
        n = codon_to_int('GCU')
        self.assertEqual(synonymous_path(n, n), [n])
    def test_synonymous_path_ala(self):
        a = codon_to_int('GCU'); b = codon_to_int('GCG')
        path = synonymous_path(a, b)
        self.assertIsNotNone(path); self.assertEqual(path[0], a); self.assertEqual(path[-1], b)

    def test_synonymous_path_different_aa_returns_none(self):
        """synonymous_path возвращает None для кодонов разных аминокислот."""
        a = codon_to_int('GCU')  # Ala
        b = codon_to_int('AUG')  # Met
        assert translate(a) != translate(b)
        path = synonymous_path(a, b)
        assert path is None

    def test_is_synonymous_mutation_non_adjacent_returns_false(self):
        """is_synonymous_mutation возвращает False при расстоянии != 1."""
        # GCU и CGU отличаются в двух позициях (расстояние > 1)
        a = codon_to_int('GCU')
        b = codon_to_int('CGU')
        assert mutation_distance(a, b) > 1
        assert is_synonymous_mutation(a, b) is False


class TestThirdPosition(unittest.TestCase):
    def test_third_position_structure_keys(self):
        self.assertEqual(len(third_position_structure()), 16)
    def test_third_position_fourfold_ala(self):
        self.assertTrue(third_position_structure()['GC']['fourfold'])
    def test_third_position_fourfold_count(self):
        self.assertEqual(sum(1 for v in third_position_structure().values() if v['fourfold']), 8)
    def test_wobble_pairs_count(self):
        self.assertGreater(len(wobble_pairs()), 0)
    def test_wobble_pairs_same_aa(self):
        for a, b in wobble_pairs(): self.assertEqual(translate(a), translate(b))
    def test_wobble_pairs_same_first_two(self):
        for a, b in wobble_pairs(): self.assertEqual(a >> 2, b >> 2)


def _NUC_BIT(c):
    return {'A': 0b00, 'C': 0b01, 'G': 0b10, 'U': 0b11}[c]


class TestNucleotideContent(unittest.TestCase):
    def test_gc_content_aaa(self):
        self.assertEqual(gc_content(codon_to_int('AAA')), 0.0)
    def test_gc_content_ggg(self):
        n = (_NUC_BIT('G') << 4) | (_NUC_BIT('G') << 2) | _NUC_BIT('G')
        self.assertEqual(gc_content(n), 1.0)
    def test_gc_content_aug(self):
        self.assertAlmostEqual(gc_content(codon_to_int('AUG')), 1/3)
    def test_gc_content_range(self):
        for n in range(64):
            self.assertGreaterEqual(gc_content(n), 0.0); self.assertLessEqual(gc_content(n), 1.0)
    def test_purine_count_aaa(self):
        self.assertEqual(purine_count(codon_to_int('AAA')), 3)
    def test_purine_count_uuu(self):
        self.assertEqual(purine_count(codon_to_int('UUU')), 0)
    def test_purine_count_aug(self):
        self.assertEqual(purine_count(codon_to_int('AUG')), 2)
    def test_purine_count_range(self):
        for n in range(64):
            self.assertGreaterEqual(purine_count(n), 0); self.assertLessEqual(purine_count(n), 3)


class TestRSCU(unittest.TestCase):
    def test_rscu_uniform_usage(self):
        usage = {n: 1 for n in range(64)}
        result = rscu(usage)
        for n in range(64):
            if translate(n) != '*':
                self.assertAlmostEqual(result[n], 1.0, places=10)
    def test_rscu_single_codon_preferred(self):
        codons = synonymous_codons('F')
        usage = {n: 0 for n in range(64)}
        usage[codons[0]] = 100
        result = rscu(usage)
        self.assertGreater(result[codons[0]], result[codons[1]])
    def test_cai_weights_range(self):
        usage = {n: max(1, n) for n in range(64)}
        weights = codon_adaptation_index_weights(usage)
        for n in range(64):
            self.assertGreaterEqual(weights[n], 0.0); self.assertLessEqual(weights[n], 1.0 + 1e-10)
    def test_cai_max_weight_one(self):
        usage = {n: max(1, n) for n in range(64)}
        weights = codon_adaptation_index_weights(usage)
        aa_codons: dict = {}
        for n in range(64): aa_codons.setdefault(translate(n), []).append(n)
        for aa, codons in aa_codons.items():
            max_w = max(weights[c] for c in codons)
            self.assertTrue(abs(max_w - 1.0) < 1e-10 or max_w == 0.0)


class TestGranthamDistance(unittest.TestCase):
    def test_same_aa_distance_zero(self):
        self.assertEqual(amino_acid_distance('A', 'A'), 0)
    def test_known_distance(self):
        self.assertEqual(amino_acid_distance('A', 'R'), 112)
    def test_symmetric(self):
        self.assertEqual(amino_acid_distance('A', 'R'), amino_acid_distance('R', 'A'))
    def test_il_distance_small(self):
        self.assertEqual(amino_acid_distance('I', 'L'), 5)
    def test_cw_distance_max(self):
        self.assertEqual(amino_acid_distance('C', 'W'), 215)


# ============================================================
# Матрица эволюционных расстояний
# ============================================================

class TestEvolutionaryDistanceMatrix:
    def test_returns_dict(self):
        edm = evolutionary_distance_matrix()
        assert isinstance(edm, dict)

    def test_size(self):
        """Пар (a, b) с a < b: C(64, 2) = 2016."""
        edm = evolutionary_distance_matrix()
        assert len(edm) == 2016

    def test_keys_are_pairs(self):
        edm = evolutionary_distance_matrix()
        for key in list(edm.keys())[:5]:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_ordered_pairs(self):
        """Ключ (a, b) всегда a < b."""
        edm = evolutionary_distance_matrix()
        for a, b in edm:
            assert a < b

    def test_distance_positive(self):
        """Расстояние между разными кодонами > 0."""
        edm = evolutionary_distance_matrix()
        for dist in edm.values():
            assert dist > 0

    def test_distance_at_most_6(self):
        """В Q6 расстояние не превышает диаметр = 6."""
        edm = evolutionary_distance_matrix()
        for dist in edm.values():
            assert dist <= 6

    def test_neighbors_have_distance_1(self):
        """Соседи в Q6 имеют расстояние = 1."""
        from libs.hexcore.hexcore import neighbors
        edm = evolutionary_distance_matrix()
        h = 0
        for nb in neighbors(h):
            key = (min(h, nb), max(h, nb))
            assert edm[key] == 1


# ============================================================
# CLI-функции
# ============================================================

class TestCLI:
    def _capture(self, fn, args=None):
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(args or [])
        return buf.getvalue()

    def test_cmd_info_produces_output(self):
        out = self._capture(_cmd_info)
        assert len(out) > 0
        assert 'кодон' in out.lower() or 'Q6' in out

    def test_cmd_codon_valid(self):
        out = self._capture(_cmd_codon, ['GCU'])
        assert 'GCU' in out or 'Ala' in out

    def test_cmd_codon_no_args(self):
        out = self._capture(_cmd_codon, [])
        assert 'Использование' in out

    def test_cmd_mutation_valid(self):
        out = self._capture(_cmd_mutation, ['GCU', 'GCG'])
        assert 'расстояние' in out.lower() or 'GCU' in out

    def test_cmd_mutation_no_args(self):
        out = self._capture(_cmd_mutation, [])
        assert 'Использование' in out

    def test_cmd_graph_produces_output(self):
        out = self._capture(_cmd_graph)
        assert len(out) > 0

    def test_cmd_wobble_produces_output(self):
        out = self._capture(_cmd_wobble)
        assert len(out) > 0


class TestBioCLI:
    """Тесты для hexbio.main() (lines 577-595)."""

    def _run(self, args):
        import io
        from contextlib import redirect_stdout
        from projects.hexbio.hexbio import main
        old_argv = sys.argv
        sys.argv = ['hexbio.py'] + args
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_no_args_shows_usage(self):
        out = self._run([])
        assert 'Использование' in out

    def test_cmd_info(self):
        out = self._run(['info'])
        assert len(out) > 0

    def test_cmd_codon(self):
        out = self._run(['codon', 'GCU'])
        assert len(out) > 0

    def test_cmd_mutation(self):
        out = self._run(['mutation', 'GCU', 'GCG'])
        assert len(out) > 0

    def test_cmd_graph(self):
        out = self._run(['graph'])
        assert len(out) > 0

    def test_cmd_wobble(self):
        out = self._run(['wobble'])
        assert len(out) > 0

    def test_unknown_cmd(self):
        out = self._run(['unknown'])
        assert 'Неизвестная' in out


class TestSynonymousPath:
    """Тесты для synonymous_path BFS (lines 107-120)."""

    def test_synonymous_path_same_codon(self):
        """Путь от кодона до него самого (line 98-99)."""
        # GCU = 39 (Ala)
        path = synonymous_path(39, 39)
        assert path == [39]

    def test_synonymous_path_same_aa(self):
        """Синонимичный путь между GCU(39) и GCC(37) (оба Ala) — hits lines 111-118."""
        path = synonymous_path(39, 37)
        assert path is not None
        assert path[0] == 39
        assert path[-1] == 37

    def test_synonymous_path_returns_none_different_aa(self):
        """synonymous_path между кодонами разных AA → None (line 105-106)."""
        # GCU(39)=Ala and UUU(63)=Phe - different AA → early return None
        path = synonymous_path(39, 63)
        assert path is None

    def test_synonymous_path_returns_none_no_bfs_path(self):
        """synonymous_path когда BFS не находит путь → None (line 120)."""
        aug = codon_to_int('AUG')
        ugc = codon_to_int('UGC')  # Cys
        path = synonymous_path(aug, ugc)  # different AA → None at line 105
        assert path is None

    def test_synonymous_path_leu_multi_hop_triggers_visited(self):
        """BFS revisits visited nodes (line 112) for CUU(31)→UUA(60), both Leu."""
        path = synonymous_path(31, 60)   # CUU=31, UUA=60 both Leu
        assert path is not None
        assert path[0] == 31
        assert path[-1] == 60
        assert all(translate(c) == 'L' for c in path)

    def test_synonymous_path_ser_disconnected_returns_none(self):
        """BFS exhaustion (line 120): UCU(55) and AGU(11) both Ser but disconnected."""
        ucu = codon_to_int('UCU')  # 55, Ser (UC* group)
        agu = codon_to_int('AGU')  # 11, Ser (AG* group)
        path = synonymous_path(ucu, agu)
        assert path is None


if __name__ == '__main__':
    unittest.main()
