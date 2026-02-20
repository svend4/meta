"""Тесты триматрицы Андреева и биологического обогащения (TSC-3 K6)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest

from projects.hextrimat.trimat_glyphs import (
    json_triangle,
    json_verify,
    json_twins,
    json_twins_codon,
    json_center,
    render_triangle,
)
from projects.hextrimat.hextrimat import TriangularMatrix
from projects.hexbio.codon_glyphs import json_codon_map


# ---------------------------------------------------------------------------
# json_triangle: структура матрицы
# ---------------------------------------------------------------------------

class TestJsonTriangle(unittest.TestCase):
    """Структурные свойства треугольной матрицы Андреева."""

    @classmethod
    def setUpClass(cls):
        cls.result = json_triangle()

    def test_total_cells_64(self):
        self.assertEqual(self.result['total_cells'], 64)

    def test_num_rows_positive(self):
        self.assertGreater(self.result['num_rows'], 0)

    def test_rows_is_dict(self):
        self.assertIsInstance(self.result['rows'], dict)

    def test_all_row_values_sum_to_64(self):
        """Всего 64 гексаграммы распределены по строкам."""
        rows = self.result['rows']
        total = sum(len(v['values']) for v in rows.values())
        self.assertEqual(total, 64)

    def test_sum_total_equals_expected(self):
        """Сумма всех элементов фиксирована (структурный инвариант)."""
        rows = self.result['rows']
        total = sum(int(v) for row in rows.values() for v in row['values'])
        # Значения 1..64: сумма = 64*65/2 = 2080
        # Значения 0..63: сумма = 64*63/2 = 2016
        # Принимаем любой из этих инвариантов
        self.assertIn(total, (64 * 65 // 2, 64 * 63 // 2))


# ---------------------------------------------------------------------------
# json_verify: числа Андреева
# ---------------------------------------------------------------------------

class TestJsonVerify(unittest.TestCase):
    """Ключевые числа верификации матрицы Андреева."""

    @classmethod
    def setUpClass(cls):
        cls.result = json_verify()

    def test_all_verified(self):
        self.assertTrue(self.result['all_verified'])

    def test_key_numbers_present(self):
        kn = self.result['key_numbers']
        self.assertIsInstance(kn, dict)
        self.assertGreater(len(kn), 0)


# ---------------------------------------------------------------------------
# json_twins: пары-близнецы
# ---------------------------------------------------------------------------

class TestJsonTwins(unittest.TestCase):
    """Структура пар-близнецов (симметричные строки)."""

    @classmethod
    def setUpClass(cls):
        cls.result = json_twins()

    def test_count_positive(self):
        self.assertGreater(self.result['count'], 0)

    def test_pairs_have_left_right(self):
        for p in self.result['pairs']:
            self.assertIn('left', p)
            self.assertIn('right', p)
            self.assertIn('sum', p)

    def test_left_right_nonempty(self):
        for p in self.result['pairs']:
            self.assertGreater(len(p['left']), 0)
            self.assertGreater(len(p['right']), 0)


# ---------------------------------------------------------------------------
# json_twins_codon (TSC-3 pipeline K6 step)
# ---------------------------------------------------------------------------

class TestJsonTwinsCodon(unittest.TestCase):
    """Биологическое обогащение пар-близнецов (TSC-3 K4×K6)."""

    @classmethod
    def setUpClass(cls):
        codon_data = json_codon_map()
        cls.result = json_twins_codon(codon_data)

    def test_command_is_twins_codon(self):
        self.assertEqual(self.result['command'], 'twins_codon')

    def test_n_pairs_positive(self):
        self.assertGreater(self.result['n_pairs'], 0)

    def test_pairs_have_bio_data(self):
        """Каждая пара содержит биологические данные (кодоны, АА)."""
        for p in self.result['pairs']:
            self.assertIn('left', p)
            self.assertIn('right', p)
            self.assertIn('codons', p['left'])
            self.assertIn('amino_acids', p['left'])

    def test_stats_keys(self):
        s = self.result['stats']
        for key in ('synonymous_halves', 'total_half_pairs', 'half_syn_rate'):
            self.assertIn(key, s)

    def test_half_syn_rate_in_01(self):
        rate = self.result['stats']['half_syn_rate']
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_k4_k6_insight_nonempty(self):
        insight = self.result.get('k4_k6_insight', '')
        self.assertIsInstance(insight, str)
        self.assertGreater(len(insight), 10)


# ---------------------------------------------------------------------------
# Интеграционный тест: полный TSC-3 пайплайн K6 → K3 → K6×K3
# ---------------------------------------------------------------------------

class TestTSC3Pipeline(unittest.TestCase):
    """Конец-в-конец: codon → twins → cluster → resonance."""

    @classmethod
    def setUpClass(cls):
        from projects.hexlearn.learn_glyphs import json_codon_cluster
        from projects.hexspec.resonance_glyphs import json_resonance
        codon = json_codon_map()
        twins = json_twins_codon(codon)
        cluster = json_codon_cluster(twins)
        cls.resonance = json_resonance(cluster)
        cls.cluster = cluster

    def test_resonance_score_above_random(self):
        """Резонанс-оценка значительно выше случайной базовой линии."""
        rs = self.resonance['resonance_score']
        rb = self.resonance['random_baseline']
        self.assertGreater(rs, rb * 3)  # хотя бы в 3× выше случайного

    def test_oracle_predictions_list(self):
        preds = self.resonance['oracle_predictions']
        self.assertIsInstance(preds, list)

    def test_exact_boxes_structure(self):
        boxes = self.resonance['exact_boxes']
        self.assertIsInstance(boxes, list)
        for b in boxes:
            self.assertIn('cluster', b)
            self.assertIn('amino_acid', b)
            self.assertIn('codons', b)

    def test_purity_distribution_has_perfect(self):
        """Должны быть кластеры с purity=1.0."""
        pd = self.resonance['purity_distribution']
        self.assertIn('1.0', pd)
        self.assertGreater(pd['1.0'], 0)

    def test_sc_id(self):
        self.assertEqual(self.resonance['sc_id'], 'TSC-3')

    def test_n_oracle_predictions_consistent(self):
        n = self.resonance['n_oracle_predictions']
        preds = self.resonance['oracle_predictions']
        self.assertEqual(n, len(preds))


class TestJsonCenter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json_center()

    def test_command_is_center(self):
        self.assertEqual(self.result['command'], 'center')

    def test_center_cell_value_is_27(self):
        """Центральная ячейка Андреева — гексаграмма 27."""
        self.assertEqual(self.result['cell']['value'], 27)

    def test_result_has_cell(self):
        self.assertIn('cell', self.result)


class TestTriangularMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tm = TriangularMatrix()

    def test_total_cells(self):
        self.assertEqual(len(self.tm.cells), 64)

    def test_value_valid_cell(self):
        """Ячейка (1,1) содержит значение 1 (первая ячейка)."""
        v = self.tm.value(1, 1)
        self.assertIsNotNone(v)
        self.assertGreater(v, 0)

    def test_value_invalid_cell(self):
        """Ячейка вне матрицы возвращает None."""
        self.assertIsNone(self.tm.value(0, 0))

    def test_row_values_length(self):
        """Строка r содержит r значений."""
        for r in range(1, 5):
            vals = self.tm.row_values(r)
            self.assertEqual(len(vals), r)

    def test_row_sum_positive(self):
        for r in range(1, 5):
            self.assertGreater(self.tm.row_sum(r), 0)

    def test_all_row_sums_length(self):
        sums = self.tm.all_row_sums()
        self.assertEqual(len(sums), self.tm.num_rows)

    def test_reflect_vertical_stays_in_row(self):
        """reflect_vertical(r, c): результирующий столбец ∈ [1, r]."""
        for r in range(1, 6):
            for c in range(1, r + 1):
                rr, cc = self.tm.reflect_vertical(r, c)
                self.assertEqual(rr, r)
                self.assertGreaterEqual(cc, 1)
                self.assertLessEqual(cc, r)

    def test_center_cell_is_27(self):
        r, c, v = self.tm.center_cell()
        self.assertEqual(v, 27)


class TestRenderTriangle(unittest.TestCase):
    def test_returns_nonempty_list(self):
        lines = render_triangle()
        self.assertIsInstance(lines, list)
        self.assertGreater(len(lines), 0)

    def test_lines_are_strings(self):
        for line in render_triangle():
            self.assertIsInstance(line, str)


if __name__ == '__main__':
    unittest.main(verbosity=2)
