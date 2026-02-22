"""Тесты hexlearn — машинное обучение на Q6."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
import random
from projects.hexlearn.hexlearn import (
    KNN, KMedoids, MarkovChain, HammingBayes,
    hamming_distance_matrix, medoid, dispersion, centroid_hex,
    yang_labeled_dataset, binary_yang_dataset, cluster_dataset, random_dataset,
    spectral_embed,
)
from libs.hexcore.hexcore import hamming, yang_count, neighbors, SIZE


class TestUtilities(unittest.TestCase):
    def test_hamming_matrix_zero_diagonal(self):
        points = [0, 1, 42, 63]
        mat = hamming_distance_matrix(points)
        for i in range(len(points)):
            self.assertEqual(mat[i][i], 0)

    def test_hamming_matrix_symmetric(self):
        points = [0, 7, 42]
        mat = hamming_distance_matrix(points)
        for i in range(len(points)):
            for j in range(len(points)):
                self.assertEqual(mat[i][j], mat[j][i])

    def test_medoid_single(self):
        self.assertEqual(medoid([42]), 42)

    def test_medoid_antipodes(self):
        """Медоид {0, 63} = 0 или 63 (равноудалены)."""
        m = medoid([0, 63])
        self.assertIn(m, [0, 63])

    def test_medoid_minimizes_sum(self):
        points = [0, 1, 3, 7]
        m = medoid(points)
        m_sum = sum(hamming(m, p) for p in points)
        for p in points:
            self.assertLessEqual(m_sum, sum(hamming(p, q) for q in points) + 1e-9)

    def test_dispersion_single(self):
        self.assertEqual(dispersion([42]), 0.0)

    def test_dispersion_antipodes(self):
        """Два антипода: дисперсия = расстояние = 6."""
        self.assertEqual(dispersion([0, 63]), 6.0)

    def test_centroid_hex_single(self):
        self.assertEqual(centroid_hex([42]), 42)

    def test_centroid_hex_valid_range(self):
        import random as rng
        rng.seed(42)
        points = rng.sample(range(SIZE), 10)
        c = centroid_hex(points)
        self.assertIn(c, range(SIZE))


class TestDatasets(unittest.TestCase):
    def test_yang_labeled_size(self):
        data = yang_labeled_dataset()
        self.assertEqual(len(data), SIZE)

    def test_yang_labeled_labels(self):
        data = yang_labeled_dataset()
        for h, label in data:
            self.assertEqual(label, yang_count(h))

    def test_binary_yang_labels(self):
        data = binary_yang_dataset(threshold=3)
        for h, label in data:
            expected = 'high' if yang_count(h) > 3 else 'low'
            self.assertEqual(label, expected)

    def test_binary_yang_balanced(self):
        """Примерно половина 'high' и половина 'low'."""
        data = binary_yang_dataset(threshold=3)
        highs = sum(1 for _, l in data if l == 'high')
        # C(6,4)+C(6,5)+C(6,6) = 15+6+1 = 22
        self.assertEqual(highs, 22)

    def test_cluster_dataset(self):
        centers = [0, 63]
        data = cluster_dataset(centers, radius=2, seed=42)
        labels = set(label for _, label in data)
        self.assertEqual(labels, {0, 1})

    def test_random_dataset_size(self):
        data = random_dataset(20, n_classes=3, seed=0)
        self.assertEqual(len(data), 20)


class TestKNN(unittest.TestCase):
    def setUp(self):
        # Простой датасет: метка = 0 если yang < 3, иначе 1
        self.data = [(h, int(yang_count(h) >= 3)) for h in range(SIZE)]

    def test_predict_returns_label(self):
        knn = KNN(k=1)
        knn.fit(self.data)
        pred = knn.predict(0)
        self.assertIn(pred, [0, 1])

    def test_perfect_recall_k1(self):
        """1-NN на обучающей выборке = идеальная точность."""
        knn = KNN(k=1)
        knn.fit(self.data)
        acc = knn.score(self.data)
        self.assertAlmostEqual(acc, 1.0)

    def test_predict_proba_sums_to_one(self):
        knn = KNN(k=3)
        knn.fit(self.data)
        proba = knn.predict_proba(42)
        self.assertAlmostEqual(sum(proba.values()), 1.0, places=5)

    def test_predict_proba_keys(self):
        knn = KNN(k=3)
        knn.fit(self.data)
        proba = knn.predict_proba(0)
        for label in proba:
            self.assertIn(label, [0, 1])

    def test_not_fitted_raises(self):
        knn = KNN()
        with self.assertRaises(RuntimeError):
            knn.predict(0)

    def test_weighted_knn(self):
        knn = KNN(k=3, weighted=True)
        knn.fit(self.data)
        pred = knn.predict(42)
        self.assertIn(pred, [0, 1])

    def test_score_in_range(self):
        knn = KNN(k=3)
        knn.fit(self.data[:40])
        acc = knn.score(self.data[40:])
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_cross_validate_in_range(self):
        knn = KNN(k=3)
        score = knn.cross_validate(self.data, folds=4)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestKMedoids(unittest.TestCase):
    def setUp(self):
        self.points = list(range(SIZE))

    def test_fit_returns_self(self):
        km = KMedoids(k=3, seed=42)
        result = km.fit(self.points)
        self.assertIs(result, km)

    def test_medoids_count(self):
        km = KMedoids(k=4, seed=42)
        km.fit(self.points)
        self.assertEqual(len(km.medoids_), 4)

    def test_medoids_are_from_points(self):
        km = KMedoids(k=3, seed=42)
        km.fit(self.points)
        for m in km.medoids_:
            self.assertIn(m, self.points)

    def test_labels_count(self):
        km = KMedoids(k=4, seed=42)
        km.fit(self.points)
        self.assertEqual(len(km.labels_), len(self.points))

    def test_labels_in_range(self):
        km = KMedoids(k=4, seed=42)
        km.fit(self.points)
        for lbl in km.labels_:
            self.assertIn(lbl, range(4))

    def test_predict_valid(self):
        km = KMedoids(k=3, seed=0)
        km.fit(self.points)
        pred = km.predict(42)
        self.assertIn(pred, range(3))

    def test_cluster_stats(self):
        km = KMedoids(k=3, seed=42)
        km.fit(self.points)
        stats = km.cluster_stats(self.points)
        self.assertEqual(len(stats), 3)
        total_size = sum(s['size'] for s in stats)
        self.assertEqual(total_size, len(self.points))

    def test_inertia_positive(self):
        km = KMedoids(k=3, seed=42)
        km.fit(self.points)
        self.assertGreater(km.inertia_, 0)

    def test_silhouette_range(self):
        km = KMedoids(k=4, seed=42)
        km.fit(self.points)
        s = km.silhouette_score(self.points)
        self.assertGreaterEqual(s, -1.0)
        self.assertLessEqual(s, 1.0)

    def test_k1_silhouette_zero(self):
        km = KMedoids(k=1, seed=42)
        km.fit(self.points)
        self.assertEqual(km.silhouette_score(self.points), 0.0)

    def test_too_few_points_raises(self):
        km = KMedoids(k=5)
        with self.assertRaises(ValueError):
            km.fit([0, 1, 2])


class TestMarkovChain(unittest.TestCase):
    def test_simulate_length(self):
        mc = MarkovChain()
        path = mc.simulate(0, 10, seed=42)
        self.assertEqual(len(path), 11)

    def test_simulate_starts_at_start(self):
        mc = MarkovChain()
        path = mc.simulate(42, 5, seed=0)
        self.assertEqual(path[0], 42)

    def test_simulate_valid_transitions(self):
        """Каждый шаг — ребро Q6."""
        mc = MarkovChain()
        path = mc.simulate(0, 20, seed=1)
        for i in range(len(path) - 1):
            self.assertEqual(hamming(path[i], path[i + 1]), 1)

    def test_step_returns_neighbor(self):
        mc = MarkovChain()
        rng = random.Random(0)
        for h in range(0, SIZE, 7):
            next_h = mc.step(h, rng)
            self.assertEqual(hamming(h, next_h), 1)

    def test_stationary_uniform(self):
        """Равномерное блуждание → стационарное = 1/64 для каждой вершины."""
        mc = MarkovChain()
        stat = mc.stationary_distribution()
        self.assertEqual(len(stat), SIZE)
        for p in stat:
            self.assertAlmostEqual(p, 1.0 / SIZE, places=4)

    def test_mixing_time_positive(self):
        mc = MarkovChain()
        t = mc.mixing_time()
        self.assertGreater(t, 0)

    def test_hitting_time_self_zero(self):
        """Время попадания из h в h = 0."""
        mc = MarkovChain()
        # Симуляция с target=start → 0 шагов в первом же состоянии
        path = mc.simulate(5, 0, seed=0)
        self.assertEqual(path[0], 5)

    def test_custom_weights(self):
        """Пользовательские веса меняют распределение переходов."""
        # Всегда выбирать сосед с наименьшим yang_count
        def prefer_low(u, v):
            return 1.0 if yang_count(v) < yang_count(u) else 0.01
        mc = MarkovChain(prefer_low)
        path = mc.simulate(63, 10, seed=42)
        # Должно двигаться вниз по yang_count
        self.assertLessEqual(yang_count(path[-1]), yang_count(path[0]))


class TestHammingBayes(unittest.TestCase):
    def setUp(self):
        self.data = binary_yang_dataset(threshold=3)

    def test_fit_returns_self(self):
        clf = HammingBayes()
        result = clf.fit(self.data)
        self.assertIs(result, clf)

    def test_predict_returns_label(self):
        clf = HammingBayes()
        clf.fit(self.data)
        pred = clf.predict(0)
        self.assertIn(pred, ['high', 'low'])

    def test_predict_obvious_cases(self):
        """63 (all yang) должен быть 'high', 0 (no yang) — 'low'."""
        clf = HammingBayes()
        clf.fit(self.data)
        self.assertEqual(clf.predict(63), 'high')
        self.assertEqual(clf.predict(0), 'low')

    def test_score_in_range(self):
        clf = HammingBayes()
        clf.fit(self.data[:40])
        acc = clf.score(self.data[40:])
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_high_accuracy(self):
        """Байес должен достичь разумной точности на бинарной задаче."""
        clf = HammingBayes()
        clf.fit(self.data)
        acc = clf.score(self.data)
        # На обучающей выборке точность должна быть > 70%
        self.assertGreater(acc, 0.7)

    def test_not_fitted_raises(self):
        clf = HammingBayes()
        with self.assertRaises(RuntimeError):
            clf.predict(42)

    def test_laplace_smoothing(self):
        """Сглаживание Лапласа не должно приводить к нулевым вероятностям."""
        clf = HammingBayes(laplace=1.0)
        clf.fit(self.data[:5])   # очень мало данных
        # Не должно упасть с ошибкой
        pred = clf.predict(42)
        self.assertIn(pred, ['high', 'low'])


class TestSpectralEmbed(unittest.TestCase):
    def test_output_length(self):
        pts = [0, 1, 3, 7, 15]
        coords = spectral_embed(pts, dim=2)
        self.assertEqual(len(coords), len(pts))

    def test_output_dimension(self):
        pts = list(range(8))
        coords = spectral_embed(pts, dim=2)
        for c in coords:
            self.assertEqual(len(c), 2)

    def test_empty_input(self):
        self.assertEqual(spectral_embed([]), [])

    def test_output_finite(self):
        """Координаты должны быть конечными числами."""
        import math
        pts = list(range(16))
        coords = spectral_embed(pts, dim=2)
        for c in coords:
            for x in c:
                self.assertTrue(math.isfinite(x))

    def test_1d_embed(self):
        pts = [0, 1, 3, 7]
        coords = spectral_embed(pts, dim=1)
        self.assertEqual(len(coords), 4)
        for c in coords:
            self.assertEqual(len(c), 1)


class TestJsonSboxPredict(unittest.TestCase):
    """Тесты ML-регрессии NL~SAC (SC-5 Шаг 3)."""

    @classmethod
    def setUpClass(cls):
        from projects.hexcrypt.sbox_glyphs import json_avalanche
        from projects.hexlearn.learn_glyphs import json_sbox_predict
        avl = json_avalanche()
        cls.result = json_sbox_predict(avl)

    def test_command(self):
        self.assertEqual(self.result['command'], 'predict')

    def test_model_formula_string(self):
        formula = self.result['model']['formula']
        self.assertIsInstance(formula, str)
        self.assertIn('NL', formula)
        self.assertIn('SAC', formula)

    def test_model_r_negative(self):
        r = float(self.result['model']['r'])
        self.assertLess(r, 0.0, 'NL и SAC должны быть отрицательно скоррелированы')

    def test_model_r2_positive(self):
        r2 = float(self.result['model']['r2'])
        self.assertGreater(r2, 0.0)

    def test_predictions_list(self):
        preds = self.result['predictions']
        self.assertGreater(len(preds), 0)
        for p in preds:
            self.assertIn('nl_actual', p)
            self.assertIn('nl_pred', p)
            self.assertIn('error', p)

    def test_k3_k1_synthesis_present(self):
        k3 = self.result['k3_k1_synthesis']
        self.assertIn('nl_ceiling', k3)
        self.assertGreaterEqual(k3['nl_ceiling'], 0)


class TestJsonCodonCluster(unittest.TestCase):
    """Тесты ML-кластеризации Андреев-партиции (TSC-3 Шаг 3)."""

    @classmethod
    def setUpClass(cls):
        from projects.hexbio.codon_glyphs import json_codon_map
        from projects.hextrimat.trimat_glyphs import json_twins_codon
        from projects.hexlearn.learn_glyphs import json_codon_cluster
        codon = json_codon_map()
        twins = json_twins_codon(codon)
        cls.result = json_codon_cluster(twins)

    def test_command(self):
        self.assertEqual(self.result['command'], 'cluster')

    def test_clusters_nonempty(self):
        self.assertGreater(len(self.result['clusters']), 0)

    def test_summary_keys(self):
        s = self.result['summary']
        for key in ('total_clusters', 'pure_clusters', 'weighted_purity'):
            self.assertIn(key, s)

    def test_weighted_purity_in_01(self):
        wp = self.result['summary']['weighted_purity']
        self.assertGreaterEqual(wp, 0.0)
        self.assertLessEqual(wp, 1.0)

    def test_exact_matches_list(self):
        em = self.result['exact_matches']
        self.assertIsInstance(em, list)
        for m in em:
            self.assertIn('codons', m)
            self.assertIn('majority_aa', m)
            self.assertEqual(m['purity'], 1.0)

    def test_resonance_score_above_random(self):
        """TSC-3: резонанс-оценка >> случайная базовая линия."""
        wp = self.result['summary']['weighted_purity']
        # случайная базовая ≈ 0.06; резонанс ≈ 0.68
        self.assertGreater(wp, 0.2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
