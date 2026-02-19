"""
hexlearn — машинное обучение на пространстве Q6

Q6 = (Z₂)⁶ со метрикой Хэмминга образует дискретное метрическое пространство.
Многие задачи ML допускают постановку на этом пространстве без вещественных координат.

Реализованные модели:
  KNN             — k ближайших соседей по Хэммингу
  KMedoids        — k-медоидов (аналог k-means в метрическом пространстве)
  SpectralEmbed   — спектральное вложение в R^d через лапласиан Q6
  MarkovChain     — случайное блуждание / Марковская цепь на Q6
  HammingBayes    — наивный Байес (по битам)

Все модели работают только с Python stdlib (без numpy/scikit-learn).
"""

from __future__ import annotations
import sys
import random
import math
from collections import defaultdict, Counter
from typing import Callable

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import (
    neighbors, hamming, yang_count, to_bits, SIZE,
    sphere, ball, all_hexagrams,
)


# ---------------------------------------------------------------------------
# Типы
# ---------------------------------------------------------------------------

Label = int | str
Dataset = list[tuple[int, Label]]   # (hexagram, label)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def hamming_distance_matrix(points: list[int]) -> list[list[int]]:
    """Матрица попарных расстояний Хэмминга."""
    n = len(points)
    return [[hamming(points[i], points[j]) for j in range(n)] for i in range(n)]


def medoid(points: list[int]) -> int:
    """Медоид: точка с минимальной суммой расстояний до остальных."""
    if not points:
        raise ValueError("Пустое множество")
    return min(points, key=lambda p: sum(hamming(p, q) for q in points))


def dispersion(points: list[int]) -> float:
    """Среднее попарное расстояние в множестве."""
    n = len(points)
    if n < 2:
        return 0.0
    total = sum(hamming(points[i], points[j])
                for i in range(n) for j in range(i + 1, n))
    return total / (n * (n - 1) / 2)


def centroid_hex(points: list[int]) -> int:
    """
    Гексаграмма, ближайшая к «центроиду» набора (минимизирует суммарное расстояние).
    Эквивалентно медоиду; в Q6 по каждому биту берём большинство.
    """
    if not points:
        raise ValueError("Пустое множество")
    k = len(points)
    result = 0
    for bit in range(6):
        count_one = sum(1 for p in points if (p >> bit) & 1)
        if count_one > k // 2:
            result |= (1 << bit)
    return result


# ---------------------------------------------------------------------------
# k-NN классификатор
# ---------------------------------------------------------------------------

class KNN:
    """
    k ближайших соседей по расстоянию Хэмминга.

    Параметры:
        k : число соседей (нечётное рекомендуется для бинарной классификации)
        weighted : если True — веса обратно пропорциональны расстоянию
    """

    def __init__(self, k: int = 1, weighted: bool = False) -> None:
        self.k = k
        self.weighted = weighted
        self._data: Dataset = []

    def fit(self, data: Dataset) -> 'KNN':
        """Запомнить обучающую выборку."""
        self._data = list(data)
        return self

    def predict(self, query: int) -> Label:
        """Предсказать метку для query."""
        if not self._data:
            raise RuntimeError("Модель не обучена (fit() не вызван)")
        # Отсортировать по расстоянию
        neighbors_sorted = sorted(self._data, key=lambda item: hamming(query, item[0]))
        top_k = neighbors_sorted[:self.k]

        if self.weighted:
            votes: dict[Label, float] = defaultdict(float)
            for h, label in top_k:
                d = hamming(query, h)
                weight = 1.0 / (d + 1)
                votes[label] += weight
        else:
            votes = Counter(label for _, label in top_k)

        return max(votes, key=lambda lbl: votes[lbl])

    def predict_proba(self, query: int) -> dict[Label, float]:
        """Вероятности меток (нормированные веса)."""
        if not self._data:
            raise RuntimeError("Модель не обучена")
        neighbors_sorted = sorted(self._data, key=lambda item: hamming(query, item[0]))
        top_k = neighbors_sorted[:self.k]
        votes: dict[Label, float] = defaultdict(float)
        for h, label in top_k:
            d = hamming(query, h)
            weight = 1.0 / (d + 1) if self.weighted else 1.0
            votes[label] += weight
        total = sum(votes.values())
        return {lbl: w / total for lbl, w in votes.items()}

    def score(self, test: Dataset) -> float:
        """Точность на тестовой выборке."""
        correct = sum(1 for h, label in test if self.predict(h) == label)
        return correct / len(test) if test else 0.0

    def cross_validate(self, data: Dataset, folds: int = 5,
                       rng: random.Random | None = None) -> float:
        """k-fold кросс-валидация."""
        rng = rng or random.Random(42)
        shuffled = list(data)
        rng.shuffle(shuffled)
        fold_size = len(shuffled) // folds
        scores = []
        for i in range(folds):
            test = shuffled[i * fold_size:(i + 1) * fold_size]
            train = shuffled[:i * fold_size] + shuffled[(i + 1) * fold_size:]
            self.fit(train)
            scores.append(self.score(test))
        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# k-медоиды кластеризация
# ---------------------------------------------------------------------------

class KMedoids:
    """
    Кластеризация k-медоидами в метрическом пространстве Q6.

    Алгоритм PAM (Partitioning Around Medoids):
    1. Случайно инициализировать k медоидов
    2. Присвоить каждую точку ближайшему медоиду
    3. Для каждого кластера найти новый медоид (минимизирует суммарное расстояние)
    4. Повторять до сходимости
    """

    def __init__(self, k: int, max_iter: int = 100,
                 n_init: int = 5, seed: int | None = None) -> None:
        self.k = k
        self.max_iter = max_iter
        self.n_init = n_init
        self.rng = random.Random(seed)
        self.medoids_: list[int] = []
        self.labels_: list[int] = []
        self.inertia_: float = float('inf')

    def fit(self, points: list[int]) -> 'KMedoids':
        """Кластеризовать points."""
        if len(points) < self.k:
            raise ValueError(f"Точек ({len(points)}) меньше чем k={self.k}")

        best_medoids = None
        best_inertia = float('inf')

        for _ in range(self.n_init):
            medoids = self.rng.sample(points, self.k)
            medoids, labels, inertia = self._run(points, medoids)
            if inertia < best_inertia:
                best_inertia = inertia
                best_medoids = medoids[:]
                best_labels = labels[:]

        self.medoids_ = best_medoids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def _run(self, points: list[int],
             init_medoids: list[int]) -> tuple[list[int], list[int], float]:
        medoids = init_medoids[:]
        for _ in range(self.max_iter):
            # Присвоить точки кластерам
            labels = [
                min(range(self.k), key=lambda j: hamming(p, medoids[j]))
                for p in points
            ]
            # Обновить медоиды
            new_medoids = []
            for j in range(self.k):
                cluster = [points[i] for i, lbl in enumerate(labels) if lbl == j]
                if cluster:
                    new_medoids.append(medoid(cluster))
                else:
                    new_medoids.append(medoids[j])
            if new_medoids == medoids:
                break
            medoids = new_medoids

        inertia = sum(hamming(points[i], medoids[labels[i]])
                      for i in range(len(points)))
        return medoids, labels, float(inertia)

    def predict(self, h: int) -> int:
        """Предсказать кластер для новой точки."""
        return min(range(self.k), key=lambda j: hamming(h, self.medoids_[j]))

    def cluster_stats(self, points: list[int]) -> list[dict]:
        """Статистика кластеров."""
        stats = []
        for j in range(self.k):
            cluster = [points[i] for i, lbl in enumerate(self.labels_) if lbl == j]
            stats.append({
                'id': j,
                'medoid': self.medoids_[j],
                'size': len(cluster),
                'dispersion': round(dispersion(cluster), 4),
                'mean_yang': round(sum(yang_count(h) for h in cluster) / len(cluster), 2)
                if cluster else 0.0,
            })
        return stats

    def silhouette_score(self, points: list[int]) -> float:
        """
        Силуэтный коэффициент (среднее по точкам):
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
        где a(i) — среднее расстояние внутри кластера,
            b(i) — среднее расстояние до ближайшего соседнего кластера.
        """
        if self.k == 1:
            return 0.0
        n = len(points)
        scores = []
        for idx, p in enumerate(points):
            my_cluster = self.labels_[idx]
            # a(i): среднее до точек своего кластера
            same = [points[j] for j, lbl in enumerate(self.labels_)
                    if lbl == my_cluster and j != idx]
            if not same:
                scores.append(0.0)
                continue
            a = sum(hamming(p, q) for q in same) / len(same)

            # b(i): min среднее до других кластеров
            b = float('inf')
            for other_cluster in range(self.k):
                if other_cluster == my_cluster:
                    continue
                other = [points[j] for j, lbl in enumerate(self.labels_)
                         if lbl == other_cluster]
                if other:
                    avg = sum(hamming(p, q) for q in other) / len(other)
                    b = min(b, avg)

            s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
            scores.append(s)
        return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Спектральное вложение
# ---------------------------------------------------------------------------

def _mat_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
    n = len(vec)
    return [sum(mat[i][j] * vec[j] for j in range(n)) for i in range(n)]


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 1e-12 else vec


def spectral_embed(
    points: list[int],
    dim: int = 2,
    sigma: float | None = None,
) -> list[tuple[float, ...]]:
    """
    Спектральное вложение: points → R^dim.

    Строит матрицу аффинностей W[i][j] = exp(-d(i,j)^2 / sigma),
    лапласиан L = D - W, и берёт dim наименьших ненулевых собственных векторов.

    Без numpy: степенной метод для наибольшего и deflation.

    Возвращает список dim-мерных координат для каждой точки.
    """
    n = len(points)
    if n == 0:
        return []
    if sigma is None:
        # Автоматически выбрать sigma как медиана расстояний
        dists = [hamming(points[i], points[j])
                 for i in range(n) for j in range(i + 1, n)]
        dists.sort()
        sigma = dists[len(dists) // 2] or 1.0

    # Матрица аффинностей
    W = [[math.exp(-(hamming(points[i], points[j]) ** 2) / sigma)
          for j in range(n)] for i in range(n)]

    # Степени
    D = [sum(W[i]) for i in range(n)]

    # Нормализованный лапласиан L_sym = I - D^{-1/2} W D^{-1/2}
    D_inv_sqrt = [1.0 / math.sqrt(d) if d > 1e-12 else 0.0 for d in D]
    L = [[
        (1.0 if i == j else 0.0) - D_inv_sqrt[i] * W[i][j] * D_inv_sqrt[j]
        for j in range(n)
    ] for i in range(n)]

    # Наибольшие собственные значения I - L = D^{-1/2} W D^{-1/2}
    # → наименьшие L через deflation
    I_minus_L = [[
        (1.0 if i == j else 0.0) - L[i][j]
        for j in range(n)
    ] for i in range(n)]

    eigvecs = []
    current = [row[:] for row in I_minus_L]

    for _ in range(dim + 1):   # +1 чтобы пропустить тривиальный вектор
        vec = _normalize([1.0 / n] * n)
        lam = 0.0
        for _ in range(300):
            new_vec = _mat_vec(current, vec)
            norm = math.sqrt(sum(x * x for x in new_vec))
            if norm < 1e-12:
                break
            new_vec = [x / norm for x in new_vec]
            lam_new = sum(_mat_vec(current, new_vec)[i] * new_vec[i] for i in range(n))
            if abs(lam_new - lam) < 1e-10:
                lam = lam_new
                vec = new_vec
                break
            lam = lam_new
            vec = new_vec

        eigvecs.append((lam, vec[:]))
        # Deflate
        for i in range(n):
            for j in range(n):
                current[i][j] -= lam * vec[i] * vec[j]

    # Пропустить тривиальный (наибольший ~ 1) вектор
    eigvecs.sort(key=lambda x: -x[0])
    selected = eigvecs[1:dim + 1]   # dim наибольших нетривиальных

    # Координаты: i-я строка = (v1[i], v2[i], ...)
    coords = []
    for idx in range(n):
        coord = tuple(ev[idx] for _, ev in selected)
        coords.append(coord)
    return coords


# ---------------------------------------------------------------------------
# Марковская цепь (случайное блуждание на Q6)
# ---------------------------------------------------------------------------

class MarkovChain:
    """
    Марковская цепь на Q6 с заданными вероятностями переходов.

    По умолчанию — равномерное случайное блуждание по Q6 (каждый сосед = 1/6).
    Можно задать произвольные веса переходов.
    """

    def __init__(
        self,
        transition_weights: Callable[[int, int], float] | None = None,
    ) -> None:
        """
        transition_weights(u, v) → вес перехода u→v (для соседей v ∈ nbrs(u)).
        None → равномерное блуждание.
        """
        self._weights = transition_weights

    def _transition_probs(self, h: int) -> list[tuple[int, float]]:
        """Вернуть список (сосед, вероятность) для состояния h."""
        nbrs = neighbors(h)
        if self._weights is None:
            p = 1.0 / len(nbrs)
            return [(nb, p) for nb in nbrs]
        raw = [(nb, self._weights(h, nb)) for nb in nbrs]
        total = sum(w for _, w in raw)
        if total <= 0:
            p = 1.0 / len(nbrs)
            return [(nb, p) for nb in nbrs]
        return [(nb, w / total) for nb, w in raw]

    def step(self, h: int, rng: random.Random | None = None) -> int:
        """Один шаг цепи: выбрать следующее состояние."""
        rng = rng or random
        probs = self._transition_probs(h)
        r = rng.random()
        cumulative = 0.0
        for nb, p in probs:
            cumulative += p
            if r < cumulative:
                return nb
        return probs[-1][0]

    def simulate(
        self,
        start: int,
        steps: int,
        seed: int | None = None,
    ) -> list[int]:
        """Симулировать траекторию цепи."""
        rng = random.Random(seed)
        path = [start]
        current = start
        for _ in range(steps):
            current = self.step(current, rng)
            path.append(current)
        return path

    def stationary_distribution(self, max_iter: int = 1000) -> list[float]:
        """
        Стационарное распределение: итерация вектора вероятностей.
        Для равномерного блуждания на Q6 = равномерное 1/64.
        """
        # Матрица переходов P[i][j]
        states = list(range(SIZE))
        n = SIZE
        P = [[0.0] * n for _ in range(n)]
        for h in states:
            for nb, p in self._transition_probs(h):
                P[h][nb] += p

        # Начальное распределение
        dist = [1.0 / n] * n

        for _ in range(max_iter):
            # dist_new[j] = sum_i dist[i] * P[i][j]
            new_dist = [0.0] * n
            for i in range(n):
                for j in range(n):
                    new_dist[j] += dist[i] * P[i][j]
            # Проверить сходимость
            diff = sum(abs(new_dist[j] - dist[j]) for j in range(n))
            dist = new_dist
            if diff < 1e-10:
                break

        return dist

    def mixing_time(
        self,
        start: int = 0,
        eps: float = 0.25,
        max_steps: int = 200,
    ) -> int:
        """
        Приближённое время перемешивания: число шагов до ||P^t(start,·) - π||_TV < eps.
        """
        stat = self.stationary_distribution()
        n = SIZE

        # Итерационное умножение распределения
        dist = [0.0] * n
        dist[start] = 1.0

        for t in range(1, max_steps + 1):
            # dist_new[j] = sum_i dist[i] * P[i][j]
            new_dist = [0.0] * n
            for h in range(n):
                if dist[h] == 0:
                    continue
                for nb, p in self._transition_probs(h):
                    new_dist[nb] += dist[h] * p
            dist = new_dist
            tv = 0.5 * sum(abs(dist[j] - stat[j]) for j in range(n))
            if tv < eps:
                return t

        return max_steps

    def hitting_time_empirical(
        self,
        start: int,
        target: int,
        n_trials: int = 1000,
        seed: int | None = None,
    ) -> float:
        """
        Эмпирическое среднее время первого попадания в target, стартуя из start.
        """
        rng = random.Random(seed)
        total = 0
        for _ in range(n_trials):
            current = start
            steps = 0
            while current != target:
                current = self.step(current, rng)
                steps += 1
                if steps > 10 * SIZE:   # предохранитель
                    break
            total += steps
        return total / n_trials


# ---------------------------------------------------------------------------
# Наивный байесовский классификатор
# ---------------------------------------------------------------------------

class HammingBayes:
    """
    Наивный байесовский классификатор для гексаграмм.
    Каждый бит условно независим при данном классе.

    P(h | class) = Π_{bit} P(bit_i | class)
    """

    def __init__(self, laplace: float = 1.0) -> None:
        """laplace — сглаживание Лапласа."""
        self.laplace = laplace
        self._class_counts: Counter = Counter()
        self._bit_counts: dict[Label, list[int]] = {}   # label → [count_1 per bit]
        self._total: int = 0

    def fit(self, data: Dataset) -> 'HammingBayes':
        self._class_counts = Counter()
        self._bit_counts = {}
        self._total = 0

        for h, label in data:
            self._class_counts[label] += 1
            if label not in self._bit_counts:
                self._bit_counts[label] = [0] * 6
            for bit in range(6):
                if (h >> bit) & 1:
                    self._bit_counts[label][bit] += 1
            self._total += 1

        return self

    def _log_prob(self, h: int, label: Label) -> float:
        """Log P(h | label) = Σ log P(bit_i | label)."""
        n_class = self._class_counts[label]
        counts = self._bit_counts.get(label, [0] * 6)
        log_p = 0.0
        for bit in range(6):
            bit_val = (h >> bit) & 1
            # Laplace smoothing
            p_one = (counts[bit] + self.laplace) / (n_class + 2 * self.laplace)
            log_p += math.log(p_one if bit_val else (1 - p_one))
        return log_p

    def predict(self, h: int) -> Label:
        """Предсказать метку."""
        if not self._class_counts:
            raise RuntimeError("Модель не обучена")
        return max(
            self._class_counts,
            key=lambda lbl: (
                math.log(self._class_counts[lbl] / self._total) +
                self._log_prob(h, lbl)
            )
        )

    def score(self, test: Dataset) -> float:
        correct = sum(1 for h, label in test if self.predict(h) == label)
        return correct / len(test) if test else 0.0


# ---------------------------------------------------------------------------
# Утилиты для генерации датасетов
# ---------------------------------------------------------------------------

def yang_labeled_dataset() -> Dataset:
    """Датасет: все 64 гексаграммы, метка = yang_count."""
    return [(h, yang_count(h)) for h in range(SIZE)]


def binary_yang_dataset(threshold: int = 3) -> Dataset:
    """Датасет: метка 'high' если yang > threshold, иначе 'low'."""
    return [(h, 'high' if yang_count(h) > threshold else 'low') for h in range(SIZE)]


def random_dataset(
    n: int,
    n_classes: int = 2,
    seed: int | None = None,
) -> Dataset:
    """Случайный датасет: n случайных (hexagram, label) пар."""
    rng = random.Random(seed)
    labels = list(range(n_classes))
    return [(rng.randrange(SIZE), rng.choice(labels)) for _ in range(n)]


def cluster_dataset(
    centers: list[int],
    radius: int = 2,
    seed: int | None = None,
) -> Dataset:
    """
    Датасет с кластерами вокруг заданных центров.
    Каждая точка помечена номером ближайшего центра.
    """
    rng = random.Random(seed)
    data = []
    for label, center in enumerate(centers):
        cluster_pts = ball(center, radius)
        for pt in cluster_pts:
            data.append((pt, label))
    rng.shuffle(data)
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='hexlearn — ML на Q6')
    sub = parser.add_subparsers(dest='cmd')

    p_knn = sub.add_parser('knn', help='k-NN классификация (yang_count)')
    p_knn.add_argument('--k', type=int, default=3)
    p_knn.add_argument('--query', type=int, default=42)
    p_knn.add_argument('--folds', type=int, default=5)

    p_km = sub.add_parser('kmeans', help='k-медоиды кластеризация')
    p_km.add_argument('--k', type=int, default=4)
    p_km.add_argument('--seed', type=int, default=42)

    p_mc = sub.add_parser('markov', help='Марковская цепь (случайное блуждание Q6)')
    p_mc.add_argument('--start', type=int, default=0)
    p_mc.add_argument('--steps', type=int, default=20)
    p_mc.add_argument('--seed', type=int, default=42)
    p_mc.add_argument('--mixing', action='store_true', help='Вычислить время перемешивания')

    p_embed = sub.add_parser('embed', help='Спектральное вложение в R²')
    p_embed.add_argument('--points', nargs='*', type=int,
                          help='Гексаграммы (по умолч. — все 64)')
    p_embed.add_argument('--dim', type=int, default=2)

    p_bayes = sub.add_parser('bayes', help='Наивный байес')
    p_bayes.add_argument('--query', type=int, default=42)
    p_bayes.add_argument('--threshold', type=int, default=3)

    args = parser.parse_args()

    if args.cmd == 'knn':
        # Кластерный датасет: 4 кластера вокруг центров {0, 21, 42, 63}
        centers = [0, 21, 42, 63]
        data = cluster_dataset(centers, radius=2, seed=42)
        rng = random.Random(42)
        shuffled = list(data)
        rng.shuffle(shuffled)
        split = int(len(shuffled) * 0.8)
        train, test = shuffled[:split], shuffled[split:]

        knn = KNN(k=args.k)
        knn.fit(train)
        pred = knn.predict(args.query)
        true_label = min(range(len(centers)), key=lambda j: hamming(args.query, centers[j]))
        acc = knn.score(test)
        print(f"Запрос: {args.query}  Предсказан кластер={pred}  (истинный={true_label})")
        print(f"Точность на тестовой выборке: {acc:.3f}")
        cv_score = knn.cross_validate(data, folds=args.folds)
        print(f"Кросс-валидация ({args.folds}-fold): {cv_score:.3f}")

    elif args.cmd == 'kmeans':
        points = list(range(SIZE))
        km = KMedoids(k=args.k, seed=args.seed)
        km.fit(points)
        print(f"k-медоиды (k={args.k}):")
        for stat in km.cluster_stats(points):
            med = stat['medoid']
            print(f"  Кластер {stat['id']}: медоид={med} ({yang_count(med)} ян), "
                  f"размер={stat['size']}, дисперсия={stat['dispersion']}, "
                  f"avg_yang={stat['mean_yang']}")
        sil = km.silhouette_score(points)
        print(f"Силуэт: {sil:.4f}")

    elif args.cmd == 'markov':
        mc = MarkovChain()
        path = mc.simulate(args.start, args.steps, seed=args.seed)
        print(f"Случайное блуждание из {args.start}, {args.steps} шагов:")
        for i, h in enumerate(path):
            print(f"  [{i:3d}] {h:2d}  {yang_count(h)} ян")
        if args.mixing:
            t_mix = mc.mixing_time()
            print(f"\nВремя перемешивания (eps=0.25): {t_mix} шагов")

    elif args.cmd == 'embed':
        pts = args.points if args.points else list(range(SIZE))
        coords = spectral_embed(pts, dim=args.dim)
        print(f"Спектральное вложение {len(pts)} точек в R^{args.dim}:")
        for h, c in zip(pts, coords):
            coord_str = '  '.join(f"{x:+.4f}" for x in c)
            print(f"  {h:2d} ({yang_count(h)} ян): [{coord_str}]")

    elif args.cmd == 'bayes':
        data = binary_yang_dataset(threshold=args.threshold)
        rng = random.Random(42)
        shuffled = list(data)
        rng.shuffle(shuffled)
        split = int(len(shuffled) * 0.8)
        train, test = shuffled[:split], shuffled[split:]

        clf = HammingBayes()
        clf.fit(train)
        pred = clf.predict(args.query)
        true = 'high' if yang_count(args.query) > args.threshold else 'low'
        acc = clf.score(test)
        print(f"Запрос: {args.query}  Предсказано={pred}  Истинное={true}")
        print(f"Точность: {acc:.3f}")

    else:
        parser.print_help()
