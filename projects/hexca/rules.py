"""
rules.py — библиотека правил перехода для HexCA

Правило задаёт функцию: (текущая гексаграмма, список соседей) → новая гексаграмма.
Новая гексаграмма должна быть соседом в Q6 (изменение ≤ 1 черты) — «плавность».

Встроенные правила:
  majority_vote  — переворачивает черту, если большинство соседей её изменили
  xor_rule       — XOR всех соседей
  conway_like    — порог активации по числу ян-соседей
  identity       — ничего не меняет (стабильный паттерн)
  random_walk    — случайный сосед (для тестов)
"""

from __future__ import annotations
import random
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import neighbors, flip, yang_count, to_bits, SIZE


RuleFn = callable  # (current: int, nbrs: list[int]) -> int


def _clamp(x: int) -> int:
    return max(0, min(SIZE - 1, x))


def majority_vote(current: int, nbrs: list[int]) -> int:
    """
    Для каждой из 6 черт: если большинство соседей имеют другое значение —
    перевернуть эту черту. Принятие решения по большинству.
    """
    result = current
    n = len(nbrs)
    if n == 0:
        return current
    for bit in range(6):
        cur_val = (current >> bit) & 1
        count_other = sum(1 for nb in nbrs if ((nb >> bit) & 1) != cur_val)
        if count_other > n // 2:
            result = flip(result, bit)
    return result


def xor_rule(current: int, nbrs: list[int]) -> int:
    """
    XOR текущего состояния со всеми соседями.
    Результат ограничивается соседом в Q6 (только 1 бит изменяется).
    """
    if not nbrs:
        return current
    xored = current
    for nb in nbrs:
        xored ^= nb
    xored &= 0x3F
    # Найти ближайшего соседа в Q6 к xored (изменить только 1 бит от current)
    diff = current ^ xored
    if diff == 0:
        return current
    # Возьмём самый значимый отличающийся бит
    bit = diff.bit_length() - 1
    return flip(current, bit)


def conway_like(birth: set[int] = frozenset({3}),
                survive: set[int] = frozenset({2, 3})):
    """
    Аналог правила B/S Конвея, адаптированный для Q6.
    Считает число ян-соседей. Клетка «мертва» если yang_count=0.
    """
    def rule(current: int, nbrs: list[int]) -> int:
        alive = yang_count(current) > 0
        yang_nbrs = sum(1 for nb in nbrs if yang_count(nb) > 0)
        if alive and yang_nbrs in survive:
            return current  # выживает
        elif alive and yang_nbrs not in survive:
            # умирает: перевернуть случайную ян-черту в 0
            yang_bits = [b for b in range(6) if (current >> b) & 1]
            if yang_bits:
                return flip(current, yang_bits[0])
            return current
        elif not alive and yang_nbrs in birth:
            # рождается: перевернуть черту 0
            return flip(current, 0)
        return current
    return rule


def identity(current: int, nbrs: list[int]) -> int:
    """Правило-тождество: состояние не меняется."""
    return current


def random_walk(current: int, nbrs: list[int]) -> int:
    """Случайный переход (для тестирования)."""
    nb = neighbors(current)
    return random.choice(nb)


def outer_totalistic(table: dict[tuple[int, int], int]) -> RuleFn:
    """
    Внешний тотализм: новое состояние зависит только от
    (yang_count(current), sum_yang_neighbors).
    table[(c, s)] = новый yang_count. Если ключа нет — состояние не меняется.

    Пример: outer_totalistic({(3,9): 6, (0,18): 1}) —
    клетка с 3 ян при сумме 9 ян у соседей переходит в 6 ян.
    """
    def rule(current: int, nbrs: list[int]) -> int:
        c = yang_count(current)
        s = sum(yang_count(nb) for nb in nbrs)
        new_c = table.get((c, s))
        if new_c is None or new_c == c:
            return current
        # Изменить ровно abs(new_c - c) черт за минимальное число шагов
        # (так как правило Q6 меняет 1 черту за шаг, делаем 1 шаг)
        diff = new_c - c
        if diff > 0:
            # Установить первую инь-черту в ян
            for b in range(6):
                if not ((current >> b) & 1):
                    return flip(current, b)
        else:
            # Сбросить первую ян-черту в инь
            for b in range(6):
                if (current >> b) & 1:
                    return flip(current, b)
        return current
    return rule


def smooth_rule(current: int, nbrs: list[int]) -> int:
    """
    Правило «сглаживания» (дискретный Лапласиан на Q6):
    если среднее ян соседей > ян текущей — добавить одну ян-черту,
    если меньше — убрать одну ян-черту,
    иначе — оставить как есть.
    Приводит к усреднению поля со временем.
    """
    if not nbrs:
        return current
    avg_yang = sum(yang_count(nb) for nb in nbrs) / len(nbrs)
    c = yang_count(current)
    if avg_yang > c + 0.5:
        for b in range(6):
            if not ((current >> b) & 1):
                return flip(current, b)
    elif avg_yang < c - 0.5:
        for b in range(6):
            if (current >> b) & 1:
                return flip(current, b)
    return current


def cyclic_rule(step: int = 1) -> RuleFn:
    """
    Циклическое правило: если хотя бы один сосед имеет ян-счёт
    на step больше (по модулю 7), перейти к следующему ян-уровню.
    Создаёт распространяющиеся волны.
    """
    n_states = 7  # 0..6 ян-черт
    def rule(current: int, nbrs: list[int]) -> int:
        c = yang_count(current)
        next_c = (c + step) % n_states
        for nb in nbrs:
            if yang_count(nb) == next_c:
                # Перейти к next_c ян
                diff = next_c - c
                if diff > 0:
                    for b in range(6):
                        if not ((current >> b) & 1):
                            return flip(current, b)
                elif diff < 0:
                    for b in range(6):
                        if (current >> b) & 1:
                            return flip(current, b)
        return current
    return rule


# Реестр правил по имени
RULES: dict[str, RuleFn] = {
    'majority_vote': majority_vote,
    'xor_rule': xor_rule,
    'conway_b3s23': conway_like(birth={3}, survive={2, 3}),
    'conway_b36s23': conway_like(birth={3, 6}, survive={2, 3}),
    'identity': identity,
    'random_walk': random_walk,
    'smooth': smooth_rule,
    'cyclic': cyclic_rule(step=1),
    'cyclic2': cyclic_rule(step=2),
}


def get_rule(name: str) -> RuleFn:
    if name not in RULES:
        raise ValueError(f"Неизвестное правило: '{name}'. Доступны: {list(RULES)}")
    return RULES[name]
