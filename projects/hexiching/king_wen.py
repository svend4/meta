"""
king_wen.py — Последовательность Вэнь-вана как граф Q6

Анализирует 64 гексаграммы традиционного порядка (王序)
как путь в Q6. Задаёт вопросы:

  - Какое расстояние Хэмминга между соседними гексаграммами?
  - Существует ли скрытая структура «прыжков»?
  - Как соотносится KW-порядок с кодом Грея?
  - Каков «граф Вэнь-вана»: рёбра Q6 vs. длинные дуги?

НАБЛЮДЕНИЯ:
  Расстояние между соседними гексаграммами в KW-порядке
  варьируется от 1 до 6, в среднем около 2.6.

  Код Грея: стандартная последовательность всех 64 состояний,
  где каждый шаг меняет ровно 1 бит. Это Гамильтонов цикл Q6.

  «Грей-И-цзин»: порядок гексаграмм по коду Грея показывает,
  какая «физическая» последовательность изменений соответствует
  стандартному обходу Q6.

ПАРЫ Вэнь-вана:
  Порядок организован парами: KW1↔KW2, KW3↔KW4, ..., KW63↔KW64.
  Пары образованы либо инверсией черт (переворот всей гексаграммы),
  либо антиподом в Q6 (63-h).
"""

import sys
import os
import argparse
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, antipode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
from hexiching import Hexagram, KW_DATA, _KW_TO_H, KW_FROM_H

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

DIST_COLORS = {
    1: "\033[92m",   # зелёный  — 1 шаг (Q6-ребро)
    2: "\033[93m",   # жёлтый  — 2 шага
    3: "\033[32m",   # тёмно-зелёный — 3 шага
    4: "\033[33m",   # оранжевый — 4 шага
    5: "\033[35m",   # фиолет — 5 шагов
    6: "\033[31m",   # красный — 6 шагов (диаметр)
}


# ---------------------------------------------------------------------------
# Анализ расстояний
# ---------------------------------------------------------------------------

def kw_distances() -> list:
    """Список (kw1, kw2, h1, h2, dist) для всех соседних пар KW."""
    result = []
    for i in range(63):
        h1 = _KW_TO_H[i]
        h2 = _KW_TO_H[i + 1]
        result.append((i+1, i+2, h1, h2, hamming(h1, h2)))
    return result


def distance_histogram(use_color: bool = True) -> str:
    """Гистограмма расстояний Хэмминга в KW-последовательности."""
    dists = kw_distances()
    counts = {}
    for *_, d in dists:
        counts[d] = counts.get(d, 0) + 1

    total = len(dists)
    avg = sum(d for *_, d in dists) / total
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    lines = [
        f"{bold}Расстояния Хэмминга в последовательности Вэнь-вана{r}",
        f"(63 перехода между соседними гексаграммами):",
        "",
        f"  {'Расст.':>7}  {'Кол-во':>6}  {'%':>5}  Гистограмма",
        "  " + "─" * 50,
    ]
    for d in sorted(counts):
        n = counts[d]
        pct = n / total * 100
        bar_len = int(n * 30 / total + 0.5)
        dc = DIST_COLORS.get(d, "") if use_color else ""
        lines.append(
            f"  {dc}{d:>7}  {n:>6}  {pct:>4.1f}%  {'█' * bar_len}{r}"
        )
    lines += [
        "  " + "─" * 50,
        f"  Среднее: {avg:.3f}",
        f"  Число Q6-рёбер (d=1): {counts.get(1,0)} из 63",
        f"  Максимальное: {max(counts)}  (диаметр Q6 = 6)",
    ]
    return "\n".join(lines)


def kw_pair_analysis(use_color: bool = True) -> str:
    """Анализ 32 пар Вэнь-вана: переворот vs антипод."""
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    lines = [
        f"{bold}32 пары Вэнь-вана: тип трансформации{r}",
        "",
        f"  {'#':>3}  {'H1':>3}  {'H2':>3}  {'d':>2}  {'тип':>10}  Гексаграммы",
        "  " + "─" * 65,
    ]

    reversals = antipodes = other = 0
    for i in range(0, 64, 2):
        h1 = _KW_TO_H[i]
        h2 = _KW_TO_H[i+1]
        d = hamming(h1, h2)
        kw1, kw2 = i+1, i+2
        hx1 = Hexagram(h1)
        hx2 = Hexagram(h2)

        # Тип: переворот (все черты) или антипод (63-h)?
        is_antipode  = (h2 == 63 - h1)
        # Визуальный переворот: перевернуть порядок черт
        rev_h = int(f"{h1:06b}"[::-1], 2)
        is_reversal = (h2 == rev_h) and not is_antipode

        if is_antipode:
            pair_type = "антипод"
            dc = "\033[34m" if use_color else ""
            antipodes += 1
        elif is_reversal:
            pair_type = "переворот"
            dc = "\033[33m" if use_color else ""
            reversals += 1
        else:
            pair_type = "иное"
            dc = "\033[31m" if use_color else ""
            other += 1

        lines.append(
            f"  {kw1:>3}  {h1:>3}  {h2:>3}  {d:>2}  "
            f"{dc}{pair_type:>10}{r}  "
            f"{hx1.sym}«{hx1.name_pin[:8]}» ↔ {hx2.sym}«{hx2.name_pin[:8]}»"
        )

    lines += [
        "  " + "─" * 65,
        f"  Антиподы (63-h):   {antipodes}",
        f"  Перевороты (зерк.): {reversals}",
        f"  Иные:               {other}",
        "",
        "  «Антипод» = инверсия всех 6 черт (полное превращение)",
        "  «Переворот» = гексаграмма читается снизу вверх",
        "  Если h1==h2 по одному из правил — пара «родственных» гексаграмм.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Код Грея для Q6
# ---------------------------------------------------------------------------

def standard_gray_code(n_bits: int = 6) -> list:
    """Стандартный бинарный отражённый код Грея для n бит."""
    if n_bits == 1:
        return [0, 1]
    prev = standard_gray_code(n_bits - 1)
    return prev + [x | (1 << (n_bits - 1)) for x in reversed(prev)]


def gray_as_iching(use_color: bool = True) -> str:
    """Стандартный код Грея (64 шага) интерпретированный как И-цзин."""
    gray = standard_gray_code(6)
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    lines = [
        f"{bold}Код Грея как 64 гексаграммы (Гамильтонов цикл Q6){r}",
        f"(каждый шаг = 1 меняющаяся черта)",
        "",
        f"  {'#':>3}  {'h':>3}  {'бин':>8}  {'ян':>3}  {'сим':>3}  {'KW':>4}  Название",
        "  " + "─" * 60,
    ]
    yang_colors = {
        0:"\033[90m",1:"\033[34m",2:"\033[36m",
        3:"\033[32m",4:"\033[33m",5:"\033[35m",6:"\033[37m",
    }
    for i, h in enumerate(gray):
        hx = Hexagram(h)
        yc = yang_colors[hx.yang] if use_color else ""
        # Показываем изменившийся бит
        if i > 0:
            diff_bit = (gray[i-1] ^ h).bit_length() - 1
            changed = f"бит{diff_bit}"
        else:
            changed = "    "
        lines.append(
            f"  {yc}{i+1:>3}  {h:>3}  {h:08b}  {hx.yang:>3}  "
            f"{hx.sym:>3}  {hx.kw:>4}  {hx.name_pin[:18]:18s}  {changed}{r}"
        )
    # Замыкание цикла
    d_close = hamming(gray[-1], gray[0])
    lines += [
        "  " + "─" * 60,
        f"  Замыкание: h={gray[-1]}→h={gray[0]}  d={d_close}"
        f"  {'(цикл!)' if d_close==1 else '(НЕ цикл)'}",
        "",
        "  Стандартный код Грея образует Гамильтонов ЦИКЛ в Q6.",
        "  Порядок Вэнь-вана — НЕ Гамильтонов путь (расст.≠1 между парами).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# «Серый код» Вэнь-вана: ближайший GC-порядок
# ---------------------------------------------------------------------------

def kw_vs_gray(use_color: bool = True, show_all: bool = False) -> str:
    """Сравнение порядка Вэнь-вана с кодом Грея."""
    gray = standard_gray_code(6)
    gray_pos = {h: i for i, h in enumerate(gray)}
    kw_seq = _KW_TO_H

    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    lines = [
        f"{bold}KW-порядок vs Код Грея{r}",
        "",
    ]

    # Для каждой гексаграммы KW: позиция в коде Грея
    kw_gray_diffs = []
    for i, h in enumerate(kw_seq):
        gray_i = gray_pos[h]
        kw_gray_diffs.append(abs(gray_i - i))

    avg_diff = sum(kw_gray_diffs) / len(kw_gray_diffs)
    max_diff = max(kw_gray_diffs)
    in_order = sum(1 for d in kw_gray_diffs if d <= 3)

    lines += [
        f"  Среднее отклонение KW от Грея: {avg_diff:.1f} позиций",
        f"  Максимальное отклонение:        {max_diff} позиций",
        f"  Гексаграмм в пределах ±3 от Грея: {in_order}/64",
        "",
        "  Наибольшие несовпадения:",
        f"  {'KW#':>4}  {'h':>3}  {'Грей#':>6}  {'отклон.':>8}  Название",
        "  " + "─" * 55,
    ]
    # Сортируем по отклонению
    sorted_pairs = sorted(
        [(i+1, kw_seq[i], gray_pos[kw_seq[i]], kw_gray_diffs[i])
         for i in range(64)],
        key=lambda x: -x[3]
    )
    limit = 64 if show_all else 12
    for kw, h, g, d in sorted_pairs[:limit]:
        hx = Hexagram(h)
        dc = DIST_COLORS.get(min(d // 5 + 1, 6), "") if use_color else ""
        lines.append(
            f"  {dc}{kw:>4}  {h:>3}  {g+1:>6}  {d:>8}  {hx.name_pin[:20]:20s} {hx.sym}{r}"
        )

    lines += [
        "",
        "  ВЫВОД: Порядок Вэнь-вана отличается от Грея — это",
        "  смысловое, а не математическое упорядочение гексаграмм.",
        "  Однако обе системы используют одно и то же Q6-пространство.",
    ]
    return "\n".join(lines)


def kw_subgraph(use_color: bool = True) -> str:
    """
    Подграф Q6, образованный рёбрами KW-последовательности.
    Показывает, сколько рёбер Q6 (d=1) реализовано в KW.
    """
    dists = kw_distances()
    edges_in_q6 = [(h1, h2) for _, _, h1, h2, d in dists if d == 1]
    total_edges_q6 = 64 * 6 // 2   # рёбра Q6: 192

    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    lines = [
        f"{bold}Подграф Q6 в последовательности Вэнь-вана{r}",
        "",
        f"  Всего переходов в KW: 63",
        f"  Из них — рёбра Q6 (d=1): {len(edges_in_q6)}",
        f"  Всего рёбер Q6: {total_edges_q6}",
        f"  Покрытие: {len(edges_in_q6)/total_edges_q6*100:.1f}%",
        "",
        f"  Q6-рёбра в KW (переходы d=1):",
    ]
    for kw1, kw2, h1, h2, d in dists:
        if d == 1:
            gc = "\033[92m" if use_color else ""
            bit = (h1 ^ h2).bit_length() - 1
            hx1 = Hexagram(h1)
            hx2 = Hexagram(h2)
            lines.append(
                f"    {gc}KW#{kw1:>2}→KW#{kw2:>2}"
                f"  h={h1:>2}→{h2:>2}  бит{bit}"
                f"  {hx1.sym}«{hx1.name_pin[:8]}»→{hx2.sym}«{hx2.name_pin[:8]}»{r}"
            )
    return "\n".join(lines)


def yang_level_flow(use_color: bool = True) -> str:
    """
    Поток ян-уровней в KW-последовательности.
    Показывает как yang_count меняется от гексаграммы к гексаграмме.
    """
    yang_seq = [yang_count(_KW_TO_H[i]) for i in range(64)]
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    yang_colors = {
        0:"\033[90m",1:"\033[34m",2:"\033[36m",
        3:"\033[32m",4:"\033[33m",5:"\033[35m",6:"\033[37m",
    }

    lines = [
        f"{bold}Поток ян-уровней в последовательности Вэнь-вана{r}",
        "(yang=0: всё инь, yang=6: всё ян, yang=3: равновесие)",
        "",
        "  KW#  1..16:  " +
        " ".join(
            (yang_colors[yang_seq[i]] if use_color else "") +
            str(yang_seq[i]) + (r if use_color else "")
            for i in range(16)
        ),
        "  KW# 17..32:  " +
        " ".join(
            (yang_colors[yang_seq[i]] if use_color else "") +
            str(yang_seq[i]) + (r if use_color else "")
            for i in range(16, 32)
        ),
        "  KW# 33..48:  " +
        " ".join(
            (yang_colors[yang_seq[i]] if use_color else "") +
            str(yang_seq[i]) + (r if use_color else "")
            for i in range(32, 48)
        ),
        "  KW# 49..64:  " +
        " ".join(
            (yang_colors[yang_seq[i]] if use_color else "") +
            str(yang_seq[i]) + (r if use_color else "")
            for i in range(48, 64)
        ),
        "",
    ]

    # Считаем переходы между уровнями
    transitions = {}
    for i in range(63):
        a, b = yang_seq[i], yang_seq[i+1]
        key = (a, b)
        transitions[key] = transitions.get(key, 0) + 1

    lines.append("  Матрица переходов ян-уровней (строка→столбец):")
    lines.append("      " + "  ".join(f"{j}" for j in range(7)))
    for a in range(7):
        row = f"  {a}:  "
        for b in range(7):
            n = transitions.get((a,b), 0)
            if n:
                yc = yang_colors[b] if use_color else ""
                row += f"{yc}{n:>2}{r} "
            else:
                row += "   "
        lines.append(row)

    lines += [
        "",
        f"  Среднее ян в KW-последовательности: "
        f"{sum(yang_seq)/len(yang_seq):.2f}",
        f"  (Ожидаемое для равновесного Q6: 3.0)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="king_wen — Последовательность Вэнь-вана как Q6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python king_wen.py               Гистограмма расстояний
  python king_wen.py --pairs       32 пары: антиподы vs переворот
  python king_wen.py --gray        Код Грея (64 гексаграммы)
  python king_wen.py --compare     KW vs Грей: отклонения
  python king_wen.py --subgraph    Q6-рёбра в KW-последовательности
  python king_wen.py --yang        Поток ян-уровней
        """,
    )
    parser.add_argument("--pairs",    action="store_true")
    parser.add_argument("--gray",     action="store_true")
    parser.add_argument("--compare",  action="store_true")
    parser.add_argument("--subgraph", action="store_true")
    parser.add_argument("--yang",     action="store_true")
    parser.add_argument("--all",      action="store_true",
                        help="Показать все строки (вместо первых 12)")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.pairs:
        print("\n" + kw_pair_analysis(use_color))
    elif args.gray:
        print("\n" + gray_as_iching(use_color))
    elif args.compare:
        print("\n" + kw_vs_gray(use_color, show_all=args.all))
    elif args.subgraph:
        print("\n" + kw_subgraph(use_color))
    elif args.yang:
        print("\n" + yang_level_flow(use_color))
    else:
        print("\n" + distance_histogram(use_color))


if __name__ == "__main__":
    main()
