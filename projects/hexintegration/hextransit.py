"""
hextransit.py — Анализ перехода между двумя гексаграммами

В И Цзин бросок монет или палочек даёт:
  h1 = «настоящая» (本卦 běn guà) — ситуация сейчас
  h2 = «будущая»  (之卦 zhī guà) — ситуация в движении

Переход h1 → h2 задаётся XOR-маской: diff = h1 ⊕ h2.
Биты разности = «движущиеся черты» (動爻 dòng yáo, линии 1-6).

Модуль предоставляет:
  • Список движущихся черт и их позиции (1=нижняя, 6=верхняя)
  • Расстояние Хэмминга (сложность перехода)
  • Значение каждой движущейся черты в И Цзин, теле (Касаткин), боевой зоне (Крюков)
  • Все возможные маршруты через промежуточные гексаграммы (кратчайший путь)
  • Ядерные гексаграммы h1 и h2 (из hexnuclear)
  • Принадлежность к бассейнам притяжения

Движущиеся черты (классика И Цзин):
  Линия 1: основание — исток, скрытая сила, корни
  Линия 2: внутреннее — ресурсы, соответствие, центр нижней триграммы
  Линия 3: порог — кризис, переход, вершина нижнего
  Линия 4: контакт — приближение, осторожность, основание верхней триграммы
  Линия 5: вершина — зрелость, власть, центр верхней триграммы
  Линия 6: предел — завершение, выход за рамки, новый цикл

Биты тела (Касаткин):
  Бит 0: Стопы / заземление
  Бит 1: Поясница / течение
  Бит 2: Живот / центр огня
  Бит 3: Грудь / сердечный ветер
  Бит 4: Плечи / опора-гора
  Бит 5: Голова / небесное сознание
"""

import sys
import os
import argparse
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../libs/hexcore"))
from hexcore import yang_count, hamming

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
from hexiching import Hexagram

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
YANG_ANSI = {
    0: "\033[90m", 1: "\033[94m", 2: "\033[96m",
    3: "\033[92m", 4: "\033[93m", 5: "\033[95m", 6: "\033[97m",
}
GRN = "\033[92m"
YLW = "\033[93m"
RED = "\033[91m"
CYN = "\033[96m"
MAG = "\033[95m"

HX = {h: Hexagram(h) for h in range(64)}

# ---------------------------------------------------------------------------
# Значения движущихся черт
# ---------------------------------------------------------------------------

LINE_ICHING = [
    "Основание (линия 1): исток, скрытая сила, корни",
    "Внутренний центр (линия 2): ресурсы, развитие, соответствие",
    "Порог (линия 3): кризис, переход, граница нижней/верхней",
    "Внешний контакт (линия 4): приближение, новый контекст, осторожность",
    "Вершина силы (линия 5): зрелость, власть, центр верхней триграммы",
    "Предел (линия 6): завершение, выход за рамки, начало нового цикла",
]

LINE_BODY = [
    "Стопы / Земля — заземление, укоренение",
    "Поясница / Вода — тонус, течение, запас",
    "Живот / Огонь — центр, импульс, горение",
    "Грудь / Ветер — сердечный простор, направление",
    "Плечи / Гора — опора, распределение усилия",
    "Голова / Небо — намерение, осознанность, горизонт",
]

LINE_DIRECTION = [
    "Вниз (Земля) — приземление, осевое давление",
    "Назад/Вперёд (Вода) — сагиттальный вектор",
    "Внутрь/Наружу (Огонь) — дыхательный центр",
    "Горизонталь (Ветер) — раскрытие, охват",
    "Вверх (Гора) — вертикальное усилие, подъём",
    "Сферически (Небо) — охват всего объёма",
]

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def nuclear(h: int) -> int:
    """Ядерная гексаграмма (互卦)."""
    return ((h >> 1) & 7) | (((h >> 2) & 7) << 3)


def changing_bits(h1: int, h2: int) -> list[int]:
    """Список битовых позиций, где h1 и h2 различаются (0=снизу)."""
    diff = h1 ^ h2
    return [i for i in range(6) if (diff >> i) & 1]


def all_shortest_paths(h1: int, h2: int) -> list[list[int]]:
    """Все кратчайшие пути от h1 до h2 через последовательный флип битов."""
    bits  = changing_bits(h1, h2)
    paths = []
    seen  = set()
    for perm in permutations(bits):
        cur = h1
        path = [cur]
        for b in perm:
            cur ^= (1 << b)
            path.append(cur)
        key = tuple(path[1:-1])  # промежуточные гексаграммы
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return paths


def basin(h: int) -> str:
    b2, b3 = (h >> 2) & 1, (h >> 3) & 1
    if b2 == b3 == 0: return "Кунь"
    if b2 == b3 == 1: return "Цянь"
    return "Цзи↔Вэй"


# ---------------------------------------------------------------------------
# Основное чтение
# ---------------------------------------------------------------------------

def show_transit(h1: int, h2: int, use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    hx1 = HX[h1]
    hx2 = HX[h2]
    d   = hamming(h1, h2)
    cbits = changing_bits(h1, h2)

    def hx_str(h: int, label: str = "") -> str:
        hx = HX[h]
        yc = YANG_ANSI[hx.yang] if use_color else ""
        l  = f"  {dim}«{label}»{r}" if label else ""
        return (
            f"{yc}{hx.sym} КВ#{hx.kw:>2} «{hx.name_cn} {hx.name_pin}»"
            f"  ян={hx.yang}  h={h:>2} ({h:06b}){r}{l}"
        )

    lines = [
        "",
        "═"*66,
        f"  {bold}АНАЛИЗ ПЕРЕХОДА: 本卦 → 之卦{r}",
        "═"*66,
        "",
        f"  {bold}Настоящее (本卦 běn guà):{r}",
        f"  {hx_str(h1)}",
        "",
        f"  {bold}Будущее   (之卦 zhī guà):{r}",
        f"  {hx_str(h2)}",
        "",
    ]

    if h1 == h2:
        lines += [
            f"  {DIM if use_color else ''}Гексаграммы идентичны. Расстояние = 0. Переход отсутствует.{r}",
        ]
        return "\n".join(lines)

    # ── XOR маска и расстояние ────────────────────────────────────────────
    diff = h1 ^ h2
    yc_d = YANG_ANSI[yang_count(diff)] if use_color else ""

    lines += [
        "─"*66,
        f"  {bold}Маска перехода (h1 ⊕ h2):{r}",
        f"  {yc_d}{HX[diff].sym} diff={diff:06b}  Расстояние Хэмминга: {d}{r}",
        "",
    ]

    # ── Движущиеся черты ──────────────────────────────────────────────────
    lines += [
        f"  {bold}Движущиеся черты ({d} из 6):{r}",
        "",
    ]

    for b in sorted(cbits):
        line_n = b + 1
        was    = (h1 >> b) & 1
        to     = (h2 >> b) & 1
        yc     = (GRN if use_color else "") if to == 1 else (DIM if use_color else "")
        arrow  = f"{'инь→ян' if was==0 else 'ян→инь'}"
        lines += [
            f"  {bold}Линия {line_n}:{r}  {yc}{'━' if was==1 else '╌ ╌'}  →  {'━' if to==1 else '╌ ╌'}  ({arrow}){r}",
            f"  {dim}  И Цзин: {LINE_ICHING[b]}{r}",
            f"  {dim}  Тело:   {LINE_BODY[b]}{r}",
            "",
        ]

    # ── Маршруты ─────────────────────────────────────────────────────────
    paths = all_shortest_paths(h1, h2)
    lines += [
        "─"*66,
        f"  {bold}Кратчайшие маршруты ({len(paths)} путей длиной {d}):{r}",
        "",
    ]

    # Показываем не более 6 маршрутов
    shown = paths[:6]
    for p_idx, path in enumerate(shown):
        path_str = " → ".join(
            f"{HX[v].sym}{'*' if v in (h1, h2) else ''}" for v in path
        )
        lines.append(f"  {DIM if use_color else ''}Путь {p_idx+1}:{r}  {path_str}")

    if len(paths) > 6:
        lines.append(f"  {DIM if use_color else ''}...ещё {len(paths)-6} путей{r}")

    lines.append("")

    # Детальный разбор «каноничного» пути (снизу вверх)
    lines += [
        f"  {bold}Канонический путь (линии снизу вверх):{r}",
    ]
    canon = [h1]
    cur   = h1
    for b in sorted(cbits):
        cur ^= (1 << b)
        canon.append(cur)

    for i, v in enumerate(canon):
        hxv  = HX[v]
        yv   = YANG_ANSI[hxv.yang] if use_color else ""
        if i == 0:
            label = "↓ исходная"
        elif i == len(canon) - 1:
            label = "✦ итог"
        else:
            b = sorted(cbits)[i-1]
            label = f"линия {b+1} перевёрнута"
        lines.append(
            f"  {yv}  {hxv.sym} КВ#{hxv.kw:>2} «{hxv.name_pin}»{r}"
            f"  {dim}({label}){r}"
        )

    lines.append("")

    # ── Ядерные гексаграммы ───────────────────────────────────────────────
    n1   = nuclear(h1)
    n2   = nuclear(h2)
    hxn1 = HX[n1]
    hxn2 = HX[n2]

    lines += [
        "─"*66,
        f"  {bold}Ядерные гексаграммы (互卦 hù guà):{r}",
        "",
        f"  N(настоящее):  {YANG_ANSI[hxn1.yang] if use_color else ''}{hxn1.sym} КВ#{hxn1.kw:>2} «{hxn1.name_pin}»{r}",
        f"  N(будущее):    {YANG_ANSI[hxn2.yang] if use_color else ''}{hxn2.sym} КВ#{hxn2.kw:>2} «{hxn2.name_pin}»{r}",
    ]

    n_changed = n1 != n2
    if n_changed:
        nd = hamming(n1, n2)
        lines.append(
            f"  {dim}Ядра различаются (расстояние {nd}) — переход затрагивает скрытый слой{r}"
        )
    else:
        lines.append(f"  {dim}Ядра совпадают — скрытый слой стабилен{r}")

    # ── Бассейны притяжения ───────────────────────────────────────────────
    b1, b2_str = basin(h1), basin(h2)
    lines += [
        "",
        f"  {bold}Бассейны ядерного притяжения:{r}",
        f"  Настоящее: {b1}",
        f"  Будущее:   {b2_str}",
    ]
    if b1 == b2_str:
        lines.append(f"  {dim}→ Обе гексаграммы в одном бассейне{r}")
    else:
        lines.append(f"  {bold}→ Переход МЕЖДУ бассейнами! Смена судьбы.{r}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Карта трансформации: от h ко всем соседям
# ---------------------------------------------------------------------------

def show_neighbors(h: int, use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    hx  = HX[h]
    yc  = YANG_ANSI[hx.yang] if use_color else ""

    lines = [
        "",
        f"  {bold}Соседи гексаграммы:{r}",
        f"  {yc}{hx.sym} КВ#{hx.kw} «{hx.name_pin}»  h={h} ({h:06b})  ян={hx.yang}{r}",
        "",
        f"  {dim}{'Линия':>7}  {'Переход':>10}  {'Гексаграмма':<30}  ян  {'Бассейн'}{r}",
        "  " + "─"*58,
    ]

    for b in range(6):
        nb    = h ^ (1 << b)
        hxnb  = HX[nb]
        was   = (h >> b) & 1
        to    = (nb >> b) & 1
        arrow = f"{'инь→ян' if was==0 else 'ян→инь'}"
        yc2   = YANG_ANSI[hxnb.yang] if use_color else ""
        lines.append(
            f"  Линия {b+1}: {arrow:>10}  "
            f"{yc2}{hxnb.sym} КВ#{hxnb.kw:>2} «{hxnb.name_pin}»{r:<20}  "
            f"{hxnb.yang}  {basin(nb)}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hextransit — анализ перехода между гексаграммами",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hextransit.py --from 7 --to 21        Тай → Цзи Цзи
  python hextransit.py --from 0 --to 63        Кунь → Цянь (6 черт)
  python hextransit.py --from 63 --to 0        Цянь → Кунь
  python hextransit.py --neighbors 11           Все соседи Тай
  python hextransit.py --from 21 --to 42        Цзи Цзи ↔ Вэй Цзи (2-цикл)
        """,
    )
    parser.add_argument("--from",      dest="h1", type=int, metavar="H")
    parser.add_argument("--to",        dest="h2", type=int, metavar="H")
    parser.add_argument("--neighbors", dest="nb", type=int, metavar="H")
    parser.add_argument("--no-color",  action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.nb is not None:
        if not 0 <= args.nb <= 63:
            print("Ошибка: h должно быть 0..63")
            return
        print(show_neighbors(args.nb, use_color))
    elif args.h1 is not None and args.h2 is not None:
        if not (0 <= args.h1 <= 63 and 0 <= args.h2 <= 63):
            print("Ошибка: h должно быть 0..63")
            return
        print(show_transit(args.h1, args.h2, use_color))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
