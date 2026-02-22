"""
hexsymm.py — Группа симметрий Q6 и пары Вэнь-вана

Три фундаментальных преобразования Q6:

  comp(h)     = h XOR 63          «Дополнение» — инверсия всех линий
  rev6(h)     = реверс 6 бит      «Обращение» — чтение снизу вверх
  comp_rev(h) = comp(rev6(h))     Их суперпозиция

Эти три преобразования + тождественное образуют группу Клейна Z2×Z2:
  { id, comp, rev6, comp_rev }

Четыре орбитальных типа:
  1. Палиндромы    (rev6(h) = h):              8 гексаграмм → 4 пары
  2. Антипалиндр. (rev6(h) = comp(h)):         8 гексаграмм → 4 пары
  3. Обычные       (все 4 значения различны):  48 гексаграмм → 12 орбит по 4

32 пары Вэнь-вана (64 гексаграммы = 32 пары):
  • Палиндромы: нельзя обратить (обращение даёт себя же) →
    парятся с дополнением: (h, comp(h))  ← принцип «错» cuò
  • Антипалиндромы: обращение = дополнение → те же 8 пар
  • Обычные 48: парятся с обращением: (h, rev6(h))  ← принцип «综» zōng

Итог: COMP=8 пар, REV=24 пары, OTHER=0 пар. Совершенный порядок.

Связь с ДНК:
  comp(h)  ↔ комплементарная нить ДНК (A↔T, G↔C)
  rev6(h)  ↔ обратная последовательность (3'→5' прочтение)
  comp_rev ↔ обратно-комплементарная нить (OC-strand, реальная матрица!)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../libs/hexcore"))
from hexcore import yang_count, hamming

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
from hexiching import Hexagram

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RED   = "\033[91m"
GRN   = "\033[92m"
YLW   = "\033[93m"
BLU   = "\033[94m"
MAG   = "\033[95m"
CYN   = "\033[96m"

YANG_ANSI = {
    0: "\033[90m", 1: "\033[94m", 2: "\033[96m",
    3: "\033[92m", 4: "\033[93m", 5: "\033[95m", 6: "\033[97m",
}


# ---------------------------------------------------------------------------
# Преобразования
# ---------------------------------------------------------------------------

def comp(h: int) -> int:
    """Дополнение: инверсия всех 6 линий."""
    return h ^ 63


def rev6(h: int) -> int:
    """Обращение: реверс 6 бит (чтение гексаграммы снизу вверх)."""
    return int(f"{h:06b}"[::-1], 2)


def comp_rev(h: int) -> int:
    """Суперпозиция: comp(rev6(h)) = rev6(comp(h))."""
    return comp(rev6(h))


def orbit(h: int) -> list[int]:
    """Орбита h под группой {id, comp, rev6, comp_rev}."""
    seen = set()
    result = []
    for fn in (lambda x: x, comp, rev6, comp_rev):
        v = fn(h)
        if v not in seen:
            seen.add(v)
            result.append(v)
    return sorted(result)


def classify(h: int) -> str:
    """Тип орбиты: 'palindrome', 'antipalindrome', 'regular'."""
    r = rev6(h)
    if r == h:
        return "palindrome"
    if r == comp(h):
        return "antipalindrome"
    return "regular"


# ---------------------------------------------------------------------------
# Предвычисления
# ---------------------------------------------------------------------------

KW_TO_H = {Hexagram(h).kw: h for h in range(64)}
H_TO_HX = {h: Hexagram(h) for h in range(64)}

PALINDROMES     = sorted([h for h in range(64) if classify(h) == "palindrome"])
ANTIPALINDROMES = sorted([h for h in range(64) if classify(h) == "antipalindrome"])
REGULARS        = sorted([h for h in range(64) if classify(h) == "regular"])


# ---------------------------------------------------------------------------
# Показать группу Клейна
# ---------------------------------------------------------------------------

def show_group(use_color: bool = True) -> str:
    c = lambda s, col: col + s + RESET if use_color else s
    lines = [
        "",
        c(f"{'═'*68}", BOLD),
        c("  ГРУППА СИММЕТРИЙ Q6: Z2 × Z2 (ГРУППА КЛЕЙНА)", BOLD),
        c(f"{'═'*68}", BOLD),
        "",
        "  Четыре элемента и их действие на h (6-битовый номер гексаграммы):",
        "",
        c("  ┌──────────────┬──────────────────────┬─────────────────────────────┐", DIM),
        c("  │ Элемент      │ Формула              │ Смысл в И Цзин              │", DIM),
        c("  ├──────────────┼──────────────────────┼─────────────────────────────┤", DIM),
        "  │ id           │ h → h                │ Та же гексаграмма           │",
        f"  │ comp         │ h → h XOR 63         │ Инверсия всех линий (错 cuò)│",
        "  │ rev6         │ h → реверс 6 бит     │ Чтение снизу вверх (综 zōng)│",
        f"  │ comp∘rev6    │ h → comp(rev6(h))    │ Суперпозиция двух операций  │",
        c("  └──────────────┴──────────────────────┴─────────────────────────────┘", DIM),
        "",
        "  Таблица Кэли (суперпозиция):",
        "",
        c("  ┌────────────┬────────────┬────────────┬────────────┐", DIM),
        c("  │ ∘          │ id         │ comp       │ rev6       │ comp∘rev   │", DIM),
        c("  ├────────────┼────────────┼────────────┼────────────┤", DIM),
        "  │ id         │ id         │ comp       │ rev6       │ comp∘rev   │",
        "  │ comp       │ comp       │ id         │ comp∘rev   │ rev6       │",
        "  │ rev6       │ rev6       │ comp∘rev   │ id         │ comp       │",
        "  │ comp∘rev   │ comp∘rev   │ rev6       │ comp       │ id         │",
        c("  └────────────┴────────────┴────────────┴────────────┘", DIM),
        "",
        "  Каждый элемент является своим обратным (все имеют порядок 2).",
        "  Группа изоморфна Z2 × Z2 (=V4, четвёртая группа Клейна).",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Фиксированные точки и орбиты
# ---------------------------------------------------------------------------

def show_fixed_points(use_color: bool = True) -> str:
    c = lambda s, col: col + s + RESET if use_color else s
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""

    lines = [
        f"{bold}Фиксированные точки и орбиты{r}",
        "",
        f"  {bold}1. Палиндромы — фиксированные точки rev6 (rev6(h) = h):{r}",
        f"  {DIM if use_color else ''}  Гексаграмма симметрична относительно центра: биты 0↔5, 1↔4, 2↔3{r}",
        "",
    ]

    for h in PALINDROMES:
        hx = H_TO_HX[h]
        ch = comp(h)
        hxc = H_TO_HX[ch]
        yc = YANG_ANSI[hx.yang] if use_color else ""
        lines.append(
            f"    {yc}h={h:>2} ({h:06b}){r}  {hx.sym}#{hx.kw:>2} «{hx.name_pin[:14]:<14}»"
            f"  ↔comp  h={ch:>2} ({ch:06b}) {hxc.sym}#{hxc.kw:>2} «{hxc.name_pin[:14]}»"
        )

    lines += [
        "",
        f"  {bold}2. Антипалиндромы — rev6(h) = comp(h):{r}",
        f"  {DIM if use_color else ''}  Обращение = дополнение; принцип 综 совпадает с 错{r}",
        "",
    ]

    for h in ANTIPALINDROMES:
        hx  = H_TO_HX[h]
        r6h = rev6(h)
        hxr = H_TO_HX[r6h]
        yc  = YANG_ANSI[hx.yang] if use_color else ""
        lines.append(
            f"    {yc}h={h:>2} ({h:06b}){r}  {hx.sym}#{hx.kw:>2} «{hx.name_pin[:14]:<14}»"
            f"  ↔rev=comp  h={r6h:>2} ({r6h:06b}) {hxr.sym}#{hxr.kw:>2} «{hxr.name_pin[:14]}»"
        )

    # Орбиты обычных гексаграмм (размер 4)
    lines += [
        "",
        f"  {bold}3. Обычные гексаграммы — орбиты размера 4:{r}",
        f"  {DIM if use_color else ''}  {{h, rev6(h), comp(h), comp∘rev6(h)}} — все четыре различны{r}",
        f"  {DIM if use_color else ''}  Итого: {len(REGULARS)} гексаграмм = {len(REGULARS)//4} орбит по 4{r}",
        "",
    ]

    seen = set()
    orbit_num = 0
    for h in REGULARS:
        if h in seen:
            continue
        orb  = orbit(h)
        seen.update(orb)
        orbit_num += 1
        syms   = "".join(H_TO_HX[v].sym for v in orb)
        h_strs = "  ".join(f"h={v:2d}({v:06b})" for v in orb)
        lines.append(f"    О{orbit_num:>2}: {syms}  {h_strs}")

    lines += [
        "",
        f"  {bold}Сводка орбит:{r}",
        f"    Палиндромы:    {len(PALINDROMES):>2} гексаграмм → {len(PALINDROMES)//2} пары",
        f"    Антипалиндр.: {len(ANTIPALINDROMES):>2} гексаграмм → {len(ANTIPALINDROMES)//2} пары",
        f"    Обычные:       {len(REGULARS):>2} гексаграмм → {len(REGULARS)//4} орбит по 4",
        f"    Итого орбит: {len(PALINDROMES)//2 + len(ANTIPALINDROMES)//2 + len(REGULARS)//4}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Пары Вэнь-вана
# ---------------------------------------------------------------------------

def show_kw_pairs(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    comp_pairs = []
    rev_pairs  = []

    for k in range(1, 33):
        h1 = KW_TO_H[2*k - 1]
        h2 = KW_TO_H[2*k]
        cl = classify(h1)
        if rev6(h1) == h2:
            rev_pairs.append((h1, h2, "REV"))
        else:
            comp_pairs.append((h1, h2, "COMP"))

    lines = [
        f"{bold}32 пары Вэнь-вана через группу Z2×Z2{r}",
        "",
        f"  {dim}КВ-пара  тип   h1   ←операция→  h2   гексаграммы{r}",
        "  " + "─"*62,
    ]

    for k in range(1, 33):
        h1 = KW_TO_H[2*k - 1]
        h2 = KW_TO_H[2*k]
        hx1 = H_TO_HX[h1]
        hx2 = H_TO_HX[h2]
        is_rev = (rev6(h1) == h2)
        op     = "REV" if is_rev else "COMP"
        cl     = classify(h1)
        type_s = {"palindrome": "палинд.", "antipalindrome": "антипал.", "regular": "обычная"}[cl]

        op_col  = (GRN if is_rev else YLW) if use_color else ""
        yc1 = YANG_ANSI[hx1.yang] if use_color else ""
        yc2 = YANG_ANSI[hx2.yang] if use_color else ""

        lines.append(
            f"  КВ#{2*k-1:>2}-{2*k:<2}  {op_col}{op:4}{r}"
            f"  {yc1}h={h1:>2}{r} {hx1.sym}  "
            f"{op_col}←{op}→{r}"
            f"  {yc2}h={h2:>2}{r} {hx2.sym}"
            f"  «{hx1.name_pin[:12]}» ↔ «{hx2.name_pin[:12]}»"
            f"  {dim}[{type_s}]{r}"
        )

    lines += [
        "",
        f"  {bold}Итог: COMP={len(comp_pairs)} пары, REV={len(rev_pairs)} пар, OTHER=0{r}",
        "",
        f"  {bold}Структура:{r}",
        f"  • 24 REV-пары   = 48 обычных гексаграмм (rev6(h)≠h, rev6(h)≠comp(h))",
        f"  • 4 COMP-пары   = 8 палиндромов (rev6(h)=h, нельзя парить с собой)",
        f"  • 4 COMP=REV    = 8 антипалиндромов (rev6(h)=comp(h))",
        f"",
        f"  Принципы традиционного И Цзин:",
        f"  «综» zōng — обращение (переворот гексаграммы) = rev6 = 24 пары",
        f"  «错» cuò  — дополнение (инверсия линий)       = comp = 8 пар",
        f"  Антипалиндромы: 综=错, оба принципа совпадают  = 4 пары",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Связь с ДНК
# ---------------------------------------------------------------------------

def show_dna_symm(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    lines = [
        f"{bold}Симметрии Q6 и ДНК{r}",
        "",
        "  Двойная спираль ДНК имеет три канонических операции:",
        "",
        "  1. Комплемент    A↔T, G↔C  →  comp(h)     (смена пар оснований)",
        "  2. Реверс        5'→3' → 3'→5'  →  rev6(h)    (обращение цепи)",
        "  3. Обратный комп. реальная матрица →  comp_rev(h) (вторая нить!)",
        "",
        f"  {bold}Пример: кодон AUG (h=6, стартовый){r}",
    ]

    ex_codons = [('AUG', 6), ('UAA', 16), ('UGG', 26), ('AAA', 0), ('CCC', 63)]
    for codon, h in ex_codons:
        hx   = H_TO_HX[h]
        ch   = comp(h)
        r6h  = rev6(h)
        crh  = comp_rev(h)
        hxc  = H_TO_HX[ch]
        hxr  = H_TO_HX[r6h]
        hxcr = H_TO_HX[crh]
        yc = YANG_ANSI[hx.yang] if use_color else ""
        lines += [
            f"",
            f"  {yc}h={h:>2} {codon} {hx.sym}#{hx.kw:>2}{r} «{hx.name_pin}»",
            f"    comp(h)     = h={ch:>2} {hxc.sym}#{hxc.kw:>2} «{hxc.name_pin}»",
            f"    rev6(h)     = h={r6h:>2} {hxr.sym}#{hxr.kw:>2} «{hxr.name_pin}»",
            f"    comp_rev(h) = h={crh:>2} {hxcr.sym}#{hxcr.kw:>2} «{hxcr.name_pin}»",
        ]

    lines += [
        "",
        f"  {bold}Симметрия генетического кода:{r}",
        f"  comp_rev(h) = обратно-комплементарная нить = реальная матрица репликации",
        f"  Если кодон h = (b1,b2,b3), то comp_rev(h) = (C̄b3, C̄b2, C̄b1)",
        f"  где C̄A=U, C̄U=A, C̄G=C, C̄C=G (РНК-комплемент, обращённый)",
        "",
        f"  {bold}Антипалиндромные кодоны{r} (rev6=comp = «самодуальные»):",
    ]

    # Из hexdna.py: h → codon
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from hexdna import h_to_codon, CODON_TABLE, codon_yang
        for h in ANTIPALINDROMES:
            codon = h_to_codon(h)
            hx = H_TO_HX[h]
            aa_letter, aa_long = CODON_TABLE.get(codon, ('?', '?'))
            yc = YANG_ANSI[hx.yang] if use_color else ""
            lines.append(
                f"    {yc}h={h:>2} ({h:06b}) {codon}{r}"
                f"  {hx.sym}#{hx.kw:>2} «{hx.name_pin[:16]}»"
                f"  АК={aa_letter}"
            )
    except ImportError:
        lines.append("  (hexdna.py не найден, пропускаем)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Однострочный анализ гексаграммы
# ---------------------------------------------------------------------------

def show_hexagram(h: int, use_color: bool = True) -> str:
    hx   = H_TO_HX[h]
    ch   = comp(h)
    r6h  = rev6(h)
    crh  = comp_rev(h)
    hxc  = H_TO_HX[ch]
    hxr  = H_TO_HX[r6h]
    hxcr = H_TO_HX[crh]
    cl   = classify(h)
    yc   = (YANG_ANSI[hx.yang]) if use_color else ""
    r    = RESET if use_color else ""

    type_str = {
        "palindrome":     "ПАЛИНДРОМ (rev6=id)",
        "antipalindrome": "АНТИПАЛИНДРОМ (rev6=comp)",
        "regular":        "ОБЫЧНАЯ",
    }[cl]

    orb = orbit(h)
    orb_syms = "".join(H_TO_HX[v].sym for v in orb)

    lines = [
        "",
        f"  {yc}h={h} ({h:06b})  {hx.sym}#{hx.kw} «{hx.name_pin}»  ян={hx.yang}  [{type_str}]{r}",
        "",
        f"  comp(h)     = h={ch:>2}  {hxc.sym}#{hxc.kw:>2} «{hxc.name_pin}»  (инверсия линий)",
        f"  rev6(h)     = h={r6h:>2}  {hxr.sym}#{hxr.kw:>2} «{hxr.name_pin}»  (переворот)",
        f"  comp_rev(h) = h={crh:>2}  {hxcr.sym}#{hxcr.kw:>2} «{hxcr.name_pin}»  (суперпозиция)",
        "",
        f"  Орбита (размер {len(orb)}): {orb_syms}  {orb}",
    ]

    # Найти КВ-пару
    kw = hx.kw
    partner_kw = kw - 1 if kw % 2 == 0 else kw + 1
    partner_h  = KW_TO_H.get(partner_kw)
    if partner_h is not None:
        hxp = H_TO_HX[partner_h]
        op  = "REV" if rev6(h) == partner_h else "COMP"
        lines.append(f"  КВ-пара: КВ#{partner_kw} {hxp.sym} «{hxp.name_pin}»  (операция: {op})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexsymm — группа симметрий Q6 и пары Вэнь-вана",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexsymm.py              Группа Klein Z2×Z2 и фиксированные точки
  python hexsymm.py --kw         32 пары Вэнь-вана с типами операций
  python hexsymm.py --dna        Связь симметрий с ДНК
  python hexsymm.py --h 7        Анализ гексаграммы h=7
  python hexsymm.py --h 21       Анализ Ji Ji (антипалиндром)
        """,
    )
    parser.add_argument("--kw",      action="store_true")
    parser.add_argument("--dna",     action="store_true")
    parser.add_argument("--h",       type=int, metavar="N")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.kw:
        print()
        print(show_kw_pairs(use_color))
    elif args.dna:
        print()
        print(show_dna_symm(use_color))
    elif args.h is not None:
        if not 0 <= args.h <= 63:
            print("Ошибка: h должно быть в диапазоне 0..63")
            return
        print(show_hexagram(args.h, use_color))
    else:
        print(show_group(use_color))
        print()
        print(show_fixed_points(use_color))


if __name__ == "__main__":
    main()
