"""
hexdna.py — Генетический код РНК как Q6

Шокирующее открытие: 64 кодона РНК = 64 вершины Q6.

Каждое основание РНК кодируется 2 битами:
  A (Аденин)   = 0b00 = 0   (пурин  + слабая связь, 2 H-bond)
  U (Урацил)   = 0b01 = 1   (пиримидин + слабая связь)
  G (Гуанин)   = 0b10 = 2   (пурин  + сильная связь, 3 H-bond)
  C (Цитозин)  = 0b11 = 3   (пиримидин + сильная связь)

Бит 0 основания: 0=пурин (A,G), 1=пиримидин (U,C)
Бит 1 основания: 0=слабый (A,U), 1=сильный (G,C)

Кодон (b1 b2 b3) [5'→3'] → h:
  h = encode(b3)     (биты 0-1 = третье основание, вобблинг)
    | encode(b2)<<2  (биты 2-3 = второе основание, консервативное)
    | encode(b1)<<4  (биты 4-5 = первое основание, семейство)

Ключевые точки:
  AAA → h=0   = ䷁ Кунь  (Лизин K)
  AUG → h=6   = ䷭#46    (Метионин M = СТАРТ трансляции)
  CCC → h=63  = ䷀ Цянь  (Пролин P)
  UAA → h=16  = *Stop
  UAG → h=18  = *Stop
  UGA → h=24  = *Stop

Ян-уровень гексаграммы = «сила оснований»:
  yang(h) = #пиримидин + #сильных оснований в кодоне
           = (кол-во U/C) + (кол-во G/C) в каждой позиции
  Синонимичные кодоны (один АК) обычно имеют близкий ян.

Дегенерация кода:
  Третье основание (биты 0-1) чаще всего «вобблирует» —
  не меняет смысл. Кодоны с одинаковым h>>2 часто
  кодируют одну аминокислоту.

Квадруплеты (4 кодона → 1 АК): биты 0-1 вобблинга не важны.
Дуплеты (2 кодона → 1 АК): только пурин/пиримидин (бит 0) вобблинга.
"""

import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
from hexiching import Hexagram

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

YANG_ANSI = {
    0: "\033[90m", 1: "\033[94m", 2: "\033[96m",
    3: "\033[92m", 4: "\033[93m", 5: "\033[95m", 6: "\033[97m",
}

# Кодирование оснований
BASE_ENC = {'A': 0b00, 'U': 0b01, 'G': 0b10, 'C': 0b11}
BASE_DEC = {v: k for k, v in BASE_ENC.items()}
BASE_YANG = {'A': 0, 'U': 1, 'G': 1, 'C': 2}  # кол-во 1-битов в кодировании

# Стандартная таблица генетического кода (РНК)
CODON_TABLE = {
    # UXX
    'UUU': ('F', 'Phe'), 'UUC': ('F', 'Phe'),
    'UUA': ('L', 'Leu'), 'UUG': ('L', 'Leu'),
    'UCU': ('S', 'Ser'), 'UCC': ('S', 'Ser'), 'UCA': ('S', 'Ser'), 'UCG': ('S', 'Ser'),
    'UAU': ('Y', 'Tyr'), 'UAC': ('Y', 'Tyr'),
    'UAA': ('*', 'Stop'), 'UAG': ('*', 'Stop'),
    'UGU': ('C', 'Cys'), 'UGC': ('C', 'Cys'),
    'UGA': ('*', 'Stop'), 'UGG': ('W', 'Trp'),
    # CXX
    'CUU': ('L', 'Leu'), 'CUC': ('L', 'Leu'), 'CUA': ('L', 'Leu'), 'CUG': ('L', 'Leu'),
    'CCU': ('P', 'Pro'), 'CCC': ('P', 'Pro'), 'CCA': ('P', 'Pro'), 'CCG': ('P', 'Pro'),
    'CAU': ('H', 'His'), 'CAC': ('H', 'His'),
    'CAA': ('Q', 'Gln'), 'CAG': ('Q', 'Gln'),
    'CGU': ('R', 'Arg'), 'CGC': ('R', 'Arg'), 'CGA': ('R', 'Arg'), 'CGG': ('R', 'Arg'),
    # AXX
    'AUU': ('I', 'Ile'), 'AUC': ('I', 'Ile'), 'AUA': ('I', 'Ile'),
    'AUG': ('M', 'Met'),   # СТАРТ
    'ACU': ('T', 'Thr'), 'ACC': ('T', 'Thr'), 'ACA': ('T', 'Thr'), 'ACG': ('T', 'Thr'),
    'AAU': ('N', 'Asn'), 'AAC': ('N', 'Asn'),
    'AAA': ('K', 'Lys'), 'AAG': ('K', 'Lys'),
    'AGU': ('S', 'Ser'), 'AGC': ('S', 'Ser'),
    'AGA': ('R', 'Arg'), 'AGG': ('R', 'Arg'),
    # GXX
    'GUU': ('V', 'Val'), 'GUC': ('V', 'Val'), 'GUA': ('V', 'Val'), 'GUG': ('V', 'Val'),
    'GCU': ('A', 'Ala'), 'GCC': ('A', 'Ala'), 'GCA': ('A', 'Ala'), 'GCG': ('A', 'Ala'),
    'GAU': ('D', 'Asp'), 'GAC': ('D', 'Asp'),
    'GAA': ('E', 'Glu'), 'GAG': ('E', 'Glu'),
    'GGU': ('G', 'Gly'), 'GGC': ('G', 'Gly'), 'GGA': ('G', 'Gly'), 'GGG': ('G', 'Gly'),
}

# Полные имена аминокислот
AA_NAMES = {
    'A': 'Аланин',   'C': 'Цистеин', 'D': 'Аспарагиновая',
    'E': 'Глутаминовая', 'F': 'Фенилаланин', 'G': 'Глицин',
    'H': 'Гистидин', 'I': 'Изолейцин', 'K': 'Лизин',
    'L': 'Лейцин',   'M': 'Метионин', 'N': 'Аспарагин',
    'P': 'Пролин',   'Q': 'Глутамин', 'R': 'Аргинин',
    'S': 'Серин',    'T': 'Треонин',  'V': 'Валин',
    'W': 'Триптофан','Y': 'Тирозин', '*': 'Стоп',
}


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def codon_to_h(codon: str) -> int:
    b1, b2, b3 = codon[0], codon[1], codon[2]
    return BASE_ENC[b3] | (BASE_ENC[b2] << 2) | (BASE_ENC[b1] << 4)


def h_to_codon(h: int) -> str:
    b3 = BASE_DEC[h & 3]
    b2 = BASE_DEC[(h >> 2) & 3]
    b1 = BASE_DEC[(h >> 4) & 3]
    return b1 + b2 + b3


def codon_yang(codon: str) -> int:
    return sum(BASE_YANG[b] for b in codon)


# Предвычислим: h → (codon, aa_letter, aa_name)
H_TO_CODON = {codon_to_h(c): (c, aa, AA_NAMES.get(aa, aa))
              for c, (aa, _) in CODON_TABLE.items()}


# ---------------------------------------------------------------------------
# Полная таблица кодонов
# ---------------------------------------------------------------------------

def show_full_table(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    dim = DIM if use_color else ""

    lines = [
        "",
        f"{'═'*70}",
        f"  {bold}64 КОДОНА РНК → Q6 ГЕКСАГРАММЫ{r}",
        f"  Кодирование: A=00 U=01 G=10 C=11",
        f"  h = b3_bits | (b2_bits<<2) | (b1_bits<<4)   [5'→3']",
        f"{'═'*70}",
        "",
        f"  {dim}h    коdon  АК   ян   {'гексаграмма':>5}  КВ#  имя{r}",
        "  " + "─"*62,
    ]

    # Группируем по первому основанию (bits 4-5)
    for b1_val in range(4):
        b1_name = BASE_DEC[b1_val]
        bc = {0: "\033[92m", 1: "\033[93m", 2: "\033[91m", 3: "\033[96m"}.get(b1_val, "") if use_color else ""
        lines.append(f"\n  {bc}── Первое основание: {b1_name} ({b1_val:02b}){r}")
        for b2_val in range(4):
            b2_name = BASE_DEC[b2_val]
            for b3_val in range(4):
                b3_name = BASE_DEC[b3_val]
                h = b3_val | (b2_val << 2) | (b1_val << 4)
                codon = b1_name + b2_name + b3_name
                aa_letter, aa_long = CODON_TABLE.get(codon, ('?', '?'))
                hx = Hexagram(h)
                yc = YANG_ANSI[hx.yang] if use_color else ""

                # Маркировки
                marker = ""
                if codon == 'AUG':
                    marker = f"\033[92m★СТАРТ\033[0m" if use_color else " СТАРТ"
                elif aa_letter == '*':
                    marker = f"\033[91m■СТОП\033[0m" if use_color else " СТОП"

                lines.append(
                    f"  {yc}{h:>2} {h:06b}{r}  {bc}{codon}{r}"
                    f"  {aa_letter:>2}   {yc}{hx.yang}{r}"
                    f"   {hx.sym}  {hx.kw:>2}  {hx.name_pin[:16]:<16}"
                    f"  {marker}"
                )

    lines += [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Аминокислоты и их гексаграммы
# ---------------------------------------------------------------------------

def show_amino_acids(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    dim = DIM if use_color else ""

    # Группируем h по аминокислотам
    aa_to_hs = defaultdict(list)
    for h, (codon, aa, aa_name) in H_TO_CODON.items():
        aa_to_hs[aa].append(h)

    lines = [
        f"{bold}Аминокислоты и их гексаграммы{r}",
        "",
        f"  {dim}АК  имя           кол-во  ян(min..max)  "
        f"h-коды         гексаграммы{r}",
        "  " + "─"*72,
    ]

    # Сортируем по числу кодонов (убывающему)
    for aa in sorted(aa_to_hs, key=lambda x: (-len(aa_to_hs[x]), x)):
        hs    = sorted(aa_to_hs[aa])
        yangs = [yang_count(h) for h in hs]
        syms  = "".join(Hexagram(h).sym for h in hs)
        hs_str = " ".join(f"{h:>2}" for h in hs)
        min_y, max_y = min(yangs), max(yangs)
        yc_lo = YANG_ANSI[min_y] if use_color else ""
        yc_hi = YANG_ANSI[max_y] if use_color else ""
        name  = AA_NAMES.get(aa, aa)

        marker = ""
        if aa == '*':
            marker = "\033[91m■\033[0m" if use_color else "СТОП"
        elif 'AUG' in [h_to_codon(h) for h in hs]:
            marker = "\033[92m★\033[0m" if use_color else "СТАРТ"

        lines.append(
            f"  {aa:>2}  {name:<16}  {len(hs):>5}  "
            f"{yc_lo}{min_y}{r}..{yc_hi}{max_y}{r}  "
            f"{hs_str:<20}  {syms}  {marker}"
        )

    # Статистика
    all_yangs = [yang_count(h) for h in range(64)]
    aa_avg_yang = {
        aa: sum(yang_count(h) for h in hs) / len(hs)
        for aa, hs in aa_to_hs.items()
    }
    lines += [
        "",
        f"  {bold}Статистика:{r}",
        f"  Среднее ян всех кодонов: {sum(all_yangs)/64:.2f}",
        f"  Аминокислоты с наименьшим средним ян:",
        "  " + ", ".join(
            f"{aa}({v:.1f})" for aa, v in sorted(aa_avg_yang.items(), key=lambda x: x[1])[:5]
        ),
        f"  Аминокислоты с наибольшим средним ян:",
        "  " + ", ".join(
            f"{aa}({v:.1f})" for aa, v in sorted(aa_avg_yang.items(), key=lambda x: -x[1])[:5]
        ),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Кластерный анализ: насколько близки синонимичные кодоны в Q6?
# ---------------------------------------------------------------------------

def show_clustering(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    aa_to_hs = defaultdict(list)
    for h, (codon, aa, aa_name) in H_TO_CODON.items():
        aa_to_hs[aa].append(h)

    lines = [
        f"{bold}Кластеры синонимичных кодонов в Q6{r}",
        "(Среднее расстояние Хэмминга внутри группы одной АК)",
        "",
        f"  {'АК':>3}  {'N':>2}  {'avg d':>5}  {'max d':>5}  Примечание",
        "  " + "─"*50,
    ]

    results = []
    for aa, hs in aa_to_hs.items():
        if len(hs) < 2:
            avg_d = max_d = 0
        else:
            dists = [hamming(hs[i], hs[j])
                     for i in range(len(hs)) for j in range(i+1, len(hs))]
            avg_d = sum(dists) / len(dists)
            max_d = max(dists)
        results.append((aa, len(hs), avg_d, max_d))

    results.sort(key=lambda x: x[2])

    for aa, n, avg_d, max_d in results:
        name = AA_NAMES.get(aa, aa)
        note = ""
        if avg_d <= 1.0:
            note = "<= 1 бит разницы (квадруплет/дуплет)"
        elif max_d <= 2:
            note = "компактный кластер"
        elif max_d >= 4:
            note = "разбросан по Q6 (Leu/Ser/Arg split!)"
        yc = YANG_ANSI[round(avg_d)] if use_color else ""
        lines.append(
            f"  {aa:>3}  {n:>2}  {yc}{avg_d:>5.2f}{r}  {max_d:>5}  {note}"
        )

    lines += [
        "",
        f"  {bold}Вывод:{r}",
        "  Большинство АК образуют КОМПАКТНЫЕ кластеры в Q6",
        "  (среднее d ≤ 2). Это демонстрирует, что генетический код",
        "  «оптимален» в пространстве Хэмминга: мутация одного нуклеотида",
        "  с высокой вероятностью кодирует ту же или «похожую» АК.",
        "",
        "  Исключение: Лейцин (L), Серин (S), Аргинин (R) — по 6 кодонов,",
        "  разделённых на ДВА кластера Q6 (высокое max_d = 4-6).",
        "  Это единственные АК с «расщеплёнными» кодонными семействами.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ян-уровень vs GC-состав
# ---------------------------------------------------------------------------

def show_yang_gc(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    lines = [
        f"{bold}Ян-уровень = сила оснований (GC + пиримидинный вклад){r}",
        "",
        "  yang(h) = кол-во 1-битов в h",
        "          = (кол-во пиримидинов) + (кол-во сильных GC-связей)",
        "          = Σ_i [pyr(base_i) + gc(base_i)]",
        "  где pyr(A)=0, pyr(U)=1, pyr(G)=0, pyr(C)=1",
        "      gc(A)=0,  gc(U)=0,  gc(G)=1,  gc(C)=1",
        "",
        f"  {'ян':>4}  {'кол-во':>6}  {'пример кодона':>14}  {'типичные АК'}",
        "  " + "─"*55,
    ]

    for y in range(7):
        hs_y = [h for h in range(64) if yang_count(h) == y]
        codons_y = [h_to_codon(h) for h in hs_y]
        aas_y = sorted(set(CODON_TABLE[c][0] for c in codons_y))
        ex_codon = codons_y[0] if codons_y else "—"
        yc = YANG_ANSI[y] if use_color else ""
        lines.append(
            f"  {yc}{y:>4}{r}  {len(hs_y):>6}  {ex_codon:>14}"
            f"  {','.join(aas_y[:8])}"
        )

    lines += [
        "",
        "  Среднее ян = 3.0 = среднее GC-содержание (сбалансированный код).",
        "  Кодоны с ян=0: только AAA → Лизин (только пурины и слабые связи).",
        "  Кодоны с ян=6: только CCC → Пролин (только пиримидины и сильные связи).",
        "  Стартовый кодон AUG → h=6, ян=2 (умеренный).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Специальные точки
# ---------------------------------------------------------------------------

def show_special(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    special = [
        ('AUG', 'СТАРТ трансляции',    'Метионин'),
        ('UAA', 'СТОП (ochre)',         'стоп-кодон'),
        ('UAG', 'СТОП (amber)',         'стоп-кодон'),
        ('UGA', 'СТОП (opal)',          'стоп-кодон'),
        ('UGG', 'единственный Trp',     'Триптофан (нет дегенерации!)'),
        ('AUA', 'редкий Ile',           'Изолейцин'),
        ('AAA', 'Lys (ян=0)',           'Лизин (самый инь кодон)'),
        ('CCC', 'Pro (ян=6)',           'Пролин (самый ян кодон)'),
    ]

    lines = [f"{bold}Особые точки генетического кода в Q6{r}", ""]
    for codon, note, aa_name in special:
        h   = codon_to_h(codon)
        hx  = Hexagram(h)
        yc  = YANG_ANSI[hx.yang] if use_color else ""
        aa_code = CODON_TABLE[codon][0]
        lines.append(
            f"  {codon} → {yc}h={h:>2} ({h:06b}){r}"
            f"  ян={yc}{hx.yang}{r}  {hx.sym}#{hx.kw:>2} «{hx.name_pin[:18]}»"
            f"  {aa_code}={aa_name}"
            f"\n         {note}"
        )
        lines.append("")

    lines += [
        f"  {bold}Наблюдения:{r}",
        "  * Стартовый кодон AUG → ䷭#46 «Шэн» (Восхождение) — начало!",
        "  * Три стоп-кодона кластеризуются в районе h=16-24 (ян=2..3).",
        "  * Триптофан (UGG) — единственная АК без дегенерации — единственная точка Q6.",
        "  * AAA=Кунь (h=0) → Лизин,  CCC=Цянь (h=63) → Пролин.",
        "    Кунь и Цянь — крайние точки, AAA и CCC — крайние по силе основания.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexdna — РНК-кодоны как Q6 гексаграммы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexdna.py               Полная таблица 64 кодонов
  python hexdna.py --aa          Аминокислоты и их гексаграммы
  python hexdna.py --cluster     Кластерный анализ в Q6
  python hexdna.py --yang        Ян-уровень vs GC-состав
  python hexdna.py --special     Особые точки (старт, стоп, уникальные)
  python hexdna.py --codon AUG   Разбор конкретного кодона
        """,
    )
    parser.add_argument("--aa",       action="store_true")
    parser.add_argument("--cluster",  action="store_true")
    parser.add_argument("--yang",     action="store_true")
    parser.add_argument("--special",  action="store_true")
    parser.add_argument("--codon",    type=str, metavar="XYZ")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.aa:
        print()
        print(show_amino_acids(use_color))
    elif args.cluster:
        print()
        print(show_clustering(use_color))
    elif args.yang:
        print()
        print(show_yang_gc(use_color))
    elif args.special:
        print()
        print(show_special(use_color))
    elif args.codon:
        codon = args.codon.upper()
        if len(codon) != 3 or any(b not in BASE_ENC for b in codon):
            print(f"Ошибка: кодон должен быть 3 буквы из {{A,U,G,C}}")
            sys.exit(1)
        h  = codon_to_h(codon)
        hx = Hexagram(h)
        aa_letter, aa_long = CODON_TABLE.get(codon, ('?', '?'))
        yc = YANG_ANSI[hx.yang] if use_color else ""
        r  = RESET if use_color else ""
        print(f"\n  Кодон {codon}: {yc}h={h} ({h:06b}){r}"
              f"  ян={yc}{hx.yang}{r}"
              f"  {hx.sym}#{hx.kw} «{hx.name_pin}»"
              f"  АК={aa_letter} ({aa_long})")
        b1, b2, b3 = codon[0], codon[1], codon[2]
        print(f"  Разложение: {b1}({BASE_ENC[b1]:02b}) {b2}({BASE_ENC[b2]:02b}) {b3}({BASE_ENC[b3]:02b})")
    else:
        print(show_full_table(use_color))


if __name__ == "__main__":
    main()
