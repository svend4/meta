"""
hexreading.py — Гадание на И-цзин в трёх системах одновременно

Бросок монет / счёт тысячелистника → гексаграмма + меняющиеся черты →
результирующая гексаграмма. Это движение в Q6:

  h_start  XOR  changing_mask  =  h_result

Интерпретируется в трёх системах:
  Касаткин:  (x,y,z) → (x',y',z')  в кубе 4×4×4
  Крюков:    закрытые/открытые окна тела
  Лю-Синь:   активные/неактивные цвета

ИСПОЛЬЗОВАНИЕ:
  Стандартный И-цзин:
    1. Бросок = начальная гексаграмма h (по номеру или 6 чертам)
    2. Меняющиеся черты → результирующая гексаграмма h'
  Наш синтез:
    1. Начальное состояние системы = h
    2. Воздействие (флип битов) = меняющиеся черты
    3. Конечное состояние = h'
    4. Путь h→h' = маршрут в Q6
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, antipode, neighbors, shortest_path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexkub"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexboya"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexliuxing"))

from hexiching import Hexagram, KW_FROM_H, KW_DATA, TRIGRAMS, _KW_TO_H
from hexkub import KubNumber
from hexboya import BodyState, ZONE_NAMES
from hexliuxing import LiuElement, ELEMENTS, RELATIONS, LiuSystem

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

YANG_COLORS = {
    0: "\033[90m", 1: "\033[34m", 2: "\033[36m",
    3: "\033[32m", 4: "\033[33m", 5: "\033[35m", 6: "\033[37m",
}


# ---------------------------------------------------------------------------
# Reading — одно гадание
# ---------------------------------------------------------------------------

class Reading:
    """
    Единое чтение: начальная гексаграмма + меняющиеся черты + результат.

    Атрибуты:
      h_start:  начальная гексаграмма (Q6-вершина)
      changing: набор номеров черт {1..6}, которые меняются
      h_result: результирующая гексаграмма
    """

    def __init__(self, h_start: int, changing: set = None):
        self.h_start  = h_start
        self.changing = set(changing) if changing else set()
        # Меняющаяся черта N (1-based) = бит N-1
        mask = sum(1 << (n - 1) for n in self.changing)
        self.h_result = h_start ^ mask
        self.mask = mask

    # --- геттеры ---

    @property
    def start(self) -> Hexagram:  return Hexagram(self.h_start)
    @property
    def result(self) -> Hexagram: return Hexagram(self.h_result)

    def distance(self) -> int:
        """Расстояние Хэмминга = число меняющихся черт."""
        return hamming(self.h_start, self.h_result)

    def path(self) -> list:
        """Кратчайший путь в Q6 от старта к результату."""
        return shortest_path(self.h_start, self.h_result)

    # --- три системы: старт ---

    def kasatkin_start(self) -> dict:
        x = self.h_start & 3
        y = (self.h_start >> 2) & 3
        z = (self.h_start >> 4) & 3
        vol = (x+1)*(y+1)*(z+1)
        return {"xyz": (x,y,z), "volume": vol}

    def kasatkin_result(self) -> dict:
        x = self.h_result & 3
        y = (self.h_result >> 2) & 3
        z = (self.h_result >> 4) & 3
        vol = (x+1)*(y+1)*(z+1)
        return {"xyz": (x,y,z), "volume": vol}

    def kryukov_start(self) -> BodyState:  return BodyState(self.h_start)
    def kryukov_result(self) -> BodyState: return BodyState(self.h_result)

    def liuxing_start(self) -> list:
        return [LiuElement(i) for i in range(6) if (self.h_start >> i) & 1]
    def liuxing_result(self) -> list:
        return [LiuElement(i) for i in range(6) if (self.h_result >> i) & 1]

    def liuxing_changed_elements(self) -> list:
        """Элементы, которые изменились (активированы или деактивированы)."""
        return [LiuElement(i) for i in range(6) if (self.mask >> i) & 1]

    # --- полное описание ---

    def describe(self, use_color: bool = True) -> str:
        bold = BOLD if use_color else ""
        r    = RESET if use_color else ""
        dim  = DIM  if use_color else ""

        hs = self.start
        hr = self.result
        ks = self.kasatkin_start()
        kr = self.kasatkin_result()
        bs = self.kryukov_start()
        br = self.kryukov_result()
        ls = self.liuxing_start()
        lr = self.liuxing_result()
        changed_elems = self.liuxing_changed_elements()

        yc_s = YANG_COLORS[hs.yang] if use_color else ""
        yc_r = YANG_COLORS[hr.yang] if use_color else ""

        elem_shorts = [e["short"] for e in ELEMENTS]
        elem_colors = ["\033[31m","\033[33m","\033[32m","\033[36m","\033[34m","\033[35m"]

        def liu_fmt(elems):
            if not elems:
                return "∅ (Чёрный)"
            return "+".join(
                ((elem_colors[e.idx] if use_color else "") + e.short + r)
                for e in elems
            )

        sep = "─" * 60

        lines = [
            "",
            sep,
            f"  {bold}ГАДАНИЕ{r}  {dim}h={self.h_start}  →  h={self.h_result}{r}"
            f"  (расстояние={self.distance()})",
            sep,
            "",
            # --- НАЧАЛО ---
            f"  {bold}НАЧАЛО{r}  {yc_s}{hs.sym} #{hs.kw:>2}  «{hs.name_ru}»{r}",
            f"  {hs.name_pin}  ({hs.name_cn})   h={self.h_start} ({self.h_start:06b})   ян={hs.yang}",
            "",
            f"  {dim}└ И-цзин:{r}  {hs.upper['sym']} {hs.upper['ru']} (верх)  /  "
            f"{hs.lower['sym']} {hs.lower['ru']} (низ)",
            "",
            f"  {dim}└ Касаткин:{r}  координаты {ks['xyz']}  объём={ks['volume']}",
            f"  {dim}└ Крюков:{r}   [{', '.join(bs.open_zones()) or '∅'}]  "
            f"сфера={_sphere_label(self.h_start, use_color)}",
            f"  {dim}└ Лю-Синь:{r}  {liu_fmt(ls)}",
            "",
        ]

        if not self.changing:
            lines += [
                "  (нет меняющихся черт — статическое чтение)",
                "",
            ]
        else:
            # --- МЕНЯЮЩИЕСЯ ЧЕРТЫ ---
            lines.append(f"  {bold}МЕНЯЮЩИЕСЯ ЧЕРТЫ: {sorted(self.changing)}{r}")
            for n in sorted(self.changing):
                bit = n - 1
                was = (self.h_start >> bit) & 1
                ec = (elem_colors[bit] if use_color else "")
                elem = elem_shorts[bit]
                zone = ZONE_NAMES[bit]
                lines.append(
                    f"    Черта {n}  {ec}{elem}{r} / {zone}  "
                    f"{'ян→инь' if was else 'инь→ян'}  "
                    f"({'деактивация' if was else 'активация'})"
                )
            lines.append("")

            # --- ПУТЬ ---
            path = self.path()
            if len(path) > 2:
                path_hexs = [Hexagram(h) for h in path]
                lines.append(f"  {bold}ПУТЬ В Q6:{r}")
                for i in range(len(path) - 1):
                    ha, hb = path_hexs[i], path_hexs[i+1]
                    diff_bit = (path[i] ^ path[i+1]).bit_length() - 1
                    ec = (elem_colors[diff_bit] if use_color else "")
                    lines.append(
                        f"    {ha.sym}#{ha.kw:<2}→{hb.sym}#{hb.kw:<2}"
                        f"  бит{diff_bit}={ec}{elem_shorts[diff_bit]}{r}/{ZONE_NAMES[diff_bit]}"
                    )
                lines.append("")

            # --- РЕЗУЛЬТАТ ---
            lines += [
                f"  {bold}РЕЗУЛЬТАТ{r}  {yc_r}{hr.sym} #{hr.kw:>2}  «{hr.name_ru}»{r}",
                f"  {hr.name_pin}  ({hr.name_cn})   h={self.h_result} ({self.h_result:06b})   ян={hr.yang}",
                "",
                f"  {dim}└ И-цзин:{r}  {hr.upper['sym']} {hr.upper['ru']} (верх)  /  "
                f"{hr.lower['sym']} {hr.lower['ru']} (низ)",
                "",
                f"  {dim}└ Касаткин:{r}  координаты {kr['xyz']}  объём={kr['volume']}",
                f"  {dim}└ Крюков:{r}   [{', '.join(br.open_zones()) or '∅'}]  "
                f"сфера={_sphere_label(self.h_result, use_color)}",
                f"  {dim}└ Лю-Синь:{r}  {liu_fmt(lr)}",
                "",
            ]

        # --- СИНТЕЗ ---
        delta_y = hr.yang - hs.yang
        ks_vol_ratio = kr['volume'] / ks['volume'] if ks['volume'] else 0
        lines += [
            f"  {bold}СИНТЕЗ:{r}",
            f"    Ян-изменение:   {delta_y:+d}  ({hs.yang} → {hr.yang})",
            f"    Объём (Касат.): {ks['volume']} → {kr['volume']}"
            f"  {'↑' if kr['volume']>ks['volume'] else '↓' if kr['volume']<ks['volume'] else '='}"
            f"  (×{ks_vol_ratio:.2f})",
            f"    Антипод старта: h={antipode(self.h_start)} = {Hexagram(antipode(self.h_start)).sym}"
            f"#{KW_FROM_H[antipode(self.h_start)]} «{Hexagram(antipode(self.h_start)).name_ru}»",
            f"    Антипод резул.: h={antipode(self.h_result)} = {Hexagram(antipode(self.h_result)).sym}"
            f"#{KW_FROM_H[antipode(self.h_result)]} «{Hexagram(antipode(self.h_result)).name_ru}»",
            "",
            sep,
        ]
        return "\n".join(lines)


def _sphere_label(h: int, use_color: bool = True) -> str:
    """Метка сферы Крюкова."""
    d = hamming(0, h)
    if d == 0: return "центр"
    if d == 1: return "МВС"
    if d == 2: return "СВС"
    if d <= 4: return "БВС"
    return f"вне (d={d})"


# ---------------------------------------------------------------------------
# Генерация случайного броска
# ---------------------------------------------------------------------------

def random_reading(seed: int = None) -> Reading:
    """Случайное гадание (6 монет)."""
    import random
    rng = random.Random(seed)
    h = rng.randint(0, 63)
    # Каждая черта меняется с вероятностью 1/4 (как в методе трёх монет)
    changing = {i+1 for i in range(6) if rng.random() < 0.25}
    return Reading(h, changing)


# ---------------------------------------------------------------------------
# Артефакты: особые гадания из истории
# ---------------------------------------------------------------------------

FAMOUS_READINGS = {
    "tai_pi": {
        "h": 7,   # Тай #11 «Мир»
        "changing": {3},
        "desc": "Тай → Гуй Мэй: равновесие нарушено средней чертой",
    },
    "ji_ji": {
        "h": 21,  # Цзи Цзи #63 «После завершения»
        "changing": {1, 2, 3, 4, 5, 6},
        "desc": "Цзи Цзи → Вэй Цзи: все черты меняются = полный переход через центр",
    },
    "qian_kun": {
        "h": 63,  # Цянь #1 «Небо»
        "changing": {1, 2, 3, 4, 5, 6},
        "desc": "Цянь → Кунь: полное превращение неба в землю",
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_h(s: str) -> int:
    s = s.strip()
    if s.startswith(("KW:", "kw:")):
        return _KW_TO_H[int(s[3:]) - 1]
    if len(s) == 6 and all(c in "01" for c in s):
        return int(s, 2)
    return int(s)


def main():
    parser = argparse.ArgumentParser(
        description="hexreading — Гадание на И-цзин в трёх системах",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexreading.py --hex 21              Чистое чтение h=21 (без изменений)
  python hexreading.py --hex KW:63           Гексаграмма #63 по Вэнь-вану
  python hexreading.py --hex 21 --change 1 3 5  Меняем черты 1,3,5
  python hexreading.py --hex 0 --change 1 3 5   Кунь → Цзи Цзи!
  python hexreading.py --random              Случайное гадание
  python hexreading.py --random --seed 42    Воспроизводимое гадание
  python hexreading.py --famous ji_ji        Известное гадание «Цзи Цзи»
  python hexreading.py --famous qian_kun     Цянь → Кунь (небо→земля)
        """,
    )
    parser.add_argument("--hex",    type=str, metavar="H")
    parser.add_argument("--change", type=int, nargs="+", metavar="N",
                        help="Номера меняющихся черт (1-6)")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--famous", type=str, choices=list(FAMOUS_READINGS),
                        metavar="{" + "|".join(FAMOUS_READINGS) + "}")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    use_color = not args.no_color

    if args.random:
        reading = random_reading(args.seed)
    elif args.famous:
        fr = FAMOUS_READINGS[args.famous]
        reading = Reading(fr["h"], fr["changing"])
        print(f"\n  «{fr['desc']}»")
    elif args.hex:
        h = _parse_h(args.hex)
        changing = set(args.change) if args.change else set()
        reading = Reading(h, changing)
    else:
        parser.print_help()
        return

    print(reading.describe(use_color))


if __name__ == "__main__":
    main()
