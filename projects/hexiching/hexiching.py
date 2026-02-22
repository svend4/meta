"""
hexiching.py — И-цзин как Q6 (64 гексаграммы = 64 вершины 6-мерного куба)

КОДИРОВКА:
  h ∈ {0..63} = 6-битное число = гексаграмма
  бит 0 = нижняя черта (初爻, первая яо)
  бит 5 = верхняя черта (上爻, шестая яо)
  ян (—) = 1,  инь (-- --) = 0

ТРИГРАММЫ (нижние 3 бита / верхние 3 бита):
  h & 7   = нижняя триграмма (нижняя гуа, 下卦)
  h >> 3  = верхняя триграмма (верхняя гуа, 上卦)

ПРИМЕРЫ:
  h=0   (000000) = Кунь/Кунь = ䷁ #2  «Кунь» (Земля) — всё инь
  h=63  (111111) = Цянь/Цянь = ䷀ #1  «Цянь» (Небо) — всё ян
  h=21  (010101) = Кань/Ли   = ䷾ #63 «Цзи Цзи» (После завершения)
  h=42  (101010) = Ли/Кань   = ䷿ #64 «Вэй Цзи» (До завершения)

  h=21 и h=42 — антиподы в Q6, они же — последние две гексаграммы Вэнь-вана!
  h=0  и h=63 — антиподы, они же — первые две гексаграммы!

СВЯЗЬ С ЛЮ-СИНЬ:
  Восемь триграмм (八卦) задают 8 углов Q3 ⊂ Q6
  h & 7 = позиция в Q3 = нижняя триграмма = инь/ян три стихии

ДВИЖЕНИЕ В Q6 = «МЕНЯЮЩАЯСЯ ЧЕРТА» (变爻):
  Переход h1→h2 по ребру Q6 = одна меняющаяся черта
  Расстояние Хэмминга = число меняющихся черт
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, antipode, neighbors, shortest_path

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
YAN_C  = "\033[33m"   # янтарный — ян
YIN_C  = "\033[34m"   # синий    — инь


# ---------------------------------------------------------------------------
# 8 ТРИГРАММ (八卦)
# ---------------------------------------------------------------------------

# Ключ = числовое значение 3-битной триграммы (биты 2,1,0 = верх,сред,низ)
TRIGRAMS = {
    0: {"cn": "坤", "pin": "Kūn",  "ru": "Кунь",  "en": "Earth",    "sym": "☷",
        "elem": "Земля",   "attr": "Послушание, восприимчивость",  "body": "Живот"},
    1: {"cn": "震", "pin": "Zhèn", "ru": "Чжэнь", "en": "Thunder",  "sym": "☳",
        "elem": "Гром",    "attr": "Движение, побуждение",          "body": "Стопа"},
    2: {"cn": "坎", "pin": "Kǎn",  "ru": "Кань",  "en": "Water",    "sym": "☵",
        "elem": "Вода",    "attr": "Опасность, глубина",            "body": "Ухо"},
    3: {"cn": "兌", "pin": "Duì",  "ru": "Дуй",   "en": "Lake",     "sym": "☱",
        "elem": "Озеро",   "attr": "Радость, открытость",           "body": "Рот"},
    4: {"cn": "艮", "pin": "Gèn",  "ru": "Гэнь",  "en": "Mountain", "sym": "☶",
        "elem": "Гора",    "attr": "Покой, неподвижность",          "body": "Рука"},
    5: {"cn": "離", "pin": "Lí",   "ru": "Ли",    "en": "Fire",     "sym": "☲",
        "elem": "Огонь",   "attr": "Ясность, сцепление",            "body": "Глаз"},
    6: {"cn": "巽", "pin": "Xùn",  "ru": "Сюнь",  "en": "Wind",     "sym": "☴",
        "elem": "Ветер",   "attr": "Мягкость, проникновение",       "body": "Бедро"},
    7: {"cn": "乾", "pin": "Qián", "ru": "Цянь",  "en": "Heaven",   "sym": "☰",
        "elem": "Небо",    "attr": "Творчество, сила",              "body": "Голова"},
}

# ---------------------------------------------------------------------------
# 64 ГЕКСАГРАММЫ (порядок Вэнь-вана, KW1..KW64)
# kw_to_h[i] = h для i-й гексаграммы (i = 0..63, KW = i+1)
# ---------------------------------------------------------------------------

# Формула: h = (upper_tri << 3) | lower_tri
# Проверено: KW63→h=21 (Кань/Ли), KW64→h=42 (Ли/Кань)
_KW_TO_H = [
    63, 0,17,34,23,58, 2,16,55,59, 7,56,61,47, 4, 8,
    25,38, 3,48,41,37,32, 1,57,39,33,30,18,45,28,14,
    60,15,40, 5,53,43,20,10,35,49,31,62,24, 6,26,22,
    29,46, 9,36,52,11,13,44,54,27,50,19,51,12,21,42,
]

# Обратное: KW_FROM_H[h] = номер по Вэнь-вану (1..64)
KW_FROM_H = [0] * 64
for _kw0, _h in enumerate(_KW_TO_H):
    KW_FROM_H[_h] = _kw0 + 1

# 64 имени гексаграмм (пиньинь + кратко по-русски)
_HEXAGRAM_NAMES = [
    # KW 1..16
    ("Цянь",     "乾", "Небо / Творчество"),
    ("Кунь",     "坤", "Земля / Восприятие"),
    ("Чжунь",    "屯", "Прорастание / Трудное начало"),
    ("Мэн",      "蒙", "Юность / Незнание"),
    ("Сюй",      "需", "Ожидание / Питание"),
    ("Сун",      "訟", "Тяжба / Конфликт"),
    ("Ши",       "師", "Войско / Дисциплина"),
    ("Би",       "比", "Единение / Союз"),
    ("Сяо Сюй",  "小畜", "Малое накопление"),
    ("Люй",      "履", "Поступь / Поведение"),
    ("Тай",      "泰", "Мир / Расцвет"),
    ("Пи",       "否", "Застой / Упадок"),
    ("Тун Жэнь", "同人", "Братство / Товарищество"),
    ("Да Ю",     "大有", "Великое обладание"),
    ("Цянь",     "謙", "Смирение / Скромность"),
    ("Юй",       "豫", "Воодушевление / Восторг"),
    # KW 17..32
    ("Суй",      "隨", "Следование"),
    ("Гу",       "蠱", "Исправление / Разложение"),
    ("Линь",     "臨", "Приближение / Надзор"),
    ("Гуань",    "觀", "Созерцание"),
    ("Ши Хэ",    "噬嗑", "Прокусывание / Откусывание"),
    ("Би",       "賁", "Убранство / Изящество"),
    ("Бо",       "剝", "Разрушение / Распад"),
    ("Фу",       "復", "Возврат / Возвращение"),
    ("У Ван",    "無妄", "Непорочность / Невинность"),
    ("Да Сюй",   "大畜", "Великое накопление"),
    ("И",        "頤", "Питание / Кормление"),
    ("Да Го",    "大過", "Великий перевес / Избыток"),
    ("Кань",     "坎", "Бездна / Вода"),
    ("Ли",       "離", "Сияние / Огонь"),
    ("Сянь",     "咸", "Влечение / Соприкосновение"),
    ("Хэн",      "恆", "Постоянство / Длительность"),
    # KW 33..48
    ("Дунь",     "遯", "Отступление / Уход"),
    ("Да Чжуан", "大壯", "Великая мощь"),
    ("Цзинь",    "晉", "Восход / Прогресс"),
    ("Мин И",    "明夷", "Помрачение / Сумерки"),
    ("Цзя Жэнь", "家人", "Семья"),
    ("Куй",      "睽", "Разлад / Противостояние"),
    ("Цзянь",    "蹇", "Препятствие / Хромота"),
    ("Цзе",      "解", "Разрешение / Освобождение"),
    ("Сунь",     "損", "Убыль / Уменьшение"),
    ("И",        "益", "Прибавление / Увеличение"),
    ("Гуай",     "夬", "Решимость / Прорыв"),
    ("Гоу",      "姤", "Встреча / Сближение"),
    ("Цуй",      "萃", "Сбор / Объединение"),
    ("Шэн",      "升", "Подъём / Восхождение"),
    ("Кунь",     "困", "Истощение / Угнетение"),
    ("Цзин",     "井", "Колодец"),
    # KW 49..64
    ("Гэ",       "革", "Революция / Перемена"),
    ("Дин",      "鼎", "Котёл / Жертвенник"),
    ("Чжэнь",    "震", "Гром / Возбуждение"),
    ("Гэнь",     "艮", "Гора / Покой"),
    ("Цзянь",    "漸", "Постепенность / Рост"),
    ("Гуй Мэй",  "歸妹", "Невеста / Возвратная дева"),
    ("Фэн",      "豐", "Полнота / Изобилие"),
    ("Люй",      "旅", "Странник / Путешествие"),
    ("Сюнь",     "巽", "Ветер / Мягкость"),
    ("Дуй",      "兌", "Радость / Озеро"),
    ("Хуань",    "渙", "Рассеяние / Дисперсия"),
    ("Цзе",      "節", "Ограничение / Умеренность"),
    ("Чжун Фу",  "中孚", "Внутренняя правда"),
    ("Сяо Го",   "小過", "Малый перевес"),
    ("Цзи Цзи",  "既濟", "После завершения"),
    ("Вэй Цзи",  "未濟", "До завершения"),
]

# Сборный массив: KW_DATA[kw-1] = dict
KW_DATA = []
for i, (pin, cn, ru) in enumerate(_HEXAGRAM_NAMES):
    h = _KW_TO_H[i]
    lower_t = h & 7
    upper_t = (h >> 3) & 7
    KW_DATA.append({
        "kw":    i + 1,
        "h":     h,
        "pin":   pin,
        "cn":    cn,
        "ru":    ru,
        "lower": lower_t,
        "upper": upper_t,
        "yang":  yang_count(h),
        "sym":   chr(0x4DC0 + i),   # Unicode ䷀..䷿
    })


# ---------------------------------------------------------------------------
# Класс Hexagram
# ---------------------------------------------------------------------------

class Hexagram:
    """Одна гексаграмма: Q6-вершина h с И-цзин интерпретацией."""

    def __init__(self, h: int):
        if not 0 <= h <= 63:
            raise ValueError(f"h ∈ {{0..63}}, получено {h}")
        self.h = h
        self.kw = KW_FROM_H[h]
        self._kw = KW_DATA[self.kw - 1]

    # --- базовые свойства ---

    @property
    def name_ru(self) -> str: return self._kw["ru"]
    @property
    def name_pin(self) -> str: return self._kw["pin"]
    @property
    def name_cn(self) -> str: return self._kw["cn"]
    @property
    def sym(self) -> str: return self._kw["sym"]
    @property
    def yang(self) -> int: return yang_count(self.h)
    @property
    def lower(self) -> dict: return TRIGRAMS[self.h & 7]
    @property
    def upper(self) -> dict: return TRIGRAMS[(self.h >> 3) & 7]

    # --- ASCII-арт ---

    def ascii_art(self, use_color: bool = True) -> str:
        """6 строк: верхняя черта → нижняя черта."""
        lines = []
        for bit in range(5, -1, -1):  # от старшего к младшему
            yang = (self.h >> bit) & 1
            if yang:
                line = "  ——————  "
                c = YAN_C if use_color else ""
            else:
                line = "  ——  ——  "
                c = YIN_C if use_color else ""
            r = RESET if use_color else ""
            # Разделяем триграммы пробелом
            spacer = "·" if bit == 3 else " "
            lines.append(f"  {c}{line}{r}  {spacer}")
        return "\n".join(lines)

    def ascii_art_compact(self, use_color: bool = True) -> str:
        """Одностроковое представление 6 черт (верх→низ)."""
        parts = []
        for bit in range(5, -1, -1):
            yang = (self.h >> bit) & 1
            c = (YAN_C if use_color else "")
            r = (RESET if use_color else "")
            parts.append(f"{c}{'—' if yang else '⚊'}{r}")
        return " ".join(parts)

    def describe(self, use_color: bool = True) -> str:
        bold = BOLD if use_color else ""
        r = RESET if use_color else ""
        yc = YAN_C if use_color else ""
        ic = YIN_C if use_color else ""

        low = self.lower
        upp = self.upper
        lines = [
            f"{bold}{self.sym} #{self.kw:>2}  {self.name_pin} ({self.name_cn}){r}",
            f"  h={self.h:2d} ({self.h:06b})  ян={self.yang}  «{self.name_ru}»",
            "",
            "  Верхняя (上卦):",
            f"    {upp['sym']} {upp['ru']} ({upp['cn']}, {upp['pin']}) — {upp['attr']}",
            "  Нижняя (下卦):",
            f"    {low['sym']} {low['ru']} ({low['cn']}, {low['pin']}) — {low['attr']}",
            "",
            self.ascii_art(use_color),
            "",
            f"  Антипод в Q6: h={antipode(self.h)} → {Hexagram(antipode(self.h)).sym}"
            f" #{KW_FROM_H[antipode(self.h)]} «{Hexagram(antipode(self.h)).name_ru}»",
        ]
        return "\n".join(lines)

    def changing_lines_to(self, other: "Hexagram") -> list:
        """Список номеров изменяющихся черт (1=нижняя, 6=верхняя)."""
        diff = self.h ^ other.h
        return [i + 1 for i in range(6) if (diff >> i) & 1]

    def neighbors_hexagrams(self) -> list:
        """6 соседних гексаграмм (по одной меняющейся черте)."""
        return [Hexagram(n) for n in neighbors(self.h)]


# ---------------------------------------------------------------------------
# Функции для отображения
# ---------------------------------------------------------------------------

def trigram_table(use_color: bool = True) -> str:
    """Таблица 8 триграмм с их свойствами."""
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    lines = [
        f"{bold}Восемь триграмм (八卦) = 8 вершин Q3{r}",
        "─" * 60,
        f"{'h':>3}  {'бин':>4}  {'сим':>3}  {'пинь':>8}  {'кит':>3}  "
        f"{'рус':>6}  {'стихия':>10}",
        "─" * 60,
    ]
    for h3 in range(8):
        t = TRIGRAMS[h3]
        yc = YAN_C if bin(h3).count('1') >= 2 else YIN_C
        c = yc if use_color else ""
        lines.append(
            f"{c}{h3:>3}  {h3:04b}  {t['sym']:>3}  {t['pin']:>8}  "
            f"{t['cn']:>3}  {t['ru']:>6}  {t['elem']}{r}"
        )
    lines += [
        "─" * 60,
        "  Каждая триграмма = 3-битная вершина Q3.",
        "  Пары триграмм (верх+низ) → 64 гексаграммы = Q6.",
    ]
    return "\n".join(lines)


def hexagram_table(use_color: bool = True, yang_filter: int = None) -> str:
    """Таблица 64 гексаграмм, опц. отфильтрованных по yan-уровню."""
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    yang_colors = {
        0: "\033[90m", 1: "\033[34m", 2: "\033[36m",
        3: "\033[32m", 4: "\033[33m", 5: "\033[35m", 6: "\033[37m",
    }
    lines = [
        f"{bold}64 гексаграммы = Q6{r}"
        + (f"  (фильтр: ян={yang_filter})" if yang_filter is not None else ""),
        f"{'#KW':>4}  {'h':>3}  {'бин':>8}  {'ян':>3}  {'сим':>3}  "
        f"{'верх':>5}  {'низ':>5}  {'название'}",
        "─" * 65,
    ]
    for d in KW_DATA:
        if yang_filter is not None and d["yang"] != yang_filter:
            continue
        yc = yang_colors[d["yang"]] if use_color else ""
        low = TRIGRAMS[d["lower"]]
        upp = TRIGRAMS[d["upper"]]
        lines.append(
            f"{yc}{d['kw']:>4}  {d['h']:>3}  {d['h']:08b}  "
            f"{d['yang']:>3}  {d['sym']:>3}  "
            f"{upp['sym']}{upp['ru'][:3]:>3}  "
            f"{low['sym']}{low['ru'][:3]:>3}  "
            f"{d['pin']} «{d['ru'][:28]}»{r}"
        )
    return "\n".join(lines)


def antipode_pairs(use_color: bool = True) -> str:
    """32 пары антиподов в Q6 = пары гексаграмм."""
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    lines = [
        f"{bold}32 пары антиподов в Q6{r}",
        "(Каждая пара: h + (63-h) = 63, инверсия всех черт)",
        "─" * 60,
    ]
    seen = set()
    for h in range(64):
        ap = 63 - h
        if h < ap and h not in seen:
            seen.add(h)
            seen.add(ap)
            hx = Hexagram(h)
            ax = Hexagram(ap)
            yc1 = "\033[33m" if use_color else ""
            yc2 = "\033[34m" if use_color else ""
            lines.append(
                f"  {yc1}{hx.sym}#{hx.kw:>2} h={h:>2} «{hx.name_ru[:16]:>16}»{r}"
                f"  ↔  "
                f"{yc2}{ax.sym}#{ax.kw:>2} h={ap:>2} «{ax.name_ru[:16]}»{r}"
            )
    return "\n".join(lines)


def path_as_iching(path: list, use_color: bool = True) -> str:
    """Показать путь в Q6 как последовательность изменяющихся черт."""
    lines = [
        f"Путь {path[0]}→{path[-1]} как И-цзин (меняющиеся черты):",
        "",
    ]
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        ha = Hexagram(a)
        hb = Hexagram(b)
        changed = ha.changing_lines_to(hb)
        line_no = changed[0]  # всегда 1 черта (ребро Q6)
        yc = YAN_C if (b >> (line_no-1)) & 1 else YIN_C
        r = RESET if use_color else ""
        c = yc if use_color else ""
        action = "→ янь" if (b >> (line_no-1)) & 1 else "→ инь"
        lines.append(
            f"  {ha.sym}#{ha.kw:<2} {ha.name_pin:>12} "
            f"  черта {line_no} {c}{action}{r}  "
            f"→ {hb.sym}#{hb.kw:<2} {hb.name_pin}"
        )
    return "\n".join(lines)


def show_special_hexagrams(use_color: bool = True) -> str:
    """Особые гексаграммы и их место в Q6."""
    lines = [
        "Особые гексаграммы в Q6:",
        "",
        "  КРАЙНИЕ (ян=0 и ян=6):",
    ]
    h0 = Hexagram(0)
    h63 = Hexagram(63)
    lines += [
        f"    {h0.sym} #{h0.kw} h=0  ян=0  «{h0.name_ru}» — всё инь",
        f"    {h63.sym} #{h63.kw} h=63 ян=6  «{h63.name_ru}» — всё ян",
        f"    Расстояние Хэмминга: {hamming(0,63)} (максимальное = диаметр Q6)",
        "",
        "  АНТИПОДЫ h=21 ↔ h=42 (КЛЮЧЕВЫЕ!):",
    ]
    h21 = Hexagram(21)
    h42 = Hexagram(42)
    lines += [
        f"    {h21.sym} #{h21.kw} h=21 (010101) ян=3  «{h21.name_ru}»",
        f"       Верх: {h21.upper['sym']} {h21.upper['ru']},  Низ: {h21.lower['sym']} {h21.lower['ru']}",
        f"    {h42.sym} #{h42.kw} h=42 (101010) ян=3  «{h42.name_ru}»",
        f"       Верх: {h42.upper['sym']} {h42.upper['ru']},  Низ: {h42.lower['sym']} {h42.lower['ru']}",
        "",
        "    Обе — последние в порядке Вэнь-вана (#63 и #64).",
        "    Обе имеют ян=3 (равновесие инь-ян).",
        "    Касаткин: (1,1,1) объём=8=2³ и (2,2,2) объём=27=3³.",
        "    Лю-Синь: R+G+B=Белый и Y+C+M=Белый (дополнительные триады).",
        "",
        "  ЯН=3 СИММЕТРИЧНЫЕ (центр Q6, «серединные гексаграммы»):",
    ]
    yang3_h = [h for h in range(64) if yang_count(h) == 3]
    lines.append(f"    Всего {len(yang3_h)} гексаграмм с ян=3:")
    for h in yang3_h[:8]:  # первые 8 для компактности
        hx = Hexagram(h)
        lines.append(f"    {hx.sym}#{hx.kw:>2} h={h:>2}  «{hx.name_ru[:20]}»")
    lines.append(f"    ... (и ещё {len(yang3_h)-8})")
    return "\n".join(lines)


def wuwang_cycle(use_color: bool = True) -> str:
    """
    6-цикл «меняющейся черты» на уровне ян=1.
    Аналог цикла творения Лю-Синь: каждый шаг = одна меняющаяся черта.
    """
    yang1 = sorted([h for h in range(64) if yang_count(h) == 1],
                   key=lambda h: h.bit_length()-1)
    lines = [
        "6-цикл на ян=1 гексаграммах = цикл творения Лю-Синь в И-цзин:",
        "",
    ]
    for i in range(len(yang1)):
        h_cur = yang1[i]
        h_next = yang1[(i+1) % len(yang1)]
        hx = Hexagram(h_cur)
        lines.append(
            f"  h={h_cur:2d} ({h_cur:06b}) {hx.sym}#{hx.kw:>2} «{hx.name_ru}»"
        )
    lines += [
        "",
        "  Каждый переход = флип двух соседних битов в Q6.",
        "  Это не кратчайший путь — расстояние Хэмминга = 2.",
        "  В И-цзин: две меняющиеся черты одновременно.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_h(s: str) -> int:
    """Разбор h: число 0..63 или 'KW:N' или 6-битная строка."""
    s = s.strip()
    if s.startswith(("KW:", "kw:")):
        kw = int(s[3:])
        return _KW_TO_H[kw - 1]
    if len(s) == 6 and all(c in "01" for c in s):
        return int(s, 2)
    return int(s)


def main():
    parser = argparse.ArgumentParser(
        description="hexiching — И-цзин как Q6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexiching.py                   Особые гексаграммы
  python hexiching.py --trigrams        Таблица 8 триграмм
  python hexiching.py --hex 21          Гексаграмма h=21
  python hexiching.py --hex KW:63       Гексаграмма #63 по Вэнь-вану
  python hexiching.py --hex 010101      Гексаграмма по 6 битам
  python hexiching.py --table           Все 64 гексаграммы
  python hexiching.py --table --yang 3  Гексаграммы с ян=3 (20 штук)
  python hexiching.py --pairs           32 пары антиподов
  python hexiching.py --path 0 21       Путь 0→21 как меняющиеся черты
  python hexiching.py --cycle           6-цикл ян=1 (цикл творения)
        """,
    )
    parser.add_argument("--hex",     type=str,       metavar="H")
    parser.add_argument("--trigrams",action="store_true")
    parser.add_argument("--table",   action="store_true")
    parser.add_argument("--yang",    type=int,        metavar="N")
    parser.add_argument("--pairs",   action="store_true")
    parser.add_argument("--path",    type=str, nargs=2, metavar=("H1","H2"))
    parser.add_argument("--cycle",   action="store_true")
    parser.add_argument("--no-color",action="store_true")
    args = parser.parse_args()

    use_color = not args.no_color

    if args.hex:
        h = _parse_h(args.hex)
        print()
        print(Hexagram(h).describe(use_color))

    elif args.trigrams:
        print()
        print(trigram_table(use_color))

    elif args.table:
        print()
        print(hexagram_table(use_color, yang_filter=args.yang))

    elif args.pairs:
        print()
        print(antipode_pairs(use_color))

    elif args.path:
        h1 = _parse_h(args.path[0])
        h2 = _parse_h(args.path[1])
        path = shortest_path(h1, h2)
        print()
        print(path_as_iching(path, use_color))

    elif args.cycle:
        print()
        print(wuwang_cycle(use_color))

    else:
        print()
        print(show_special_hexagrams(use_color))


if __name__ == "__main__":
    main()
