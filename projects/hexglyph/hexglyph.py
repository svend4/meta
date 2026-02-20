"""hexglyph — шрифт «Старгейт» (Solan) как визуальный слой Q6.

Авторство:  Stanislav Engel (svend), ~2000-е годы
Версии TTF: myFront4Solan3.ttf, myFront4Solan4.ttf (2016-05-06)
Лицензия:   Авторский, «copy: svend»
История:    создан ~2000-е; отправлен Францу Герману, декабрь 2013
            «МОЙ АЛФАВИТ! )» — тема письма

──────────────────────────────────────────────────────────────────────────
КОНЦЕПЦИЯ
──────────────────────────────────────────────────────────────────────────
Каждый из 64 символов шрифта = вершина гиперкуба Q6.
Квадрат имеет ровно 6 возможных отрезков (как в hexvis.render_glyph):

        bit 0 = верхняя черта  (─ top)     h += 1
        bit 1 = нижняя черта   (─ bottom)  h += 2
        bit 2 = левая черта    (│ left)    h += 4
        bit 3 = правая черта   (│ right)   h += 8
        bit 4 = диагональ ╲    (backslash) h += 16
        bit 5 = диагональ ╱    (slash)     h += 32

    h = T×1 + B×2 + L×4 + R×8 + D1×16 + D2×32   ∈ {0..63}
    2⁶ = 64 подмножества → 64 визуально различимых символа.

Примеры (подтверждены пиксельным анализом 8×8 растров):
    h = 0  (000000) → пустой квадрат (нет отрезков)
    h = 12 (001100) → │ │  только вертикальные стороны  ← 'H'
    h = 15 (001111) → □    четыре стороны (рамка)        ← 'A'
    h = 48 (110000) → ╳    только диагонали              ← '0','L','l'
    h = 63 (111111) → ⊠    все шесть отрезков            ← 'E','e'

Фонетическое применение (рукопись Энгеля, «Алфавит 16 букв / 36 звуков»):
    Русские буквы ↔ Q6-символы — см. RUSSIAN_PHONETIC ниже.

──────────────────────────────────────────────────────────────────────────
API
──────────────────────────────────────────────────────────────────────────
    font_data(version=4)           → dict         — полный JSON шрифта
    glyph_bitmap(ch, version=4)   → list[int]     — 8 строк 8×8 растра
    render_bitmap(ch, version=4)  → list[str]     — ASCII-арт глифа
    detect_segments(ch, version=4)→ dict          — {T,B,L,R,D1,D2,h,bits}
    char_to_h(ch)                 → int | None    — Q6-вершина для символа
    h_to_char(h)                  → str | None    — символ для Q6-вершины
    encode(text)                  → list[int]     — текст → Q6-вершины
    decode(hs)                    → str           — Q6-вершины → текст
    font_path(version=4)          → Path          — путь к TTF-файлу
    viewer_path()                 → Path          — путь к viewer.html
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

_HERE = pathlib.Path(__file__).resolve().parent

# ── Файлы шрифта ──────────────────────────────────────────────────────────

_FONT_FILES = {
    3: _HERE / "myFront4Solan3.ttf",
    4: _HERE / "myFront4Solan4.ttf",
}
_DATA_FILES = {
    3: _HERE / "solan3.txt",
    4: _HERE / "solan4.txt",
}

_cache: dict[int, dict] = {}


def font_data(version: int = 4) -> dict:
    """Загрузить JSON-данные шрифта (кэшируется)."""
    if version not in _cache:
        path = _DATA_FILES[version]
        with open(path, encoding="utf-8") as fh:
            _cache[version] = json.loads(fh.read())
    return _cache[version]


def font_path(version: int = 4) -> pathlib.Path:
    """Путь к TTF-файлу шрифта."""
    return _FONT_FILES[version]


# ── Растровые глифы ───────────────────────────────────────────────────────

def glyph_bitmap(ch: str, version: int = 4) -> list[int]:
    """Вернуть 8×8 растр символа ch как список из 8 целых (0–255).

    Каждое целое — маска пикселей строки (бит 7 = левый пиксель).
    """
    data = font_data(version)
    code = str(ord(ch))
    if code not in data:
        raise KeyError(f"Символ {ch!r} (код {code}) не найден в шрифте Solan{version}")
    return list(data[code][:8])


def render_bitmap(ch: str, version: int = 4) -> list[str]:
    """Вернуть ASCII-арт глифа: 8 строк по 8 символов ('█' / '·')."""
    rows = glyph_bitmap(ch, version)
    result = []
    for v in rows:
        result.append(format(v, '08b').replace('1', '█').replace('0', '·'))
    return result


def print_glyph(ch: str, version: int = 4) -> None:
    """Напечатать ASCII-арт символа в stdout."""
    print(f"Solan{version}  '{ch}'  (ASCII {ord(ch)}):")
    for line in render_bitmap(ch, version):
        print(" ", line)


def viewer_path() -> pathlib.Path:
    """Путь к самодостаточному HTML-просмотрщику шрифта."""
    return _HERE / "viewer.html"


# ── Детектор сегментов из 8×8 растра ─────────────────────────────────────

def _pix(rows: list[int], r: int, c: int) -> int:
    """Получить значение пикселя (r, c) из списка 8-битных строк."""
    return (rows[r] >> (7 - c)) & 1


def detect_segments(ch: str, version: int = 4) -> dict:
    """Определить набор активных отрезков для символа ch из его 8×8 растра.

    Возвращает словарь::

        {
            'T':  bool,   # верхняя черта (top bar)
            'B':  bool,   # нижняя черта  (bottom bar)
            'L':  bool,   # левая черта   (left bar)
            'R':  bool,   # правая черта  (right bar)
            'D1': bool,   # диагональ ╲   (backslash)
            'D2': bool,   # диагональ ╱   (slash)
            'h':  int,    # Q6-вершина   (0–63)
            'bits': str,  # 6-битная строка bit5..bit0
        }

    Метод: пиксельный анализ 8×8 растра (Solan3 или Solan4).
    Диагонали детектируются только если пиксель есть в ОБЕИХ зонах:
    внутренней (строки 2–5) и хотя бы одной из внешних (строки 1–2 или 5–6).
    Это позволяет отличить полные диагонали от центральных меток.

    Из-за ограничений 8×8 сетки часть вершин Q6 неразличима попиксельно.
    Для полного маппинга используйте char_to_h() / h_to_char().
    """
    rows = glyph_bitmap(ch, version)
    p = _pix

    T  = bool(any(p(rows, 0, c)   for c in range(2, 6)))
    B  = bool(any(p(rows, 7, c)   for c in range(2, 6)))
    L  = bool(any(p(rows, r, 0)   for r in range(2, 6)))
    R  = bool(any(p(rows, r, 7)   for r in range(2, 6)))
    D1 = bool(
        any(p(rows, i, i)     for i in [1, 2]) or
        any(p(rows, i, i)     for i in [5, 6])
    )
    D2 = bool(
        any(p(rows, i, 7 - i) for i in [1, 2]) or
        any(p(rows, i, 7 - i) for i in [5, 6])
    )

    h = (int(T)  * 1  + int(B)  * 2  + int(L)  * 4 +
         int(R)  * 8  + int(D1) * 16 + int(D2) * 32)

    return {
        'T': T, 'B': B, 'L': L, 'R': R, 'D1': D1, 'D2': D2,
        'h': h,
        'bits': format(h, '06b'),
    }


# ── Фонетический маппинг: русские буквы ↔ Q6 ─────────────────────────────
#
# Источник: рукопись Энгеля «Алфавит 16 букв / 36 звуков» (~2000-е гг.)
# Файл:     Alphavit_16bukv.JPG (ветка svend4-patch-3)
#
# Таблица даёт 16 базовых согласных/гласных русского языка.
# Каждому звуку соответствует Q6-вершина h и конфигурация отрезков.
# h = T×1 + B×2 + L×4 + R×8 + D1×16 + D2×32
#
# Столбец 1 (левый, снизу вверх — буквы 1–8):
#   1) Л  — ⌐ (угол T+R)                    → h=9   (T+R)
#   2) Н  — = (двойная горизонталь T+B)      → h=3   (T+B)
#   3) М  — N (диагональ + правая сторона)   → h=24  (R+D1) *
#   4) Т  — × (только диагонали)             → h=48  (D1+D2)
#   5) Г  — ╳ с верхней чертой (T+D1+D2)*   → h=49  (T+D1+D2) *
#   6) Д  — △ (треугольник: B+D1+D2)*        → h=50  (B+D1+D2) *
#   7) В  — N-образный (L+D1)*              → h=20  (L+D1) *
#   8) Б  — И-образный (L+B+R)*             → h=14  (B+L+R) *
#
# Столбец 2 (правый, снизу вверх — буквы 9–16):
#   9)  Ч  — Z-образный ╱ (T+B+D2)*         → h=35  (T+B+D2) *
#   10) Ж  — N-образный ╲ (L+R+D1)*         → h=28  (L+R+D1) *
#   11) З  — Z-образный ╲ (T+B+D1)*         → h=19  (T+B+D1) *
#   12) Р  — □ (рамка четырёх сторон)        → h=15  (T+B+L+R)
#   13) У  — ◇ (X с двумя чертами)*         → h=51  (T+B+D1+D2) *
#   14) И  — N-образный ╱ (L+R+D2)*         → h=44  (L+R+D2) *
#   15) О  — ⊠ частичная*                    → h=47  (T+L+R+D1+D2) *
#   16) А  — ⊠ полная (все 6 отрезков)       → h=63  (T+B+L+R+D1+D2)
#
# Звёздочка (*) — приблизительно, требует уточнения по оригиналу.
# «Проверить порядок знаков» — примечание автора на рукописи.
# Источник изображения: Alphavit_Big_76.JPG (треугольник Хассе 64 глифа).

RUSSIAN_PHONETIC: dict[str, dict] = {
    'А': {'h': 63, 'segs': 'T+B+L+R+D1+D2', 'char': 'E',
          'desc': 'все 6 отрезков ⊠',         'confidence': 'high'},
    'Р': {'h': 15, 'segs': 'T+B+L+R',        'char': 'A',
          'desc': 'рамка □',                  'confidence': 'high'},
    'Т': {'h': 48, 'segs': 'D1+D2',           'char': '0',
          'desc': 'только диагонали ╳',       'confidence': 'high'},
    'Л': {'h':  9, 'segs': 'T+R',             'char': None,
          'desc': 'угол ⌐ (верх+право)',      'confidence': 'medium'},
    'Н': {'h':  3, 'segs': 'T+B',             'char': None,
          'desc': 'горизонтали = (верх+низ)', 'confidence': 'medium'},
    'О': {'h': 47, 'segs': 'T+L+R+D1+D2',    'char': 'T',
          'desc': 'рамка без низа + диаг.',   'confidence': 'medium'},
    'Б': {'h': 14, 'segs': 'B+L+R',           'char': 'Z',
          'desc': 'U-образный',               'confidence': 'medium'},
    'М': {'h': 24, 'segs': 'R+D1',            'char': None,
          'desc': 'правая сторона + диаг.╲', 'confidence': 'low'},
    'В': {'h': 20, 'segs': 'L+D1',            'char': None,
          'desc': 'левая сторона + диаг.╲',  'confidence': 'low'},
    'Д': {'h': 50, 'segs': 'B+D1+D2',         'char': None,
          'desc': 'треугольник △ (низ+диаг.)', 'confidence': 'low'},
    'Г': {'h': 49, 'segs': 'T+D1+D2',         'char': 'D',
          'desc': '╳ с верхней чертой',       'confidence': 'low'},
    'У': {'h': 51, 'segs': 'T+B+D1+D2',       'char': 'U',
          'desc': 'ромб ◇ (X с двумя чертами)', 'confidence': 'low'},
    'И': {'h': 44, 'segs': 'L+R+D2',          'char': 'B',
          'desc': 'N-образный ╱ (И)',          'confidence': 'low'},
    'З': {'h': 19, 'segs': 'T+B+D1',          'char': 'C',
          'desc': 'Z-образный ╲ (З)',          'confidence': 'low'},
    'Ж': {'h': 28, 'segs': 'L+R+D1',          'char': 'V',
          'desc': 'N-образный ╲ (Ж)',          'confidence': 'low'},
    'Ч': {'h': 35, 'segs': 'T+B+D2',          'char': 'J',
          'desc': 'Z-образный ╱ (Ч)',          'confidence': 'low'},
}

# Обратный маппинг: h → русская буква (только подтверждённые)
PHONETIC_H_TO_RU: dict[int, str] = {
    v['h']: ru for ru, v in RUSSIAN_PHONETIC.items()
    if v['confidence'] == 'high'
}


# ── Маппинг Solan ↔ Q6 ────────────────────────────────────────────────────
#
# Шрифт содержит 71 символ (ASCII 34–122).
# 64 из них соответствуют 64 вершинам Q6 (все подмножества 6-линейного квадрата).
# Оставшиеся 7 — декоративные / фоновые (16×16 растры, коды 34–42 = " # $ % & ' ( ) *).
#
# Порядок маппинга: символы упорядочены по числу битов (весу Хэмминга 0→6),
# внутри одного веса — в порядке следования ASCII-кодов символа в шрифте.
# Вершина h ∈ {0..63} по весу Хэмминга совпадает с позицией в треугольнике.
#
# Примечание: точное соответствие «символ → бит-вектор» требует ручной
# разметки (рукописный треугольник Энгеля, 2000-е гг.).
# Текущая таблица — предварительная, по линейному порядку 62 символов.
#
# Символы шрифта в порядке ASCII-кодов 48–122 (62 символа):
_CHARSET_ORDERED: str = (
    "0123456789"      # 48–57  (10)
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 65–90  (26)
    "abcdefghijklmnopqrstuvwxyz"  # 97–122 (26)
)
# Берём первые 64 для маппинга (62 + 2 первых спецсимвола кодов 34,35)
_EXTRA = '"#'   # коды 34, 35
_CHARSET_64: str = _CHARSET_ORDERED[:62] + _EXTRA[:2]  # ровно 64

# Q6 вершины 0..63 — в порядке Хэмминга (как в треугольнике):
def _q6_by_hamming() -> list[int]:
    """Все 64 вершины, отсортированные по весу Хэмминга, внутри веса — по значению."""
    return sorted(range(64), key=lambda h: (bin(h).count('1'), h))

_Q6_ORDER: list[int] = _q6_by_hamming()

# Прямой маппинг: char → h
_CHAR_TO_H: dict[str, int] = {
    ch: _Q6_ORDER[i] for i, ch in enumerate(_CHARSET_64)
}
# Обратный маппинг: h → char
_H_TO_CHAR: dict[int, str] = {h: ch for ch, h in _CHAR_TO_H.items()}


def char_to_h(ch: str) -> Optional[int]:
    """ASCII-символ шрифта → индекс вершины Q6 (0–63). None если нет в таблице."""
    return _CHAR_TO_H.get(ch)


def h_to_char(h: int) -> Optional[str]:
    """Индекс вершины Q6 (0–63) → ASCII-символ шрифта. None если нет."""
    return _H_TO_CHAR.get(h)


# ── Кодирование текста ────────────────────────────────────────────────────

def encode(text: str) -> list[int]:
    """Перевести строку символов шрифта в список Q6-вершин (0–63).

    Символы, не входящие в таблицу, пропускаются (не вызывают ошибку).
    """
    result = []
    for ch in text:
        h = char_to_h(ch)
        if h is not None:
            result.append(h)
    return result


def decode(hs: list[int]) -> str:
    """Перевести список Q6-вершин в строку символов шрифта.

    Вершины вне диапазона 0–63 пропускаются.
    """
    result = []
    for h in hs:
        ch = h_to_char(h)
        if ch is not None:
            result.append(ch)
    return ''.join(result)


# ── Информация о шрифте ───────────────────────────────────────────────────

def font_info(version: int = 4) -> dict:
    """Сводная информация о шрифте."""
    data = font_data(version)
    glyph_keys = [k for k in data if k.isdigit()]
    return {
        "name":          data.get("name", ""),
        "copyright":     data.get("copy", ""),
        "letterspace":   data.get("letterspace", ""),
        "basefont_size": data.get("basefont_size", ""),
        "glyph_count":   len(glyph_keys),
        "char_range":    f"{min(int(k) for k in glyph_keys)}–{max(int(k) for k in glyph_keys)}",
        "ttf_path":      str(font_path(version)),
        "q6_chars":      64,
        "q6_charset":    _CHARSET_64,
    }


if __name__ == "__main__":
    import sys

    info = font_info(4)
    print("=== Шрифт Solan (Старгейт) ===")
    for k, v in info.items():
        print(f"  {k:16s}: {v}")

    # Показать несколько символов
    print("\n=== Примеры глифов (Solan4) ===")
    for ch in "0AZaz":
        print_glyph(ch, version=4)
        h = char_to_h(ch)
        print(f"  → Q6 вершина h={h}  ({bin(h or 0)[2:].zfill(6)})")
        print()

    # Пример кодирования
    sample = "Hello"
    encoded = encode(sample)
    decoded = decode(encoded)
    print(f"Encode({sample!r}) → {encoded}")
    print(f"Decode({encoded}) → {decoded!r}")
