"""hexglyph — шрифт «Старгейт» (Solan) как визуальный слой Q6.

Авторство:  Stanislav Engel (svend), ~2000-е годы
Версии TTF: myFront4Solan3.ttf, myFront4Solan4.ttf (2016-05-06)
Лицензия:   Авторский, «copy: svend»

──────────────────────────────────────────────────────────────────────────
КОНЦЕПЦИЯ
──────────────────────────────────────────────────────────────────────────
Каждый из 64 символов шрифта = вершина гиперкуба Q6.
Квадрат имеет ровно 6 возможных отрезков:

        bit 0 = верхняя черта  (─ top)
        bit 1 = нижняя черта   (─ bottom)
        bit 2 = левая черта    (│ left)
        bit 3 = правая черта   (│ right)
        bit 4 = диагональ ╲    (\\ backslash)
        bit 5 = диагональ ╱    (/ slash)

    2⁶ = 64 подмножества → 64 визуально различимых символа.

Это та же битовая карта что в hexvis.render_glyph(h):
    h = 0  (000000) → пустой квадрат
    h = 63 (111111) → полный квадрат (⊠)

Шрифт хранит символы как 8×8 пиксельные растровые глифы,
закодированные в JSON (solan3.txt / solan4.txt) и как TTF-файлы.

Формат JSON-файлов шрифта:
    { "ASCII_code": [row0, row1, ..., row15], ...,
      "name": "myFront4Solan3",  "copy": "svend",
      "letterspace": "64",       "basefont_size": "512" }
    Каждый row — 8-битное целое (маска пикселей строки 8×8).
    Первые 8 строк — сам глиф; строки 8–15 = нули (запас).

──────────────────────────────────────────────────────────────────────────
API
──────────────────────────────────────────────────────────────────────────
    font_data(version=4)          → dict  — полный JSON шрифта
    glyph_bitmap(ch, version=4)  → list[int]  — 8 строк 8×8 растра
    render_bitmap(ch, version=4) → list[str]  — ASCII-арт глифа
    char_to_h(ch)                → int | None — Q6-вершина для символа
    h_to_char(h)                 → str | None — символ для Q6-вершины
    encode(text)                 → list[int]  — текст → Q6-вершины
    decode(hs)                   → str        — Q6-вершины → текст
    font_path(version=4)         → Path — путь к TTF-файлу
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
