"""
hexpractice.py — Ежедневный практический синтез Q6

Объединяет все четыре системы в конкретные рекомендации на основе
текущей государевой гексаграммы (дата → период года).

Алгоритм:
  1. Дата → государева гексаграмма (hexsovereign)
  2. Активные зоны → что задействовать (hexboya / Крюков)
  3. Активные цвета → на что ориентироваться (hexliuxing / Лю-Синь)
  4. Объём куба → контекст пространства (hexkub / Касаткин)
  5. И-цзин → основная тема периода (hexiching)
  6. «Граничная» зона → фокус практики (переход)
  7. Рекомендации по движению для каждой зоны

Философия:
  Открытые зоны = янская активность → прорабатывать, укреплять
  Закрытые зоны = иньский покой    → восстанавливать, наполнять
  «Порог» (следующая зона)         → точка роста, фокус внимания
"""

import sys
import os
import argparse
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, antipode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexliuxing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexboya"))

from hexiching import Hexagram
from hexliuxing import ELEMENTS
from hexboya import ZONE_NAMES, BodyState

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
ITAL  = "\033[3m"

YANG_ANSI = {
    0: "\033[90m", 1: "\033[94m", 2: "\033[96m",
    3: "\033[92m", 4: "\033[93m", 5: "\033[95m", 6: "\033[97m",
}
ELEM_COLORS = ["\033[31m","\033[33m","\033[32m","\033[36m","\033[34m","\033[35m"]
PHASE_ANSI  = {
    "зима":   "\033[90m",
    "весна":  "\033[92m",
    "лето":   "\033[93m",
    "осень":  "\033[95m",
}

# ---------------------------------------------------------------------------
# Смысл каждой зоны (тело + движение + качество)
# ---------------------------------------------------------------------------

ZONE_BODY = {
    "ВЛ": ("Голова, корона",         "Медитация, вращение головы, взгляд"),
    "ВП": ("Шея, плечи",             "Вращение плеч, растяжка шеи, раскрытие"),
    "СЛ": ("Грудь, сердце",          "Дыхание грудью, раскрытие грудной клетки"),
    "СП": ("Живот, печень",          "Скручивания, дыхание животом, массаж"),
    "НЛ": ("Бёдра, паховая область", "Приседания, выпады, раскрытие тазобедренных"),
    "НП": ("Стопы, земля",           "Стояние на одной ноге, ходьба осознанная"),
}

ELEM_PRACTICE = {
    "R": ("Огонь",    "Динамические разогревающие движения, ускорение"),
    "Y": ("Земля",    "Устойчивые статические позы, заземление, центр"),
    "G": ("Дерево",   "Вертикальные растяжки, рост вверх, дерево-асаны"),
    "C": ("Вода",     "Плавные текущие движения, волновое тело"),
    "B": ("Металл",   "Чёткие усилия, сжатия, выверенная точность"),
    "M": ("Эфир",     "Трансформации, переходы между состояниями"),
}

YANG_PRACTICE = {
    0: "Глубокий покой. Только дыхание. Ничего не делать — слушать пустоту.",
    1: "Пробуждение. Один мягкий импульс. Заметить первое желание движения.",
    2: "Первые движения. Мягко, исследуя. Два потока начинают взаимодействие.",
    3: "Равновесие. Ян и инь в балансе. Это самое устойчивое состояние.",
    4: "Полнота. Четыре потока активны. Тело готово к сложным практикам.",
    5: "Почти пик. Огромная энергия. Требует внимательности и заземления.",
    6: "Пик ян. Полная яркость. Максимальное усилие и максимальная отдача.",
}

ICHING_THEME = {
    0:  "Потенциал. В покое заключена вся возможность.",
    1:  "Возврат. Первый импульс жизни после тишины.",
    2:  "Приближение. Энергия нарастает, открывается.",
    3:  "Мир. Небо и земля в гармонии, всё расцветает.",
    4:  "Великая мощь. Сила требует дисциплины.",
    5:  "Решимость. Прорыв — одна черта до полноты.",
    6:  "Творчество. Чистое ян — действуй, создавай.",
    7:  "Встреча. Инь возвращается, первая тень.",
    8:  "Отступление. Мудрое отступление сохраняет силу.",
    9:  "Застой. Небо и земля не общаются — ждать.",
    10: "Созерцание. Наблюдать издали, копить понимание.",
    11: "Распад. Уступить — значит сохранить суть.",
}

SOV_H = [0, 1, 3, 7, 15, 31, 63, 62, 60, 56, 48, 32]
SOV_SEASON = {
    0: "зима", 1: "зима", 2: "весна", 3: "весна",
    4: "весна", 5: "весна", 6: "лето", 7: "лето",
    8: "лето",  9: "осень", 10: "осень", 11: "осень",
}


# ---------------------------------------------------------------------------
# Определение суверенной гексаграммы по дате
# ---------------------------------------------------------------------------

def _sov_index(d: date) -> tuple:
    """Возвращает (индекс 0-11, pct 0-100) для даты d."""
    from projects.hexiching.hexsovereign import _date_to_sovereign
    idx, pct, h = _date_to_sovereign(d)
    return idx, pct


# ---------------------------------------------------------------------------
# Полная практика
# ---------------------------------------------------------------------------

def show_practice(d: date = None, use_color: bool = True) -> str:
    if d is None:
        d = date.today()

    try:
        idx, pct = _sov_index(d)
    except Exception:
        # fallback: оцениваем день года
        yday = d.timetuple().tm_yday
        idx  = (yday * 12 // 365) % 12
        pct  = 50

    h       = SOV_H[idx]
    hx      = Hexagram(h)
    season  = SOV_SEASON[idx]
    next_idx = (idx + 1) % 12
    next_h   = SOV_H[next_idx]
    next_hx  = Hexagram(next_h)

    # Что меняется при переходе
    diff_bit  = (h ^ next_h).bit_length() - 1
    opening   = (next_h >> diff_bit) & 1
    thresh_zn = ZONE_NAMES[diff_bit]
    thresh_el = ELEMENTS[diff_bit]

    # Активные зоны и цвета
    active_zones  = [ZONE_NAMES[i] for i in range(6) if (h >> i) & 1]
    passive_zones = [ZONE_NAMES[i] for i in range(6) if not (h >> i) & 1]
    active_elems  = [ELEMENTS[i] for i in range(6) if (h >> i) & 1]

    bold  = BOLD if use_color else ""
    r     = RESET if use_color else ""
    dim   = DIM if use_color else ""
    ital  = ITAL if use_color else ""
    sc    = PHASE_ANSI.get(season, "") if use_color else ""
    yc    = YANG_ANSI[hx.yang] if use_color else ""
    ec_th = ELEM_COLORS[diff_bit] if use_color else ""

    bar_len = 24
    filled  = round(pct * bar_len / 100)
    bar     = "█" * filled + "░" * (bar_len - filled)

    # Антипод (противоположный сезон)
    ap_h  = antipode(h)
    ap_hx = Hexagram(ap_h)
    ap_idx = (idx + 6) % 12

    lines = [
        "",
        f"{'═'*66}",
        f"  {bold}ПРАКТИКА Q6 · {d.strftime('%d.%m.%Y')}{r}",
        f"{'═'*66}",
        "",
        # Суверенная гексаграмма
        f"  {bold}Государева гексаграмма:{r}",
        f"    {yc}{hx.sym}  {hx.kw:>2} «{hx.name_ru}»  ({hx.name_pin}){r}",
        f"    h={h} ({h:06b})  ян={yc}{hx.yang}{r}  {sc}{season}{r}",
        f"    [{yc}{bar}{r}] {pct}% месяца",
        "",
        # Тема периода
        f"  {bold}Тема периода:{r}",
        f"    {ital}{ICHING_THEME.get(idx, hx.name_ru)}{r}",
        f"    {dim}(И-цзин #{hx.kw}: {hx.name_ru}){r}",
        "",
    ]

    # Тело (Крюков)
    lines += [
        f"  {bold}Тело (система Крюкова):{r}",
    ]
    if active_zones:
        for zn in active_zones:
            body, movement = ZONE_BODY[zn]
            i = next(i for i in range(6) if ZONE_NAMES[i] == zn)
            ec = ELEM_COLORS[i] if use_color else ""
            lines.append(f"    {ec}● {zn}{r}  {body}")
            lines.append(f"       Движение: {movement}")
    else:
        lines.append(f"    {dim}∅ Все зоны в покое (Кунь){r}")

    if passive_zones:
        pz_str = ", ".join(passive_zones)
        lines.append(f"    {dim}○ Пассивные зоны: {pz_str}{r}")

    lines.append("")

    # Цвета (Лю-Синь)
    lines += [
        f"  {bold}Стихии (Лю-Синь):{r}",
    ]
    if active_elems:
        for elem in active_elems:
            short = elem['short']
            prac = ELEM_PRACTICE.get(short, ("", elem['desc']))
            ec_e = ELEM_COLORS[next(i for i in range(6) if ELEMENTS[i]['short'] == short)]
            ec_e = ec_e if use_color else ""
            lines.append(
                f"    {ec_e}■ {short} {elem['name']}{r}  {prac[0]}"
            )
            lines.append(f"       {prac[1]}")
    else:
        lines.append(f"    {dim}∅ (Кунь — все стихии в потенции){r}")

    lines.append("")

    # Фокус практики (граничная зона)
    action_word = "откроется" if opening else "закроется"
    lines += [
        f"  {bold}Фокус практики (порог):{r}",
        f"    Следующий переход: {ec_th}{action_word} зона {thresh_zn}{r}",
        f"    Стихия: {ec_th}{thresh_el['short']} — {thresh_el['name']}{r}",
        f"    {ec_th}{thresh_el['desc']}{r}",
        f"",
        f"    Следующая гексаграмма: {next_hx.sym}#{next_hx.kw} «{next_hx.name_ru}»  ян={next_hx.yang}",
        "",
    ]

    # Рекомендация по ян-уровню
    lines += [
        f"  {bold}Качество энергии (ян={hx.yang}):{r}",
        f"    {YANG_PRACTICE.get(hx.yang, '')}",
        "",
    ]

    # Антипод (противоположный сезон)
    ap_season = SOV_SEASON[ap_idx]
    ap_sc = PHASE_ANSI.get(ap_season, "") if use_color else ""
    lines += [
        f"  {bold}Антипод периода ({ap_sc}{ap_season}{r}):{r}",
        f"    {ap_hx.sym}#{ap_hx.kw} «{ap_hx.name_ru}»  ян={ap_hx.yang}",
        f"    {dim}(через 6 месяцев — противоположное состояние системы){r}",
        "",
    ]

    # Касаткин (куб)
    x = h & 3; y = (h >> 2) & 3; z = (h >> 4) & 3
    vol = (x+1)*(y+1)*(z+1)
    lines += [
        f"  {bold}Пространство (Касаткин):{r}",
        f"    Координаты: ({x},{y},{z})  Объём V={vol}",
        f"    Смысл: пространство практики {'стремительно расширяется' if vol <= 8 else 'достигает полноты' if vol >= 32 else 'в активном развитии'}.",
        "",
    ]

    # Итоговый текст практики
    open_str  = "+".join(e['short'] for e in active_elems) or "∅"
    lines += [
        f"  {'─'*62}",
        f"  {bold}СИНТЕЗ:{r}",
        f"  В период {hx.sym} «{hx.name_ru}» ({sc}{season}{r}) система Q6 находится в состоянии",
        f"  h={h} с {hx.yang} активными ян-потоками [{open_str}].",
        f"  Зоны тела [{','.join(active_zones) or '∅'}] требуют внимания.",
        f"  Готовься к: {ec_th}{action_word} зону {thresh_zn} ({thresh_el['name']}){r}.",
        f"  {'─'*62}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Сезонный обзор
# ---------------------------------------------------------------------------

def show_season_overview(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    lines = [
        f"\n{bold}Сезонный обзор — 12 практических периодов{r}\n",
        f"  {'#':>2}  {'ян':>3}  {'Гексаграмма':<20}  {'Сезон':<7}  "
        f"{'Зоны':<18}  Фокус",
        "  " + "─"*75,
    ]

    for idx, h in enumerate(SOV_H):
        hx    = Hexagram(h)
        yc    = YANG_ANSI[hx.yang] if use_color else ""
        season = SOV_SEASON[idx]
        sc    = PHASE_ANSI.get(season, "") if use_color else ""
        az    = ",".join(ZONE_NAMES[i] for i in range(6) if (h >> i) & 1) or "∅"

        next_h    = SOV_H[(idx+1)%12]
        diff_bit  = (h ^ next_h).bit_length() - 1
        opening   = (next_h >> diff_bit) & 1
        arrow     = "▶" if opening else "◀"
        ec        = ELEM_COLORS[diff_bit] if use_color else ""
        thresh    = ZONE_NAMES[diff_bit]

        lines.append(
            f"  {idx+1:>2}  {yc}{hx.yang:>3}{r}  {hx.sym}{yc}{hx.name_pin[:16]:<16}{r}"
            f"  {sc}{season:<7}{r}  {az:<18}"
            f"  {ec}{arrow}{thresh}{r}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexpractice — ежедневный практический синтез Q6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexpractice.py              Практика на сегодня
  python hexpractice.py --date 2026-06-15  На конкретную дату
  python hexpractice.py --overview   Обзор 12 периодов
        """,
    )
    parser.add_argument("--date",     type=str, metavar="YYYY-MM-DD")
    parser.add_argument("--overview", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.overview:
        print(show_season_overview(use_color))
    else:
        d = date.fromisoformat(args.date) if args.date else date.today()
        print(show_practice(d, use_color))


if __name__ == "__main__":
    main()
