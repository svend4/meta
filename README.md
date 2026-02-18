# meta — монорепо hex-проектов

Монорепо для экосистемы проектов на основе **граф Q6** (система гексаграмм).

64 гексаграммы образуют 6-мерный гиперкуб: каждый узел — гексаграмма,
каждое ребро — изменение одной черты. Эта структура является основой
для всех проектов репозитория.

---

## Структура

```
meta/
├── libs/
│   └── hexcore/       — общая библиотека графа Q6 (Python)
├── projects/
│   ├── hexnav/        — интерактивный навигатор по Q6
│   ├── hexca/         — клеточный автомат (64 состояния)
│   ├── hexpath/       — абстрактная стратегическая игра
│   ├── hexforth/      — стековый язык в духе Forth
│   ├── karnaugh6/     — карта Карно для 6 переменных
│   └── hexspec/       — язык спецификации протоколов
├── docs/
│   └── q6-math.md     — математические основы графа Q6
├── flower_shop.py     — пример CLI-приложения (не hex-проект)
└── README.md
```

---

## Проекты

| Проект | Область | Ключевые файлы | Статус |
|---|---|---|---|
| [hexnav](projects/hexnav/) | Инструмент / навигация | `hexnav.py` | реализован |
| [hexca](projects/hexca/) | Наука / симуляция | `hexca.py`, `rules.py` | реализован |
| [hexpath](projects/hexpath/) | Игры | `game.py`, `cli.py` | реализован |
| [hexforth](projects/hexforth/) | Языки программирования | `interpreter.py`, `compiler.py` | реализован |
| [karnaugh6](projects/karnaugh6/) | Образование / электроника | `minimize.py` | реализован |
| [hexspec](projects/hexspec/) | Формальные методы | `verifier.py`, `generator.py` | реализован |

### Быстрый старт

```bash
# Навигатор по Q6 (интерактивный)
python3 projects/hexnav/hexnav.py 42

# Клеточный автомат (1D и 2D)
python3 projects/hexca/hexca.py --rule xor_rule --steps 12
python3 projects/hexca/hexca.py --mode 2d --rule conway_b3s23 --width 40 --height 15 --steps 5

# Стратегическая игра на Q6 (человек vs AI)
python3 projects/hexpath/cli.py

# Минимизатор булевых функций (6 переменных, Куайн–МакКласки)
python3 projects/karnaugh6/minimize.py 0 1 2 3 4 5 6 7 --table

# HexForth: запуск программы
python3 projects/hexforth/interpreter.py projects/hexforth/examples/hello.hf
# HexForth: компиляция в Python
python3 projects/hexforth/compiler.py projects/hexforth/examples/hello.hf --target python

# HexSpec: верификация автомата
python3 projects/hexspec/verifier.py projects/hexspec/examples/tcp.json
# HexSpec: генерация тестовых сценариев
python3 projects/hexspec/generator.py projects/hexspec/examples/tcp.json --coverage all
python3 projects/hexspec/generator.py projects/hexspec/examples/tcp.json --format hexforth
```

---

## Общая библиотека: hexcore

Все проекты используют `libs/hexcore` как общее ядро.
Она реализует граф Q6 без внешних зависимостей (только стандартная библиотека Python).

```python
# Пример использования из любого проекта
import sys
sys.path.insert(0, '../../')  # или настроить PYTHONPATH

from libs.hexcore.hexcore import neighbors, shortest_path, render

h = 42
print(f"Гексаграмма {h}:")
print(render(h))
print(f"Соседи: {neighbors(h)}")
print(f"Путь 0→42: {shortest_path(0, 42)}")
```

Запуск демо:

```bash
python3 libs/hexcore/hexcore.py
```

---

## Математика: что такое Q6

```
64 гексаграммы = 2^6 вершин 6-мерного гиперкуба Q6
Каждая вершина: 6-битное число 0..63
Каждое ребро:   изменение ровно 1 бита (расстояние Хэмминга = 1)
Степень каждой вершины: ровно 6
Диаметр графа: 6 (максимальный путь = переворот всех 6 черт)
```

Граф Q6 — это граф Кэли группы (Z₂)⁶. Он появляется в:
- теории кодирования (коды Хэмминга, коды Грея)
- квантовых вычислениях (ворота Паули-X на 6 кубитах)
- синтезе цифровых схем (карты Карно)
- теории автоматов и формальных языков

Подробнее: [docs/q6-math.md](docs/q6-math.md)

---

## С чего начать

1. Прочитать `libs/hexcore/README.md` — понять ядро
2. Запустить `python3 libs/hexcore/hexcore.py` — увидеть граф в действии
3. Запустить `python3 projects/hexnav/hexnav.py` — интерактивно исследовать Q6
4. Выбрать проект из таблицы выше и изучить его README

---

## Агент и модель

- Агент: Claude Code (Anthropic)
- Модель: claude-sonnet-4-6
- Назначение: разработка и сопровождение проектов монорепо
