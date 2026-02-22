# Архитектура монорепо Q6

## Обзор

```
libs/hexcore/          ← единственная общая зависимость
       │
       ├── neighbors(), hamming(), flip(), shortest_path()
       ├── gray_code(), antipode(), bfs_from(), describe()
       ├── yang_count(), yin_count(), to_bits(), from_bits()
       └── sphere(), ball(), orbit(), subcubes(), ...
       │
       ▼
projects/              ← 24 независимых проекта
  hexnav  hexca  hexpath  hexforth  karnaugh6  hexspec
  hexgraph  hexvis  hexcode  hexlearn  hexopt  hexring
  hexsym  hexnet  hexcrypt  hexstat  hexgeom  hexdim
  hexalg  hexphys  hexgf  hexmat  hexbio  hexlat
```

Каждый проект:
- импортирует только `libs.hexcore.hexcore`
- **не зависит** от других проектов
- запускается как CLI: `python3 projects/<name>/<name>.py <cmd>`

---

## hexcore — граф Q6

Граф Q6 — 6-мерный гиперкуб на 64 вершинах (гексаграммах).

```
Вершина:  6-битное число 0..63
Ребро:    изменение ровно 1 бита (расстояние Хэмминга = 1)
Степень:  каждая вершина имеет ровно 6 соседей
Диаметр:  6 (максимальное расстояние Хэмминга)
Рёбер:    192 (64 × 6 / 2)
```

Математические объекты, реализованные в hexcore:

| Функция | Описание |
|---------|----------|
| `neighbors(h)` | 6 соседей вершины h |
| `hamming(a, b)` | расстояние Хэмминга |
| `flip(h, bit)` | перевернуть один бит |
| `shortest_path(a, b)` | BFS кратчайший путь |
| `all_paths(a, b, max_len)` | все пути с ограничением длины |
| `antipode(h)` | противоположная вершина (d = 6) |
| `gray_code()` | гамильтонов путь (64 вершины) |
| `bfs_from(start)` | BFS-расстояния от start |
| `ball(center, r)` | шар Хэмминга радиуса r |
| `sphere(center, r)` | сфера Хэмминга |
| `yang_count(h)` | число сплошных черт |
| `yin_count(h)` | число прерывистых черт |
| `to_bits(h)` | 6-битная строка |
| `from_bits(bits)` | строка → вершина |
| `upper_trigram(h)` | верхняя триграмма (биты 3..5) |
| `lower_trigram(h)` | нижняя триграмма (биты 0..2) |
| `orbit(h, f)` | орбита действия f на h |
| `all_orbits(f)` | разбиение Q6 на орбиты |
| `subcubes(h, k)` | k-мерные подкубы через h |
| `describe(h)` | словарь: биты, ян, антипод, соседи |
| `render(h)` | ASCII-рисунок гексаграммы |
| `distance_spectrum(h)` | количество вершин на расстоянии 0..6 |

---

## Структура каждого проекта

```
projects/<name>/
├── <name>.py     — основной модуль + CLI (argparse)
├── README.md     — описание: что, зачем, как запустить
└── examples/     — (опционально) тестовые данные
```

Импорт hexcore из проекта:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from libs.hexcore.hexcore import neighbors, hamming, ...
```

---

## Карта проектов: Q6 как N угол зрения

```
Q6-граф
│
├── Языки и вычисления
│   ├── hexforth   — стековая машина: состояние = вершина Q6
│   └── hexspec    — конечный автомат: переходы = рёбра Q6
│
├── Игры и навигация
│   ├── hexpath    — абстрактная стратегия на Q6
│   ├── hexnav     — CLI навигатор по вершинам Q6
│   └── hexvis     — визуализация: ASCII / DOT / SVG
│
├── Математика: дискретные структуры
│   ├── hexgraph   — теория графов (спектр, подграфы, клики)
│   ├── hexgeom    — метрическая геометрия (шары, диаграммы Вороного)
│   ├── hexdim     — гиперкубическая структура (тессеракты, Q3×Q3)
│   ├── hexlat     — булева решётка B₆ (Мёбиус, антицепи)
│   └── hexsym     — группа автоморфизмов Aut(Q6) = S₆⋉(Z₂)⁶
│
├── Математика: алгебра
│   ├── hexgf      — поле Галуа GF(2⁶): умножение, BCH-коды
│   ├── hexmat     — линейная алгебра над GF(2), GL(6,2)
│   ├── hexalg     — гармонический анализ (WHT, граф Кэли, bent)
│   └── hexring    — булевы функции (WHT, ANF, коды Рида–Маллера)
│
├── Криптография и кодирование
│   ├── hexcrypt   — S-блоки, DDT, LAT, сеть Фейстеля
│   ├── hexcode    — двоичные линейные коды длины 6
│   └── karnaugh6  — минимизатор Куайна–МакКласки (6 переменных)
│
├── Машинное обучение и оптимизация
│   ├── hexlearn   — k-NN, k-медоиды, Байес, марковские цепи
│   └── hexopt     — SA, GA, TS, LocalSearch
│
├── Наука и симуляция
│   ├── hexca      — клеточный автомат (64 состояния)
│   ├── hexphys    — Изинг, MCMC, квантовые состояния
│   └── hexstat    — теория информации, случайные блуждания
│
└── Приложения к данным
    ├── hexbio     — биоинформатика (кодоны = вершины Q6)
    └── hexnet     — гиперкубическая сеть (маршрутизация, HPC)
```

---

## Тесты

```
tests/
├── test_hexcore.py      — 88 тестов: инварианты и функции hexcore
├── test_integration.py  — 40 тестов: импорт всех 24 + инварианты Q6
├── test_hexnav.py       — 46 тестов
├── test_hexca.py        — 56 тестов
├── test_hexpath.py      — 31 тест
├── test_hexpath_puzzle.py — 41 тест
├── test_hexforth.py     — 65 тестов
├── test_karnaugh6.py    — 44 теста
├── test_hexspec.py      — 49 тестов
... и ещё 18 файлов
```

Полный прогон: `make test` (1543 теста).
Smoke-тест CLI: `make smoke` (24/24 OK).

---

## Инварианты Q6 (проверяются в test_integration.py)

| Свойство | Значение |
|----------|----------|
| Вершин | 64 |
| Рёбер | 192 |
| Степень вершины | 6 |
| Диаметр | 6 |
| Двудольность | да (чётные/нечётные yang_count) |
| Гамильтонов путь | да (gray_code()) |
| hamming(a,b) == len(shortest_path(a,b))-1 | да |
| bfs_dist(a,b) == hamming(a,b) | да |
| antipode(antipode(h)) == h | да |

---

## Соглашения кода

1. **Без внешних зависимостей** — только stdlib Python 3.10+
2. **CLI через argparse** с подкомандами
3. **Типизация** — аннотации типов для публичного API
4. **Тесты** — только в классах `unittest.TestCase`
   (функции `test_*` на уровне модуля ломают pytest)
5. **Путь к hexcore** — через `sys.path.insert(0, '../../')`

Подробнее: [CONTRIBUTING.md](../CONTRIBUTING.md)
