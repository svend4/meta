# hexvis — Визуализация Q6

ASCII-арт, DOT/Graphviz, SVG-диаграммы для гексаграмм, путей и подграфов Q6.

## Статус

Реализован

## Форматы вывода

| Формат | Описание |
|--------|----------|
| `ascii` | Текстовая решётка 8×8 всех 64 гексаграмм с ANSI-цветом |
| `path` | Последовательность гексаграмм пути с символами |
| `dot` | Graphviz DOT для подграфа / пути |
| `svg` | SVG-диаграмма пути или подграфа |

## Цветовой код (ANSI)

```
yang=0 → тёмно-серый   (#  инь)
yang=1 → синий
yang=2 → голубой
yang=3 → зелёный
yang=4 → оранжевый
yang=5 → красный
yang=6 → ярко-жёлтый   (## янь)
```

## Примеры

```bash
# ASCII-решётка всех 64 гексаграмм (8×8)
python3 projects/hexvis/hexvis.py ascii

# Путь между гексаграммами 0 и 63
python3 projects/hexvis/hexvis.py path 0 63

# Graphviz DOT для подграфа (первые 8 вершин)
python3 projects/hexvis/hexvis.py dot 0 1 2 3 4 5 6 7
```

## Стек

Python 3.10+ (stdlib only)

## Зависимости

- `libs/hexcore`

## Запуск

```bash
python3 projects/hexvis/hexvis.py
```

## Структура

```
hexvis/
├── hexvis.py   — ASCII, DOT, SVG рендереры для Q6
└── README.md
```

## Связанные проекты

- `hexnav` — интерактивная навигация (использует hexvis для отображения)
- `hexgraph` — подграфы для визуализации
- `hexca` — анимация клеточного автомата
