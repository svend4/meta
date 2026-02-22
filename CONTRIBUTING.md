# Contributing — как добавить новый проект в экосистему Q6

## Предпосылки

Прочитайте `libs/hexcore/README.md` и `docs/q6-math.md`.
Убедитесь, что ваш проект использует граф Q6 как структурную основу,
а не просто импортирует несколько функций из hexcore.

---

## Структура нового проекта

```
projects/your_project/
├── your_project.py   — основной модуль (логика + CLI через argparse)
├── README.md         — описание проекта (см. шаблон ниже)
└── examples/         — (опционально) примеры данных, конфиги
```

### Минимальный шаблон модуля

```python
#!/usr/bin/env python3
"""
your_project — описание в одно предложение.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from libs.hexcore.hexcore import neighbors, hamming, shortest_path


# ── Логика ────────────────────────────────────────────────────────────────

def your_function(hexagram: int) -> ...:
    """Документация."""
    ...


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description='your_project — краткое описание')
    sub = p.add_subparsers(dest='cmd', required=True)

    cmd = sub.add_parser('info', help='показать информацию')
    cmd.add_argument('hexagram', type=int, help='гексаграмма 0..63')

    args = p.parse_args()
    if args.cmd == 'info':
        result = your_function(args.hexagram)
        print(result)


if __name__ == '__main__':
    main()
```

### Шаблон README.md

```markdown
# your_project

Одно предложение: что это и как связано с Q6.

## Что это

Подробное описание (3-5 абзацев).

## Для чего применять

- Конкретный сценарий 1
- Конкретный сценарий 2

## Запуск

\```bash
python3 projects/your_project/your_project.py info 42
\```

## API

\```python
from projects.your_project.your_project import your_function
result = your_function(42)
\```

## Зависимости

Только стандартная библиотека Python + hexcore (уже включён).
```

---

## Тесты

Добавьте файл `tests/test_your_project.py`. Минимум 20 тестов.

```python
"""Тесты для your_project."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from projects.your_project.your_project import your_function


class TestYourFunction(unittest.TestCase):

    def test_basic(self):
        result = your_function(0)
        self.assertIsNotNone(result)

    def test_all_hexagrams(self):
        """Функция должна работать для всех 64 гексаграмм."""
        for h in range(64):
            with self.subTest(h=h):
                result = your_function(h)
                self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

Проверьте, что тесты проходят:

```bash
python3 -m pytest tests/test_your_project.py -v
make test        # полный прогон
```

---

## Чеклист перед PR

- [ ] Код работает без внешних зависимостей (только stdlib + hexcore)
- [ ] Есть CLI с `argparse`, запускается через `python3 projects/.../....py`
- [ ] Есть `README.md` по шаблону выше
- [ ] Есть тест-файл `tests/test_....py` с ≥ 20 тестами
- [ ] `make test` показывает `0 failed`
- [ ] `make lint` показывает `OK: синтаксических ошибок не найдено`
- [ ] `docs/projects-overview.md` дополнен описанием нового проекта

---

## Правила кода

1. **Без внешних зависимостей** — только стандартная библиотека Python 3.10+.
2. **Типизация** — аннотации типов для публичных функций.
3. **Связь с Q6** — проект должен использовать топологию Q6 (не просто вызывать `hamming`).
4. **CLI** — каждый проект запускается как `python3 projects/.../.py <subcommand>`.
5. **Тесты** — функции с именем `test_*` на уровне модуля запрещены (это ломает pytest).
   Используйте классы `unittest.TestCase`.

---

## Запуск окружения

```bash
# Python 3.10+, зависимостей нет
python3 --version

# Убедитесь, что базовые тесты проходят
make test

# Интеграционные тесты (импорт всех проектов + инварианты hexcore)
python3 -m pytest tests/test_integration.py -v
```
