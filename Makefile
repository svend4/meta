.PHONY: test test-v test-file smoke lint clean help

PYTHON := python3
PYTEST  := $(PYTHON) -m pytest

# ─────────────────────────────────────────────────────────────
help:
	@echo "Цели:"
	@echo "  make test           — все тесты (краткий вывод)"
	@echo "  make test-v         — все тесты (подробный вывод)"
	@echo "  make test-file F=tests/test_hexgraph.py — один файл"
	@echo "  make demo           — демо hexcore"
	@echo "  make smoke          — smoke-тест: запуск всех 24 CLI"
	@echo "  make lint           — синтаксическая проверка всех .py"
	@echo "  make clean          — удалить __pycache__ и .pytest_cache"

# ─────────────────────────────────────────────────────────────
test:
	$(PYTEST) -q

test-v:
	$(PYTEST) -v

test-file:
ifndef F
	$(error Укажите файл: make test-file F=tests/test_hexgraph.py)
endif
	$(PYTEST) -v $(F)

# ─────────────────────────────────────────────────────────────
demo:
	$(PYTHON) libs/hexcore/hexcore.py

# ─────────────────────────────────────────────────────────────
smoke:
	$(PYTHON) tools/smoke_test.py

# ─────────────────────────────────────────────────────────────
lint:
	@find . -name "*.py" \
	  -not -path "./.git/*" \
	  -not -path "./*/__pycache__/*" \
	  | sort | xargs $(PYTHON) -m py_compile \
	  && echo "OK: синтаксических ошибок не найдено"

# ─────────────────────────────────────────────────────────────
clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	@echo "OK: кеш очищен"
