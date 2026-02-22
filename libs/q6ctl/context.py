"""context.py — Общий JSON-контекст для передачи данных между модулями Q6.

Концепция:
  Каждый шаг пайплайна может ЧИТАТЬ из контекста и ЗАПИСЫВАТЬ в него.
  Контекст хранится в JSON-файле в /tmp/ (или указанном каталоге).

  Пример:
    q6ctl ctx new my_run
    hexpack ring --json >> ctx                    # запись ring → контекст
    hexcrypt sbox --from-ctx ring --json >> ctx   # чтение ring, запись sbox
    hexstat entropy --from-ctx sbox               # чтение sbox, вывод

  Формат файла контекста:
  {
    "_meta": {"created": "...", "updated": "...", "name": "..."},
    "ring":  { "data": [...], "source": "hexpack:ring", "ts": "..." },
    "sbox":  { "data": {...}, "source": "hexcrypt:sbox", "ts": "..." },
    ...
  }
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_CTX_DIR = Path(os.environ.get('Q6_CTX_DIR', tempfile.gettempdir()))
_CTX_PREFIX = 'q6_ctx_'


def _ctx_path(name: str) -> Path:
    """Путь к файлу контекста по имени."""
    return _CTX_DIR / f'{_CTX_PREFIX}{name}.json'


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ─── Создание / загрузка ──────────────────────────────────────────────────────

def create(name: str) -> dict[str, Any]:
    """Создать новый пустой контекст. Перезаписывает существующий."""
    ctx: dict[str, Any] = {
        '_meta': {
            'name': name,
            'created': _now(),
            'updated': _now(),
            'steps': [],
        }
    }
    save(name, ctx)
    return ctx


def load(name: str) -> dict[str, Any]:
    """Загрузить контекст. Ошибка если не существует."""
    p = _ctx_path(name)
    if not p.exists():
        raise FileNotFoundError(f'Контекст "{name}" не найден: {p}')
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_or_create(name: str) -> dict[str, Any]:
    """Загрузить существующий контекст или создать новый."""
    try:
        return load(name)
    except FileNotFoundError:
        return create(name)


def save(name: str, ctx: dict[str, Any]) -> None:
    """Сохранить контекст на диск."""
    ctx.setdefault('_meta', {})['updated'] = _now()
    p = _ctx_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2)


def delete(name: str) -> bool:
    """Удалить контекст. Возвращает True если файл существовал."""
    p = _ctx_path(name)
    if p.exists():
        p.unlink()
        return True
    return False


def list_contexts() -> list[str]:
    """Список всех существующих контекстов."""
    return sorted(
        p.stem[len(_CTX_PREFIX):]
        for p in _CTX_DIR.glob(f'{_CTX_PREFIX}*.json')
    )


# ─── Чтение / запись ключей ───────────────────────────────────────────────────

def write_key(name: str, key: str, data: Any, source: str = '') -> None:
    """Записать данные под ключом key в контекст name."""
    ctx = load_or_create(name)
    ctx[key] = {
        'data': data,
        'source': source,
        'ts': _now(),
    }
    ctx['_meta'].setdefault('steps', []).append({
        'key': key, 'source': source, 'ts': _now()
    })
    save(name, ctx)


def read_key(name: str, key: str) -> Any:
    """Прочитать данные ключа key из контекста name."""
    ctx = load(name)
    if key not in ctx:
        raise KeyError(f'Ключ "{key}" не найден в контексте "{name}"')
    entry = ctx[key]
    return entry['data'] if isinstance(entry, dict) and 'data' in entry else entry


def has_key(name: str, key: str) -> bool:
    """Проверить наличие ключа в контексте."""
    try:
        ctx = load(name)
        return key in ctx and key != '_meta'
    except FileNotFoundError:
        return False


def keys(name: str) -> list[str]:
    """Список всех ключей данных в контексте (без _meta)."""
    ctx = load(name)
    return [k for k in ctx if k != '_meta']


# ─── JSON-pipe: stdin → контекст → stdout ────────────────────────────────────

def pipe_in(name: str, key: str, source: str = 'stdin') -> Any:
    """Читать JSON из stdin и сохранить в контекст под ключом key."""
    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError('stdin пуст')
    data = json.loads(raw)
    write_key(name, key, data, source=source)
    return data


def pipe_out(name: str, key: str) -> None:
    """Вывести данные ключа key в stdout как JSON."""
    data = read_key(name, key)
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ─── Отображение ──────────────────────────────────────────────────────────────

def show(name: str) -> list[str]:
    """Красивый вывод состояния контекста."""
    try:
        ctx = load(name)
    except FileNotFoundError:
        return [f'  Контекст "{name}" не существует']

    meta = ctx.get('_meta', {})
    lines = [
        f'  Контекст: {name}',
        f'  Создан:   {meta.get("created", "?")}',
        f'  Обновлён: {meta.get("updated", "?")}',
        f'  Путь:     {_ctx_path(name)}',
        '',
        f'  {"Ключ":<20} {"Источник":<30} {"Время"}',
        f'  {"─"*65}',
    ]
    data_keys = [k for k in ctx if k != '_meta']
    if not data_keys:
        lines.append('  (пусто)')
    else:
        for k in data_keys:
            entry = ctx[k]
            if isinstance(entry, dict) and 'source' in entry:
                src = entry.get('source', '')
                ts = entry.get('ts', '')
                d = entry.get('data', entry)
                size = (f'{len(d)} элементов' if isinstance(d, (list, dict))
                        else str(d)[:30])
                lines.append(f'  {k:<20} {src:<30} {ts}')
                lines.append(f'  {"":>20} → {size}')
            else:
                lines.append(f'  {k:<20} (raw)')
    steps = meta.get('steps', [])
    if steps:
        lines += ['', f'  История ({len(steps)} шагов):']
        for s in steps[-5:]:
            lines.append(f'    {s["ts"]}  {s["source"]} → [{s["key"]}]')
    return lines
