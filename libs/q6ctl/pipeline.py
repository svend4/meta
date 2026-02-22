"""pipeline.py — Исполнитель пайплайнов Q6.

Пайплайн — это цепочка шагов вида «модуль:команда [аргументы]».

Формат шага:
  'hexpack:ring'
  'hexcrypt:sbox --param value'
  'hexstat:entropy'

Исполнение:
  1. Каждый шаг запускается как subprocess (python -m <path> <cmd> [args])
  2. Если модуль поддерживает --json: stdout предыдущего → stdin следующего
  3. Результат каждого шага можно сохранить в контекст

Режимы запуска:
  run(steps)     — запустить цепочку, вывод в stdout
  run_to_ctx(steps, ctx_name)  — вывод в контекст
  dry_run(steps) — показать что будет выполнено (без запуска)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .registry import MODULES, SUPERCLUSTERS, get_supercluster, ModuleInfo

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── Разбор шага ──────────────────────────────────────────────────────────────

class PipelineStep:
    """Один шаг пайплайна: модуль + команда + аргументы."""

    def __init__(self, spec: str) -> None:
        """
        spec: 'hexpack:ring' или 'hexpack:ring --start 5' или просто 'hexpack ring'
        """
        # Нормализовать: поддерживаем оба разделителя ':' и ' '
        if ':' in spec:
            mod_cmd, _, rest = spec.partition(' ')
            module_name, _, cmd = mod_cmd.partition(':')
        else:
            parts = spec.split()
            module_name = parts[0]
            cmd = parts[1] if len(parts) > 1 else ''
            rest = ' '.join(parts[2:])

        self.module_name = module_name.strip()
        self.cmd = cmd.strip()
        self.extra_args = rest.strip().split() if rest.strip() else []

    def module_info(self) -> ModuleInfo | None:
        return MODULES.get(self.module_name)

    def python_args(self, json_mode: bool = False) -> list[str]:
        """Полная команда Python для этого шага.

        Порядок: python -m <path> [--json] [global_flags] <cmd> [positional_args]
        Флаги --xxx идут ДО подкоманды (argparse-соглашение).
        Позиционные аргументы идут ПОСЛЕ подкоманды.
        """
        mi = self.module_info()
        if mi is None:
            raise ValueError(f'Неизвестный модуль: {self.module_name}')
        args = [sys.executable, '-m', mi.path]
        # --json и другие --flags идут перед subcommand
        if json_mode and mi.json_ready:
            args.append('--json')
        # extra_args, начинающиеся с '--', — глобальные флаги (до subcommand)
        pre_flags  = [a for a in self.extra_args if a.startswith('--')]
        post_args  = [a for a in self.extra_args if not a.startswith('--')]
        args.extend(pre_flags)
        if self.cmd:
            args.append(self.cmd)
        args.extend(post_args)
        return args

    def __str__(self) -> str:
        base = f'{self.module_name}:{self.cmd}' if self.cmd else self.module_name
        if self.extra_args:
            return base + ' ' + ' '.join(self.extra_args)
        return base

    def __repr__(self) -> str:
        return f'PipelineStep({self!s})'


def parse_steps(specs: list[str]) -> list[PipelineStep]:
    """Распарсить список строк в шаги."""
    return [PipelineStep(s) for s in specs if s.strip()]


# ─── Запуск ───────────────────────────────────────────────────────────────────

class StepResult:
    def __init__(self, step: PipelineStep, returncode: int,
                 stdout: str, stderr: str) -> None:
        self.step = step
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.ok = returncode == 0

    def as_json(self) -> Any:
        """Попробовать распарсить stdout как JSON."""
        try:
            return json.loads(self.stdout)
        except (json.JSONDecodeError, ValueError):
            return None


def run_step(step: PipelineStep,
             stdin_data: str | None = None,
             json_mode: bool = False,
             cwd: Path | None = None) -> StepResult:
    """Запустить один шаг пайплайна."""
    try:
        args = step.python_args(json_mode=json_mode)
    except ValueError as e:
        return StepResult(step, 1, '', str(e))

    proc = subprocess.run(
        args,
        input=stdin_data,
        capture_output=True,
        text=True,
        cwd=str(cwd or _REPO_ROOT),
        env=_env(),
    )
    return StepResult(step, proc.returncode, proc.stdout, proc.stderr)


def run(specs: list[str],
        json_pipe: bool = False,
        verbose: bool = False) -> list[StepResult]:
    """
    Запустить цепочку шагов.

    json_pipe=True: stdout каждого шага → stdin следующего (JSON-пайп).
    Всегда выводит финальный stdout в sys.stdout.
    """
    steps = parse_steps(specs)
    results: list[StepResult] = []
    prev_output: str | None = None

    for i, step in enumerate(steps):
        if verbose:
            print(f'  [{i+1}/{len(steps)}] {step}', file=sys.stderr)

        result = run_step(step, stdin_data=prev_output, json_mode=json_pipe)
        results.append(result)

        if not result.ok:
            print(f'  ✗ Шаг {step} завершился с кодом {result.returncode}',
                  file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            break

        if json_pipe and result.as_json() is not None:
            prev_output = result.stdout
        else:
            prev_output = None

        if verbose and result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)

    # Финальный вывод
    if results and results[-1].ok:
        print(results[-1].stdout, end='')

    return results


def dry_run(specs: list[str], json_pipe: bool = False) -> list[str]:
    """Показать план выполнения без запуска."""
    steps = parse_steps(specs)
    lines = []
    for i, step in enumerate(steps):
        mi = step.module_info()
        if mi is None:
            lines.append(f'  [{i+1}] ✗ {step}  ← НЕИЗВЕСТНЫЙ МОДУЛЬ')
            continue
        json_flag = '(+--json)' if json_pipe and mi.json_ready else ''
        arrow = '→ stdin' if i > 0 and json_pipe else ''
        lines.append(f'  [{i+1}] {step}  [{mi.path}] {json_flag} {arrow}')
        args = step.python_args(json_mode=json_pipe)
        lines.append(f'       cmd: {" ".join(args)}')
    return lines


# ─── Запуск супер-кластеров ───────────────────────────────────────────────────

def run_supercluster(sc_id: str,
                     json_pipe: bool = True,
                     verbose: bool = True) -> list[StepResult]:
    """Запустить предопределённый пайплайн супер-кластера."""
    sc = get_supercluster(sc_id)
    if sc is None:
        raise ValueError(f'Неизвестный супер-кластер: {sc_id}')
    if verbose:
        print(f'  Супер-кластер {sc.id}: {sc.name}', file=sys.stderr)
        print(f'  Кластеры: {", ".join(sc.cluster_ids)}', file=sys.stderr)
        print(f'  Шаги: {len(sc.pipeline)}', file=sys.stderr)
        print(f'  Ожидаемый результат: {sc.emergent}', file=sys.stderr)
        print('', file=sys.stderr)
    return run(sc.pipeline, json_pipe=json_pipe, verbose=verbose)


def dry_run_supercluster(sc_id: str) -> list[str]:
    """Показать план супер-кластера без запуска."""
    sc = get_supercluster(sc_id)
    if sc is None:
        return [f'  ✗ Неизвестный супер-кластер: {sc_id}']
    lines = [
        f'  Супер-кластер: {sc.id} — {sc.name}',
        f'  Кластеры: {", ".join(sc.cluster_ids)}',
        f'  Описание: {sc.description}',
        f'  Результат: {sc.emergent}',
        '',
        '  Шаги пайплайна:',
    ]
    lines.extend(dry_run(sc.pipeline, json_pipe=True))
    return lines


# ─── Утилиты ──────────────────────────────────────────────────────────────────

def _env() -> dict[str, str]:
    """Окружение для subprocess с правильным PYTHONPATH."""
    import os
    env = os.environ.copy()
    pp = env.get('PYTHONPATH', '')
    root = str(_REPO_ROOT)
    if root not in pp:
        env['PYTHONPATH'] = f'{root}:{pp}' if pp else root
    return env


def available_pipelines() -> list[tuple[str, str, list[str]]]:
    """Список всех предопределённых пайплайнов (id, name, steps)."""
    result = []
    for sid, sc in SUPERCLUSTERS.items():
        result.append((sc.id, sc.name, sc.pipeline))
    return result
