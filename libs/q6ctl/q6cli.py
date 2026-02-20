"""q6cli.py — Главный CLI оркестратор Q6.

Использование:
  python -m libs.q6ctl.q6cli <команда> [опции]

Команды:
  list modules              — список всех 41 модулей
  list clusters             — список всех 8 кластеров
  list sc                   — список супер-кластеров
  list pipelines            — предопределённые пайплайны

  run <SC-ID> [--dry]       — запустить супер-кластер
  pipe <step1> <step2> ...  — произвольная цепочка шагов
  call <module> <cmd> [args]— запустить один модуль

  ctx new <name>            — создать контекст
  ctx show <name>           — показать состояние контекста
  ctx keys <name>           — перечислить ключи
  ctx get <name> <key>      — вывести данные ключа
  ctx list                  — список всех контекстов
  ctx del <name>            — удалить контекст

  info <module|cluster|SC>  — подробности об объекте

Флаги:
  --json   — JSON-режим пайпа (stdout→stdin)
  --dry    — показать план без выполнения
  --verbose / -v
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Добавить корень репозитория в PYTHONPATH
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from libs.q6ctl import registry as reg
from libs.q6ctl import context as ctx
from libs.q6ctl import pipeline as pip

# Карта SC-ID → путь к shell-скрипту
_SCRIPT_MAP: dict[str, str] = {
    'SC-1':  'scripts/sc1_herman_cipher.sh',
    'SC-2':  'scripts/sc2_platinum_sbox.sh',
    'SC-3':  'scripts/sc3_ca_atlas.sh',
    'SC-4':  'scripts/sc4_genomic_iching.sh',
    'SC-5':  'scripts/sc5_automl_crypto.sh',
    'SC-6':  'scripts/sc6_genomic_ca.sh',
    'SC-7':  'scripts/sc7_phi_q6.sh',
    'TSC-1': 'scripts/tsc1_cipher_symmetry.sh',
    'TSC-2': 'scripts/tsc2_automl_crypto.sh',
    'TSC-3': 'scripts/tsc3_genomic_oracle.sh',
    'MC':    'scripts/mc_genomic_oracle_q6.sh',
}


# ─── ANSI-цвета ───────────────────────────────────────────────────────────────

_R = '\033[0m'
_B = '\033[1m'
_G = '\033[1;32m'
_Y = '\033[1;33m'
_C = '\033[1;36m'
_M = '\033[1;35m'


def _h(s: str) -> str: return f'{_B}{s}{_R}'
def _g(s: str) -> str: return f'{_G}{s}{_R}'
def _y(s: str) -> str: return f'{_Y}{s}{_R}'
def _c(s: str) -> str: return f'{_C}{s}{_R}'


# ─── list ──────────────────────────────────────────────────────────────────────

def cmd_list(sub: str) -> int:
    if sub == 'modules':
        _list_modules()
    elif sub == 'clusters':
        _list_clusters()
    elif sub in ('sc', 'superclusters'):
        _list_superclusters()
    elif sub == 'pipelines':
        _list_pipelines()
    else:
        print(f'  Неизвестно: {sub!r}. Используй: modules | clusters | sc | pipelines')
        return 1
    return 0


def _list_modules() -> None:
    cluster_order = ['K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8']
    print(f'\n{_h("Модули Q6")}  ({len(reg.MODULES)} модулей)\n')
    for cid in cluster_order:
        ci = reg.CLUSTERS.get(cid)
        if not ci:
            continue
        print(f'  {_c(cid)} {_y(ci.name)}')
        for name in ci.modules:
            m = reg.MODULES.get(name)
            if not m:
                continue
            json_tag = _g('✓json') if m.json_ready else '     '
            cmds = ', '.join(m.commands[:4])
            print(f'    {json_tag}  {_b(name):<20} {m.description[:45]}')
            print(f'           {" "*20} команды: {cmds}')
    print()


def _b(s: str) -> str: return f'{_B}{s}{_R}'


def _list_clusters() -> None:
    print(f'\n{_h("Кластеры Q6")}  ({len(reg.CLUSTERS)} кластеров)\n')
    for cid, ci in sorted(reg.CLUSTERS.items()):
        mods = ', '.join(ci.modules[:5])
        more = f' +{len(ci.modules)-5}' if len(ci.modules) > 5 else ''
        print(f'  {_c(cid)}  {_y(ci.name):<25} {len(ci.modules)} модулей')
        print(f'       {ci.description}')
        print(f'       Модули: {mods}{more}')
        print(f'       {_g("→")} {ci.emergent}')
        print()


def _list_superclusters() -> None:
    print(f'\n{_h("Супер-кластеры Q6")}  ({len(reg.SUPERCLUSTERS)} комбинаций)\n')
    pairwise = [k for k in reg.SUPERCLUSTERS if k.startswith('SC-')]
    triples  = [k for k in reg.SUPERCLUSTERS if k.startswith('TSC-')]
    mega     = [k for k in reg.SUPERCLUSTERS if k == 'MC']
    for group, label in [(pairwise, 'Парные (SC)'),
                         (triples, 'Тройные (TSC)'),
                         (mega, 'Мега (MC)')]:
        if not group:
            continue
        print(f'  {_y(label)}')
        for sid in group:
            sc = reg.SUPERCLUSTERS[sid]
            clusters = ' × '.join(sc.cluster_ids)
            print(f'    {_c(sc.id):<8} {_b(sc.name):<28} [{clusters}]')
            print(f'             {sc.emergent}')
        print()


def _list_pipelines() -> None:
    print(f'\n{_h("Предопределённые пайплайны Q6")}\n')
    for sid, sc in reg.SUPERCLUSTERS.items():
        print(f'  {_c(sc.id)}: {sc.name}')
        for i, step in enumerate(sc.pipeline):
            arrow = '  →' if i > 0 else '   '
            print(f'  {arrow} {step}')
        print()


# ─── info ──────────────────────────────────────────────────────────────────────

def cmd_info(name: str) -> int:
    # Попробовать как модуль
    m = reg.get_module(name)
    if m:
        ci = reg.cluster_of_module(name)
        print(f'\n{_h("Модуль:")} {_c(m.name)}')
        print(f'  Путь:       {m.path}')
        print(f'  Кластер:    {m.cluster} ({ci.name if ci else "?"})')
        print(f'  JSON:       {"✓ поддерживается" if m.json_ready else "✗ не поддерживается"}')
        print(f'  Описание:   {m.description}')
        print(f'  Команды:    {", ".join(m.commands)}')
        print(f'  Запуск:     python -m {m.path} <команда>')
        return 0

    # Попробовать как кластер
    c = reg.get_cluster(name)
    if c:
        print(f'\n{_h("Кластер:")} {_c(c.id)} — {_y(c.name)}')
        print(f'  Описание:   {c.description}')
        print(f'  Результат:  {_g(c.emergent)}')
        print(f'  Модули ({len(c.modules)}):')
        for mn in c.modules:
            mm = reg.MODULES.get(mn)
            desc = mm.description if mm else ''
            print(f'    {mn:<20} {desc}')
        return 0

    # Попробовать как супер-кластер
    sc = reg.get_supercluster(name)
    if sc:
        print(f'\n{_h("Супер-кластер:")} {_c(sc.id)} — {_y(sc.name)}')
        print(f'  Кластеры:   {" × ".join(sc.cluster_ids)}')
        print(f'  Описание:   {sc.description}')
        print(f'  Результат:  {_g(sc.emergent)}')
        print(f'  Пайплайн ({len(sc.pipeline)} шагов):')
        for i, step in enumerate(sc.pipeline):
            arrow = '  →' if i > 0 else '   '
            print(f'  {arrow} {step}')
        return 0

    print(f'  ✗ "{name}" не найден. Попробуй: q6ctl list modules')
    return 1


# ─── run ───────────────────────────────────────────────────────────────────────

def cmd_run(sc_id: str, dry: bool = False, verbose: bool = False) -> int:
    # Специальный случай: run all
    if sc_id == 'all':
        return _run_all(dry=dry, verbose=verbose)

    sc = reg.get_supercluster(sc_id)
    if sc is None:
        print(f'  ✗ Неизвестный супер-кластер: {sc_id!r}')
        print(f'  Доступные: {", ".join(reg.all_supercluster_ids())} | all')
        return 1

    script_rel = _SCRIPT_MAP.get(sc_id)
    script_path = (_REPO / script_rel) if script_rel else None

    if dry:
        lines = pip.dry_run_supercluster(sc_id)
        for line in lines:
            print(line)
        if script_path:
            print(f'\n  {_g("✓")} Shell-скрипт: bash {script_path}')
        return 0

    # Запустить через shell-скрипт (если есть)
    if script_path and script_path.exists():
        print(f'\n{_h("Запуск:")} {_c(sc.id)} — {sc.name}')
        print(f'  {_g("→")} bash {script_path}\n')
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = str(_REPO)
        result = subprocess.run(['bash', str(script_path)], env=env, cwd=str(_REPO))
        return result.returncode

    # Запасной вариант: прямой pipeline.py
    print(f'\n{_h("Запуск супер-кластера:")} {_c(sc.id)} — {sc.name}')
    print(f'  {sc.emergent}\n')
    results = pip.run_supercluster(sc_id, json_pipe=True, verbose=verbose)
    ok_count = sum(1 for r in results if r.ok)
    print(f'\n  {_g("✓") if ok_count == len(results) else "⚠"} '
          f'{ok_count}/{len(results)} шагов успешно', file=sys.stderr)
    return 0 if ok_count == len(results) else 1


def _run_all(dry: bool = False, verbose: bool = False) -> int:
    """Запустить все супер-кластеры по порядку."""
    order = ['SC-1', 'SC-2', 'SC-3', 'SC-4', 'SC-5', 'SC-6', 'SC-7',
             'TSC-1', 'TSC-2', 'TSC-3', 'MC']

    if dry:
        print(f'\n{_h("Запуск всех супер-кластеров Q6")}  ({len(order)} SC)\n')
        for sid in order:
            sc = reg.get_supercluster(sid)
            script_rel = _SCRIPT_MAP.get(sid, '?')
            label = f'{sc.name}' if sc else sid
            clusters = ' × '.join(sc.cluster_ids) if sc else '?'
            print(f'  {_c(sid):<8} {label:<28} [{clusters}]')
            print(f'           bash {script_rel}')
        print()
        return 0

    print(f'\n{_h("Q6 Полный прогон")}  ({len(order)} супер-кластеров)\n')
    failed: list[str] = []
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(_REPO)

    for sid in order:
        sc = reg.get_supercluster(sid)
        script_rel = _SCRIPT_MAP.get(sid)
        if not script_rel:
            print(f'  {_y("⚠")} {sid}: нет скрипта, пропуск')
            continue

        script_path = _REPO / script_rel
        if not script_path.exists():
            print(f'  ✗ {sid}: скрипт не найден: {script_path}')
            failed.append(sid)
            continue

        name = sc.name if sc else sid
        print(f'  {_g("▶")} {_c(sid)}: {name}')
        result = subprocess.run(['bash', str(script_path)], env=env, cwd=str(_REPO))
        if result.returncode != 0:
            print(f'  ✗ {sid} завершился с кодом {result.returncode}')
            failed.append(sid)
        else:
            print(f'  {_g("✓")} {sid} готов\n')

    total = len(order)
    ok = total - len(failed)
    print(f'\n{_h("Итог:")} {ok}/{total} успешно', end='')
    if failed:
        print(f'  ✗ ошибки: {", ".join(failed)}')
    else:
        print(f'  {_g("✓ все SC выполнены")}')
    return 0 if not failed else 1


# ─── pipe ──────────────────────────────────────────────────────────────────────

def cmd_pipe(steps: list[str], json_mode: bool = False,
             dry: bool = False, verbose: bool = False) -> int:
    if not steps:
        print('  ✗ Нужен хотя бы один шаг.')
        return 1

    if dry:
        lines = pip.dry_run(steps, json_pipe=json_mode)
        for line in lines:
            print(line)
        return 0

    results = pip.run(steps, json_pipe=json_mode, verbose=verbose)
    ok_count = sum(1 for r in results if r.ok)
    if ok_count < len(results):
        print(f'\n  ⚠ {ok_count}/{len(results)} шагов успешно', file=sys.stderr)
        return 1
    return 0


# ─── call ──────────────────────────────────────────────────────────────────────

def cmd_call(module: str, cmd: str, args: list[str],
             json_mode: bool = False) -> int:
    spec = f'{module}:{cmd} ' + ' '.join(args)
    return cmd_pipe([spec.strip()], json_mode=json_mode)


# ─── ctx ───────────────────────────────────────────────────────────────────────

def cmd_ctx(action: str, name: str = '', key: str = '') -> int:
    if action == 'list':
        names = ctx.list_contexts()
        if not names:
            print('  (нет контекстов)')
        else:
            print(f'\n  {_h("Контексты Q6")} ({len(names)}):\n')
            for n in names:
                ks = ctx.keys(n)
                print(f'  {_c(n):<20} {len(ks)} ключей: {", ".join(ks[:5])}')
        return 0

    if not name:
        print('  ✗ Нужно имя контекста')
        return 1

    if action == 'new':
        ctx.create(name)
        print(f'  {_g("✓")} Контекст "{name}" создан: {ctx._ctx_path(name)}')
        return 0

    if action == 'show':
        lines = ctx.show(name)
        for line in lines:
            print(line)
        return 0

    if action == 'keys':
        try:
            ks = ctx.keys(name)
            print(f'  Ключи контекста "{name}": {", ".join(ks) if ks else "(пусто)"}')
        except FileNotFoundError:
            print(f'  ✗ Контекст "{name}" не найден')
            return 1
        return 0

    if action == 'get':
        if not key:
            print('  ✗ Нужно имя ключа: ctx get <name> <key>')
            return 1
        try:
            data = ctx.read_key(name, key)
            import json
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except (FileNotFoundError, KeyError) as e:
            print(f'  ✗ {e}')
            return 1
        return 0

    if action == 'del':
        if ctx.delete(name):
            print(f'  {_g("✓")} Контекст "{name}" удалён')
        else:
            print(f'  (контекст "{name}" не существовал)')
        return 0

    print(f'  ✗ Неизвестное действие: {action}')
    return 1


# ─── Главный парсер ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='q6ctl',
        description=(
            'Q6 Orchestrator — управление модулями, кластерами и пайплайнами Q6.\n'
            '  q6ctl list modules        — все 41 модуль\n'
            '  q6ctl list clusters       — все 8 кластеров\n'
            '  q6ctl list sc             — все супер-кластеры\n'
            '  q6ctl info hexpack        — информация о модуле\n'
            '  q6ctl run SC-1            — запустить супер-кластер\n'
            '  q6ctl pipe hexpack:ring hexcrypt:sbox   — цепочка\n'
            '  q6ctl ctx new my_run      — создать контекст'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('-v', '--verbose', action='store_true', help='Подробный вывод')

    sub = p.add_subparsers(dest='cmd', metavar='команда')

    # list
    lp = sub.add_parser('list', help='Список модулей/кластеров/супер-кластеров')
    lp.add_argument('what', choices=['modules', 'clusters', 'sc', 'superclusters', 'pipelines'],
                    help='Что показать')

    # info
    ip = sub.add_parser('info', help='Информация об объекте')
    ip.add_argument('name', help='Имя модуля / кластера / SC')

    # run
    rp = sub.add_parser('run', help='Запустить супер-кластер (или all)')
    rp.add_argument('sc_id', metavar='SC-ID',
                    help='ID супер-кластера (SC-1, TSC-2, MC, ...) или "all"')
    rp.add_argument('--dry', action='store_true', help='Только показать план')

    # pipe
    pp = sub.add_parser('pipe', help='Произвольный пайплайн шагов')
    pp.add_argument('steps', nargs='+', metavar='шаг',
                    help='Шаги: hexpack:ring hexcrypt:sbox ...')
    pp.add_argument('--json', dest='json_mode', action='store_true',
                    help='JSON-пайп (stdout→stdin)')
    pp.add_argument('--dry', action='store_true', help='Только показать план')

    # call
    cp = sub.add_parser('call', help='Запустить один модуль')
    cp.add_argument('module', help='Имя модуля')
    cp.add_argument('cmd', help='Команда')
    cp.add_argument('args', nargs='*', help='Дополнительные аргументы')
    cp.add_argument('--json', dest='json_mode', action='store_true')

    # ctx
    xp = sub.add_parser('ctx', help='Управление контекстами')
    xp.add_argument('action', choices=['new', 'show', 'keys', 'get', 'del', 'list'],
                    help='Действие')
    xp.add_argument('name', nargs='?', default='', help='Имя контекста')
    xp.add_argument('key', nargs='?', default='', help='Ключ (для get)')

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0

    if args.cmd == 'list':
        return cmd_list(args.what)

    if args.cmd == 'info':
        return cmd_info(args.name)

    if args.cmd == 'run':
        return cmd_run(args.sc_id, dry=args.dry, verbose=args.verbose)

    if args.cmd == 'pipe':
        return cmd_pipe(args.steps, json_mode=args.json_mode,
                        dry=args.dry, verbose=args.verbose)

    if args.cmd == 'call':
        return cmd_call(args.module, args.cmd_arg if hasattr(args, 'cmd_arg') else args.cmd,
                        args.args, json_mode=args.json_mode)

    if args.cmd == 'ctx':
        return cmd_ctx(args.action, args.name, args.key)

    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
