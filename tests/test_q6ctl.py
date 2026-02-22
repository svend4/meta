"""Тесты libs/q6ctl/ — context, pipeline, q6cli."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# ── context ──────────────────────────────────────────────────────────────────

import libs.q6ctl.context as ctx_mod


class TestContext(unittest.TestCase):
    """Тесты context.py — создание, сохранение, ключи."""

    def setUp(self):
        # Использовать временный каталог для всех контекстов
        self._tmpdir = tempfile.mkdtemp()
        self._orig_dir = ctx_mod._CTX_DIR
        ctx_mod._CTX_DIR = Path(self._tmpdir)

    def tearDown(self):
        ctx_mod._CTX_DIR = self._orig_dir
        # Почистить временные файлы
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _name(self, suffix=''):
        return f'test_{id(self)}_{suffix}'

    def test_create_returns_dict(self):
        c = ctx_mod.create(self._name())
        self.assertIsInstance(c, dict)

    def test_create_has_meta(self):
        c = ctx_mod.create(self._name())
        self.assertIn('_meta', c)
        self.assertEqual(c['_meta']['name'], self._name())

    def test_create_writes_file(self):
        name = self._name()
        ctx_mod.create(name)
        p = ctx_mod._ctx_path(name)
        self.assertTrue(p.exists())

    def test_load_after_create(self):
        name = self._name()
        created = ctx_mod.create(name)
        loaded = ctx_mod.load(name)
        self.assertEqual(created['_meta']['name'], loaded['_meta']['name'])

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            ctx_mod.load('nonexistent_xyz_12345')

    def test_load_or_create_missing(self):
        name = self._name('loc')
        c = ctx_mod.load_or_create(name)
        self.assertIn('_meta', c)

    def test_load_or_create_existing(self):
        name = self._name('loc2')
        ctx_mod.create(name)
        ctx_mod.write_key(name, 'foo', 42)
        c = ctx_mod.load_or_create(name)
        self.assertIn('foo', c)

    def test_delete_existing(self):
        name = self._name('del')
        ctx_mod.create(name)
        result = ctx_mod.delete(name)
        self.assertTrue(result)
        self.assertFalse(ctx_mod._ctx_path(name).exists())

    def test_delete_missing_returns_false(self):
        self.assertFalse(ctx_mod.delete('totally_missing_ctx'))

    def test_write_and_read_key(self):
        name = self._name('rw')
        ctx_mod.create(name)
        ctx_mod.write_key(name, 'ring', [1, 2, 3], source='hexpack:ring')
        data = ctx_mod.read_key(name, 'ring')
        self.assertEqual(data, [1, 2, 3])

    def test_read_key_missing_raises(self):
        name = self._name('rk')
        ctx_mod.create(name)
        with self.assertRaises(KeyError):
            ctx_mod.read_key(name, 'nonexistent')

    def test_write_key_records_source(self):
        name = self._name('src')
        ctx_mod.create(name)
        ctx_mod.write_key(name, 'x', 99, source='hexpack:test')
        raw = ctx_mod.load(name)
        self.assertEqual(raw['x']['source'], 'hexpack:test')

    def test_has_key_true(self):
        name = self._name('hk')
        ctx_mod.create(name)
        ctx_mod.write_key(name, 'k', 1)
        self.assertTrue(ctx_mod.has_key(name, 'k'))

    def test_has_key_false(self):
        name = self._name('hk2')
        ctx_mod.create(name)
        self.assertFalse(ctx_mod.has_key(name, 'missing'))

    def test_has_key_missing_context(self):
        self.assertFalse(ctx_mod.has_key('missing_ctx_xyz', 'k'))

    def test_keys_empty(self):
        name = self._name('ke')
        ctx_mod.create(name)
        self.assertEqual(ctx_mod.keys(name), [])

    def test_keys_with_data(self):
        name = self._name('kd')
        ctx_mod.create(name)
        ctx_mod.write_key(name, 'a', 1)
        ctx_mod.write_key(name, 'b', 2)
        ks = ctx_mod.keys(name)
        self.assertIn('a', ks)
        self.assertIn('b', ks)
        self.assertNotIn('_meta', ks)

    def test_list_contexts(self):
        name = self._name('lc')
        ctx_mod.create(name)
        names = ctx_mod.list_contexts()
        self.assertIn(name, names)

    def test_show_returns_lines(self):
        name = self._name('sh')
        ctx_mod.create(name)
        ctx_mod.write_key(name, 'ring', [1, 2], source='hexpack:ring')
        lines = ctx_mod.show(name)
        self.assertIsInstance(lines, list)
        self.assertTrue(any(name in line for line in lines))

    def test_show_missing_context(self):
        lines = ctx_mod.show('no_such_context_xyz')
        self.assertTrue(any('не существует' in l for l in lines))

    def test_meta_has_steps_after_write(self):
        name = self._name('st')
        ctx_mod.create(name)
        ctx_mod.write_key(name, 'k', 1, source='test')
        raw = ctx_mod.load(name)
        self.assertGreater(len(raw['_meta']['steps']), 0)


# ── pipeline ──────────────────────────────────────────────────────────────────

from libs.q6ctl.pipeline import (
    PipelineStep, parse_steps,
    dry_run, dry_run_supercluster,
    StepResult, available_pipelines,
)


class TestPipelineStep(unittest.TestCase):
    def test_colon_syntax(self):
        s = PipelineStep('hexpack:ring')
        self.assertEqual(s.module_name, 'hexpack')
        self.assertEqual(s.cmd, 'ring')
        self.assertEqual(s.extra_args, [])

    def test_colon_syntax_with_args(self):
        s = PipelineStep('hexpack:fixpoint --start 5')
        self.assertEqual(s.module_name, 'hexpack')
        self.assertEqual(s.cmd, 'fixpoint')
        self.assertIn('--start', s.extra_args)
        self.assertIn('5', s.extra_args)

    def test_space_syntax(self):
        s = PipelineStep('hexpack ring')
        self.assertEqual(s.module_name, 'hexpack')
        self.assertEqual(s.cmd, 'ring')

    def test_str(self):
        s = PipelineStep('hexpack:ring')
        self.assertEqual(str(s), 'hexpack:ring')

    def test_repr(self):
        s = PipelineStep('hexpack:ring')
        self.assertIn('hexpack:ring', repr(s))

    def test_module_info_known(self):
        s = PipelineStep('hexpack:ring')
        mi = s.module_info()
        self.assertIsNotNone(mi)
        self.assertEqual(mi.name, 'hexpack')

    def test_module_info_unknown(self):
        s = PipelineStep('nonexistent:cmd')
        self.assertIsNone(s.module_info())

    def test_python_args_known_json(self):
        s = PipelineStep('hexpack:ring')
        args = s.python_args(json_mode=True)
        self.assertIn('--json', args)
        self.assertIn('ring', args)

    def test_python_args_unknown_raises(self):
        s = PipelineStep('nonexistent:cmd')
        with self.assertRaises(ValueError):
            s.python_args()

    def test_pre_flags_before_cmd(self):
        """--flags идут до subcommand в python_args."""
        s = PipelineStep('hexpack:fixpoint --start 5')
        args = s.python_args()
        cmd_idx = args.index('fixpoint')
        start_idx = args.index('--start')
        self.assertLess(start_idx, cmd_idx)


class TestParseSteps(unittest.TestCase):
    def test_parses_list(self):
        steps = parse_steps(['hexpack:ring', 'hexphi:fibonacci'])
        self.assertEqual(len(steps), 2)
        self.assertIsInstance(steps[0], PipelineStep)

    def test_skips_blank(self):
        steps = parse_steps(['hexpack:ring', '', '  ', 'hexphi:fibonacci'])
        self.assertEqual(len(steps), 2)


class TestDryRun(unittest.TestCase):
    def test_returns_lines(self):
        lines = dry_run(['hexpack:ring'])
        self.assertIsInstance(lines, list)
        self.assertGreater(len(lines), 0)

    def test_contains_module_path(self):
        lines = dry_run(['hexpack:ring'])
        combined = '\n'.join(lines)
        self.assertIn('hexpack', combined)

    def test_unknown_module_marked(self):
        lines = dry_run(['nonexistent:cmd'])
        combined = '\n'.join(lines)
        self.assertIn('НЕИЗВЕСТНЫЙ', combined)

    def test_dry_run_supercluster(self):
        lines = dry_run_supercluster('SC-7')
        combined = '\n'.join(lines)
        self.assertIn('SC-7', combined)
        self.assertIn('hexpack', combined)

    def test_dry_run_supercluster_missing(self):
        lines = dry_run_supercluster('SC-99')
        self.assertTrue(any('Неизвестный' in l for l in lines))


class TestStepResult(unittest.TestCase):
    def _make_step(self):
        return PipelineStep('hexpack:ring')

    def test_ok_when_zero(self):
        r = StepResult(self._make_step(), 0, 'out', '')
        self.assertTrue(r.ok)

    def test_not_ok_when_nonzero(self):
        r = StepResult(self._make_step(), 1, '', 'err')
        self.assertFalse(r.ok)

    def test_as_json_valid(self):
        r = StepResult(self._make_step(), 0, '{"key": 1}', '')
        self.assertEqual(r.as_json(), {'key': 1})

    def test_as_json_invalid(self):
        r = StepResult(self._make_step(), 0, 'not json', '')
        self.assertIsNone(r.as_json())


class TestAvailablePipelines(unittest.TestCase):
    def test_returns_list(self):
        pl = available_pipelines()
        self.assertIsInstance(pl, list)

    def test_all_have_three_fields(self):
        for item in available_pipelines():
            self.assertEqual(len(item), 3)

    def test_sc7_present(self):
        ids = [item[0] for item in available_pipelines()]
        self.assertIn('SC-7', ids)


# ── q6cli ──────────────────────────────────────────────────────────────────────

from libs.q6ctl.q6cli import build_parser, cmd_info, cmd_list, main as q6main


class TestBuildParser(unittest.TestCase):
    def test_no_args_returns_0(self):
        rc = q6main([])
        self.assertEqual(rc, 0)

    def test_list_modules(self):
        # Should not crash; returns 0
        rc = q6main(['list', 'modules'])
        self.assertEqual(rc, 0)

    def test_list_clusters(self):
        rc = q6main(['list', 'clusters'])
        self.assertEqual(rc, 0)

    def test_list_sc(self):
        rc = q6main(['list', 'sc'])
        self.assertEqual(rc, 0)

    def test_list_pipelines(self):
        rc = q6main(['list', 'pipelines'])
        self.assertEqual(rc, 0)

    def test_info_module(self):
        rc = q6main(['info', 'hexpack'])
        self.assertEqual(rc, 0)

    def test_info_cluster(self):
        rc = q6main(['info', 'K5'])
        self.assertEqual(rc, 0)

    def test_info_supercluster(self):
        rc = q6main(['info', 'SC-7'])
        self.assertEqual(rc, 0)

    def test_info_unknown_returns_1(self):
        rc = q6main(['info', 'nonexistent_xyz'])
        self.assertEqual(rc, 1)

    def test_run_dry(self):
        rc = q6main(['run', 'SC-7', '--dry'])
        self.assertEqual(rc, 0)

    def test_run_all_dry(self):
        rc = q6main(['run', 'all', '--dry'])
        self.assertEqual(rc, 0)

    def test_run_unknown_sc_returns_1(self):
        rc = q6main(['run', 'SC-99'])
        self.assertEqual(rc, 1)

    def test_pipe_dry(self):
        rc = q6main(['pipe', '--dry', 'hexpack:ring'])
        self.assertEqual(rc, 0)

    def test_cmd_dispatch_no_collision(self):
        """После фикса: call не перезаписывает args.cmd."""
        p = build_parser()
        args = p.parse_args(['call', 'hexpack', 'ring'])
        self.assertEqual(args.cmd, 'call')
        self.assertEqual(args.module, 'hexpack')
        self.assertEqual(args.module_cmd, 'ring')

    def test_call_executes(self):
        """cmd_call реально вызывается и возвращает 0."""
        rc = q6main(['call', 'hexpack', 'ring', '--json'])
        self.assertEqual(rc, 0)

    def test_ctx_list(self):
        rc = q6main(['ctx', 'list'])
        self.assertEqual(rc, 0)

    def test_ctx_new_and_del(self):
        import tempfile, os, shutil
        tmpdir = tempfile.mkdtemp()
        orig = ctx_mod._CTX_DIR
        ctx_mod._CTX_DIR = Path(tmpdir)
        try:
            rc_new = q6main(['ctx', 'new', 'test_q6cli_ctx'])
            self.assertEqual(rc_new, 0)
            rc_del = q6main(['ctx', 'del', 'test_q6cli_ctx'])
            self.assertEqual(rc_del, 0)
        finally:
            ctx_mod._CTX_DIR = orig
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
