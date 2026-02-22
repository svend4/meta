"""Тесты libs/q6ctl/registry.py — структурная целостность реестра Q6."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest

from libs.q6ctl.registry import (
    MODULES, CLUSTERS, SUPERCLUSTERS,
    get_module, get_cluster, get_supercluster,
    modules_in_cluster, cluster_of_module,
    all_module_names, all_cluster_ids, all_supercluster_ids,
    json_ready_modules,
    ModuleInfo, ClusterInfo, SuperClusterInfo,
)

_EXPECTED_MODULES = 42
_EXPECTED_CLUSTERS = 8
_KNOWN_CLUSTERS = {'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8'}
_KNOWN_JSON_MODULES = {
    'hexcrypt', 'hexring', 'hexcode', 'karnaugh6',
    'hexca', 'hexstat', 'hexsym',
    'hexlearn', 'hexopt',
    'hexbio', 'hexspec', 'hexnav', 'hextrimat',
    'hexpack', 'hexphi',
}


class TestModulesIntegrity(unittest.TestCase):
    """Целостность словаря MODULES."""

    def test_module_count(self):
        self.assertEqual(len(MODULES), _EXPECTED_MODULES)

    def test_no_duplicate_keys(self):
        """Все ключи уникальны (дубликат молча перезаписывается Python)."""
        # Проверяем через name-поле каждого ModuleInfo
        names = [m.name for m in MODULES.values()]
        self.assertEqual(len(names), len(set(names)), f"Duplicate names: {[n for n in names if names.count(n)>1]}")

    def test_all_modules_are_moduleinfo(self):
        for name, m in MODULES.items():
            self.assertIsInstance(m, ModuleInfo, f"{name} is not ModuleInfo")

    def test_name_matches_key(self):
        """m.name совпадает с ключом словаря."""
        for key, m in MODULES.items():
            self.assertEqual(key, m.name, f"Key {key!r} != m.name {m.name!r}")

    def test_all_have_path(self):
        for name, m in MODULES.items():
            self.assertTrue(m.path, f"{name}: empty path")

    def test_all_have_commands(self):
        for name, m in MODULES.items():
            self.assertIsInstance(m.commands, list, f"{name}: commands not a list")
            self.assertGreater(len(m.commands), 0, f"{name}: no commands")

    def test_all_have_description(self):
        for name, m in MODULES.items():
            self.assertTrue(m.description.strip(), f"{name}: empty description")

    def test_all_clusters_valid(self):
        """Каждый модуль ссылается на существующий кластер."""
        for name, m in MODULES.items():
            self.assertIn(m.cluster, CLUSTERS, f"{name}: cluster {m.cluster!r} not in CLUSTERS")

    def test_known_clusters_present(self):
        for cid in _KNOWN_CLUSTERS:
            self.assertIn(cid, CLUSTERS)

    def test_hexmat_correct_commands(self):
        """hexmat должен иметь commands=['matrix','rank','kernel'] (не старый ['matrix','det','eigen'])."""
        m = MODULES['hexmat']
        self.assertIn('rank', m.commands)
        self.assertIn('kernel', m.commands)
        self.assertNotIn('eigen', m.commands, "Old duplicate definition leaked through")

    def test_hexpack_commands(self):
        m = MODULES['hexpack']
        for cmd in ('ring', 'antipode', 'fixpoint', 'packable', 'magic', 'periods'):
            self.assertIn(cmd, m.commands)

    def test_hexphi_commands(self):
        m = MODULES['hexphi']
        for cmd in ('fibonacci', 'grid'):
            self.assertIn(cmd, m.commands)

    def test_json_ready_types(self):
        for name, m in MODULES.items():
            self.assertIsInstance(m.json_ready, bool, f"{name}: json_ready not bool")

    def test_known_json_ready(self):
        for name in _KNOWN_JSON_MODULES:
            self.assertIn(name, MODULES, f"{name} missing from MODULES")
            self.assertTrue(MODULES[name].json_ready, f"{name}: expected json_ready=True")


class TestClustersIntegrity(unittest.TestCase):
    def test_cluster_count(self):
        self.assertEqual(len(CLUSTERS), _EXPECTED_CLUSTERS)

    def test_all_are_clusterinfo(self):
        for cid, c in CLUSTERS.items():
            self.assertIsInstance(c, ClusterInfo)

    def test_id_matches_key(self):
        for key, c in CLUSTERS.items():
            self.assertEqual(key, c.id)

    def test_all_have_modules(self):
        for cid, c in CLUSTERS.items():
            self.assertGreater(len(c.modules), 0, f"{cid}: no modules listed")

    def test_all_modules_exist_in_MODULES(self):
        for cid, c in CLUSTERS.items():
            for m in c.modules:
                self.assertIn(m, MODULES, f"Cluster {cid} references unknown module {m!r}")

    def test_all_have_description(self):
        for cid, c in CLUSTERS.items():
            self.assertTrue(c.description.strip(), f"{cid}: empty description")

    def test_all_have_emergent(self):
        for cid, c in CLUSTERS.items():
            self.assertTrue(c.emergent.strip(), f"{cid}: empty emergent")

    def test_k5_includes_hexpack(self):
        self.assertIn('hexpack', CLUSTERS['K5'].modules)

    def test_k7_includes_hexphi(self):
        self.assertIn('hexphi', CLUSTERS['K7'].modules)


class TestSuperclustersIntegrity(unittest.TestCase):
    def test_known_ids_present(self):
        for sid in ('SC-1', 'SC-2', 'SC-3', 'SC-4', 'SC-5', 'SC-6', 'SC-7',
                    'TSC-1', 'TSC-2', 'TSC-3', 'MC'):
            self.assertIn(sid, SUPERCLUSTERS, f"{sid} missing")

    def test_all_are_superclusterinfo(self):
        for sid, sc in SUPERCLUSTERS.items():
            self.assertIsInstance(sc, SuperClusterInfo)

    def test_id_matches_key(self):
        for key, sc in SUPERCLUSTERS.items():
            self.assertEqual(key, sc.id)

    def test_cluster_ids_valid(self):
        for sid, sc in SUPERCLUSTERS.items():
            for cid in sc.cluster_ids:
                self.assertIn(cid, CLUSTERS, f"SC {sid}: unknown cluster {cid!r}")

    def test_all_have_pipeline(self):
        for sid, sc in SUPERCLUSTERS.items():
            self.assertGreater(len(sc.pipeline), 0, f"{sid}: empty pipeline")

    def test_all_have_emergent(self):
        for sid, sc in SUPERCLUSTERS.items():
            self.assertTrue(sc.emergent.strip(), f"{sid}: empty emergent")

    def test_sc7_uses_k5_k7(self):
        sc = SUPERCLUSTERS['SC-7']
        self.assertIn('K7', sc.cluster_ids)
        self.assertIn('K5', sc.cluster_ids)

    def test_mc_uses_all_clusters(self):
        mc = SUPERCLUSTERS['MC']
        for cid in _KNOWN_CLUSTERS:
            self.assertIn(cid, mc.cluster_ids)


class TestHelperFunctions(unittest.TestCase):
    def test_get_module_existing(self):
        m = get_module('hexpack')
        self.assertIsNotNone(m)
        self.assertEqual(m.name, 'hexpack')

    def test_get_module_missing(self):
        self.assertIsNone(get_module('nonexistent'))

    def test_get_cluster_existing(self):
        c = get_cluster('K1')
        self.assertIsNotNone(c)
        self.assertEqual(c.id, 'K1')

    def test_get_cluster_missing(self):
        self.assertIsNone(get_cluster('K9'))

    def test_get_supercluster_existing(self):
        sc = get_supercluster('SC-7')
        self.assertIsNotNone(sc)
        self.assertEqual(sc.id, 'SC-7')

    def test_get_supercluster_missing(self):
        self.assertIsNone(get_supercluster('SC-99'))

    def test_modules_in_cluster_k5(self):
        mods = modules_in_cluster('K5')
        names = [m.name for m in mods]
        self.assertIn('hexpack', names)

    def test_modules_in_cluster_missing(self):
        self.assertEqual(modules_in_cluster('K9'), [])

    def test_cluster_of_module(self):
        c = cluster_of_module('hexpack')
        self.assertIsNotNone(c)
        self.assertEqual(c.id, 'K5')

    def test_cluster_of_module_missing(self):
        self.assertIsNone(cluster_of_module('nonexistent'))

    def test_all_module_names_sorted(self):
        names = all_module_names()
        self.assertEqual(names, sorted(names))

    def test_all_module_names_count(self):
        self.assertEqual(len(all_module_names()), _EXPECTED_MODULES)

    def test_all_cluster_ids_sorted(self):
        ids = all_cluster_ids()
        self.assertEqual(ids, sorted(ids))

    def test_all_cluster_ids_count(self):
        self.assertEqual(len(all_cluster_ids()), _EXPECTED_CLUSTERS)

    def test_all_supercluster_ids_sorted(self):
        ids = all_supercluster_ids()
        self.assertEqual(ids, sorted(ids))

    def test_json_ready_modules_subset(self):
        ready = json_ready_modules()
        self.assertIsInstance(ready, list)
        for name in ready:
            self.assertIn(name, MODULES)
            self.assertTrue(MODULES[name].json_ready)

    def test_json_ready_known_modules(self):
        ready = set(json_ready_modules())
        for name in _KNOWN_JSON_MODULES:
            self.assertIn(name, ready)


if __name__ == '__main__':
    unittest.main(verbosity=2)
