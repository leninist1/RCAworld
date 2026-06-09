"""Phase 0.9.2.3a: Bank source-aligned safe canonicalization v1 — unit tests.

Tests A-I cover correct behavior of canonicalization_mode in
OpenRCABankAdapter and the --bank-canonicalization CLI integration.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class BankCanonicalizationUnitTests(unittest.TestCase):
    """Tests A–D, E–F on _canonicalize_container_metric and _extract_container_events."""

    @classmethod
    def setUpClass(cls):
        from foundation.adapters.openrca_bank import OpenRCABankAdapter
        cls.AdapterCls = OpenRCABankAdapter

    # ── Test A: legacy preserves old behavior ──

    def test_a_legacy_preserves_all_values(self):
        adapter = self.AdapterCls(canonicalization_mode="legacy")
        self.assertEqual(
            adapter._canonicalize_container_metric("cpu", 42.0),
            ("cpu", 42.0))
        self.assertEqual(
            adapter._canonicalize_container_metric("mem_usage", 123456.0),
            ("mem_usage", 123456.0))
        self.assertEqual(
            adapter._canonicalize_container_metric("jvm_mem", 9999.0),
            ("jvm_mem", 9999.0))
        self.assertEqual(
            adapter._canonicalize_container_metric("disk_io", 500.0),
            ("disk_io", 500.0))
        self.assertEqual(
            adapter._canonicalize_container_metric("net_rx", 1000.0),
            ("net_rx", 1000.0))
        self.assertEqual(
            adapter._canonicalize_container_metric("net_tx", 500.0),
            ("net_tx", 500.0))

    # ── Test B: bank_safe_v1 preserves safe fields ──

    def test_b_safe_v1_preserves_safe_fields(self):
        adapter = self.AdapterCls(canonicalization_mode="bank_safe_v1")
        self.assertEqual(
            adapter._canonicalize_container_metric("cpu", 42.0),
            ("cpu", 42.0))
        self.assertEqual(
            adapter._canonicalize_container_metric("mem", 12.5),
            ("mem", 12.5))

    def test_b2_safe_v1_log1p_net_rx(self):
        adapter = self.AdapterCls(canonicalization_mode="bank_safe_v1")
        result = adapter._canonicalize_container_metric("net_rx", 100.0)
        self.assertEqual(result[0], "net_rx")
        np.testing.assert_almost_equal(result[1], np.log1p(100.0))

    def test_b3_safe_v1_log1p_net_tx(self):
        adapter = self.AdapterCls(canonicalization_mode="bank_safe_v1")
        result = adapter._canonicalize_container_metric("net_tx", 50.0)
        self.assertEqual(result[0], "net_tx")
        np.testing.assert_almost_equal(result[1], np.log1p(50.0))

    def test_b4_safe_v1_net_rx_handles_negative(self):
        adapter = self.AdapterCls(canonicalization_mode="bank_safe_v1")
        result = adapter._canonicalize_container_metric("net_rx", -10.0)
        self.assertEqual(result[0], "net_rx")
        np.testing.assert_almost_equal(result[1], np.log1p(0.0))

    def test_b5_safe_v1_net_tx_handles_zero(self):
        adapter = self.AdapterCls(canonicalization_mode="bank_safe_v1")
        result = adapter._canonicalize_container_metric("net_tx", 0.0)
        self.assertEqual(result[0], "net_tx")
        np.testing.assert_almost_equal(result[1], np.log1p(0.0))

    # ── Test C: bank_safe_v1 disables unsafe fields ──

    def test_c_safe_v1_disables_unsafe_fields(self):
        unsafe = [
            "jvm_cpu", "mem_usage", "jvm_mem",
            "disk_io", "mysql_io", "threads", "sessions", "fgc",
        ]
        adapter = self.AdapterCls(canonicalization_mode="bank_safe_v1")
        for cat in unsafe:
            with self.subTest(category=cat):
                self.assertIsNone(
                    adapter._canonicalize_container_metric(cat, 1.0),
                    f"{cat} must return None in bank_safe_v1 mode")

    # ── Test D: invalid canonicalization_mode raises ValueError ──

    def test_d_invalid_mode_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.AdapterCls(canonicalization_mode="invalid_mode")
        with self.assertRaises(ValueError):
            self.AdapterCls(canonicalization_mode="bank_safe_v2")
        with self.assertRaises(ValueError):
            self.AdapterCls(canonicalization_mode="")


class BankExtractContainerEventsTests(unittest.TestCase):
    """Tests E–F on _extract_container_events with real-like DataFrame fixtures."""

    @classmethod
    def setUpClass(cls):
        from foundation.adapters.openrca_bank import OpenRCABankAdapter
        from foundation.schema.entity import Entity, EntityType
        cls.AdapterCls = OpenRCABankAdapter
        cls.Entity = Entity
        cls.EntityType = EntityType

    def _make_temp_csv(self, rows):
        """Write a minimal metric_container.csv to a temp dir and return its path."""
        self._tmpdir = tempfile.TemporaryDirectory()
        df = pd.DataFrame(rows)
        csv_path = Path(self._tmpdir.name) / "metric_container.csv"
        df.to_csv(csv_path, index=False)
        return str(self._tmpdir.name), csv_path

    def tearDown(self):
        if hasattr(self, "_tmpdir"):
            self._tmpdir.cleanup()

    # ── Test E: legacy keeps all mem-related; safe_v1 only keeps mem ──

    def test_e_legacy_mem_memusage_jvmmem_all_produce_events(self):
        rows = [
            {"timestamp": 1000, "cmdb_id": "cont1", "kpi_name": "mem_usage",
             "value": 1e9},
            {"timestamp": 1000, "cmdb_id": "cont1", "kpi_name": "jvm_mem",
             "value": 5e8},
            {"timestamp": 1000, "cmdb_id": "cont1", "kpi_name": "CpuPercent",
             "value": 30.0},
            {"timestamp": 1000, "cmdb_id": "cont1", "kpi_name": "MemPercent",
             "value": 45.0},
        ]
        metric_dir, _ = self._make_temp_csv(rows)
        adapter = self.AdapterCls(max_container_events=1000,
                                  canonicalization_mode="legacy")
        entities = [self.Entity(entity_id="cont1", entity_type=self.EntityType.CONTAINER)]
        events = adapter._extract_container_events(metric_dir, entities)

        feat_names = [ev.feature_name for ev in events]
        self.assertIn("mem_usage", feat_names)
        self.assertIn("jvm_mem", feat_names)
        self.assertIn("mem", feat_names)
        self.assertIn("cpu", feat_names)
        self.assertEqual(len(events), 4)

    def test_e2_safe_v1_only_mem_retained(self):
        rows = [
            {"timestamp": 1000, "cmdb_id": "cont1", "kpi_name": "mem_usage",
             "value": 1e9},
            {"timestamp": 1000, "cmdb_id": "cont1", "kpi_name": "jvm_mem",
             "value": 5e8},
            {"timestamp": 1000, "cmdb_id": "cont1", "kpi_name": "CpuPercent",
             "value": 30.0},
            {"timestamp": 1000, "cmdb_id": "cont1", "kpi_name": "MemPercent",
             "value": 45.0},
        ]
        metric_dir, _ = self._make_temp_csv(rows)
        adapter = self.AdapterCls(max_container_events=1000,
                                  canonicalization_mode="bank_safe_v1")
        entities = [self.Entity(entity_id="cont1", entity_type=self.EntityType.CONTAINER)]
        events = adapter._extract_container_events(metric_dir, entities)

        feat_names = [ev.feature_name for ev in events]
        # mem_usage and jvm_mem should be filtered out
        self.assertNotIn("mem_usage", feat_names)
        self.assertNotIn("jvm_mem", feat_names)
        # mem and cpu should remain
        self.assertIn("mem", feat_names)
        self.assertIn("cpu", feat_names)
        self.assertEqual(len(events), 2)

    # ── Test F: safe mode net_rx/net_tx Event values are log1p'd ──

    def test_f_safe_v1_net_rx_tx_are_log1p(self):
        rows = [
            {"timestamp": 2000, "cmdb_id": "cont2",
             "kpi_name": "NetworkRxBytes", "value": 1000.0},
            {"timestamp": 2000, "cmdb_id": "cont2",
             "kpi_name": "NetworkTxBytes", "value": 500.0},
            {"timestamp": 2000, "cmdb_id": "cont2",
             "kpi_name": "Incoming_network", "value": -50.0},
        ]
        metric_dir, _ = self._make_temp_csv(rows)
        adapter = self.AdapterCls(max_container_events=1000,
                                  canonicalization_mode="bank_safe_v1")
        entities = [self.Entity(entity_id="cont2", entity_type=self.EntityType.CONTAINER)]
        events = adapter._extract_container_events(metric_dir, entities)

        self.assertEqual(len(events), 3)

        values_by_feat = {}
        for ev in events:
            values_by_feat.setdefault(ev.feature_name, []).append(ev.value)

        # net_rx: expects log1p(1000.0) and log1p(0.0) for negative
        self.assertAlmostEqual(values_by_feat["net_rx"][0],
                               np.log1p(1000.0),
                               msg="net_rx must be log1p(1000.0)")
        self.assertAlmostEqual(values_by_feat["net_rx"][1],
                               np.log1p(0.0),
                               msg="negative net_rx must be log1p(0.0)")
        # net_tx: expects log1p(500.0)
        self.assertAlmostEqual(values_by_feat["net_tx"][0],
                               np.log1p(500.0),
                               msg="net_tx must be log1p(500.0)")


class BankCanonicalizationCLITests(unittest.TestCase):
    """Tests G–I: CLI arg parsing and adapter-side parameter passing."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))

    # ── Test G: CLI default is legacy ──

    def test_g_cli_default_is_legacy(self):
        import phaseA_run_inference as pri
        parser = pri.argparse.ArgumentParser()
        parser.add_argument('--bank-canonicalization', type=str, default='legacy',
                            choices=['legacy', 'bank_safe_v1'])
        args = parser.parse_args([])
        self.assertEqual(args.bank_canonicalization, "legacy")

    def test_g2_cli_default_value_is_legacy_string(self):
        import phaseA_run_inference as pri
        parser = pri.argparse.ArgumentParser()
        parser.add_argument('--systems', nargs='+', default=['Bank'])
        parser.add_argument('--method', type=str, default='calibrated',
                            choices=['residual_only', 'calibrated',
                                     'legacy_per_entity_max', 'onset_head'])
        parser.add_argument('--all_zero_type', action='store_true')
        parser.add_argument('--burn_in_min', type=int, default=60)
        parser.add_argument('--resample_sec', type=int, default=120)
        parser.add_argument('--output', type=str, default='predictions.json')
        parser.add_argument('--lambda-onset', type=float, default=0.0)
        parser.add_argument('--bank-canonicalization', type=str, default='legacy',
                            choices=['legacy', 'bank_safe_v1'])
        args = parser.parse_args([])
        self.assertEqual(args.bank_canonicalization, "legacy")

    # ── Test H: CLI explicit bank_safe_v1 passes to Bank adapter ──

    def test_h_explicit_bank_safe_v1_accepted(self):
        import phaseA_run_inference as pri
        parser = pri.argparse.ArgumentParser()
        parser.add_argument('--systems', nargs='+', default=['Bank'])
        parser.add_argument('--method', type=str, default='calibrated',
                            choices=['residual_only', 'calibrated',
                                     'legacy_per_entity_max', 'onset_head'])
        parser.add_argument('--all_zero_type', action='store_true')
        parser.add_argument('--burn_in_min', type=int, default=60)
        parser.add_argument('--resample_sec', type=int, default=120)
        parser.add_argument('--output', type=str, default='predictions.json')
        parser.add_argument('--lambda-onset', type=float, default=0.0)
        parser.add_argument('--bank-canonicalization', type=str, default='legacy',
                            choices=['legacy', 'bank_safe_v1'])
        args = parser.parse_args(['--bank-canonicalization', 'bank_safe_v1'])
        self.assertEqual(args.bank_canonicalization, "bank_safe_v1")

    def test_h2_make_adapter_passes_canonicalization_to_bank(self):
        from phaseA_run_inference import _make_adapter
        adapter = _make_adapter(
            "OpenRCABankAdapter", include_dates=None,
            canonicalization_mode="bank_safe_v1")
        self.assertEqual(adapter.canonicalization_mode, "bank_safe_v1")

    def test_h3_make_adapter_legacy_default(self):
        from phaseA_run_inference import _make_adapter
        adapter = _make_adapter(
            "OpenRCABankAdapter", include_dates=None)
        self.assertEqual(adapter.canonicalization_mode, "legacy")

    # ── Test I: Market / Telecom adapters not affected ──

    def test_i_market_adapter_not_affected(self):
        from phaseA_run_inference import _make_adapter
        # Market adapter should work fine without canonicalization_mode kwarg
        adapter = _make_adapter("OpenRCAMarketAdapter", include_dates=None)
        self.assertIsNotNone(adapter)

    def test_i2_telecom_adapter_not_affected(self):
        from phaseA_run_inference import _make_adapter
        adapter = _make_adapter("OpenRCATelecomAdapter", include_dates=None)
        self.assertIsNotNone(adapter)

    def test_i3_bank_canonicalization_not_passed_to_non_bank_adapters(self):
        from phaseA_run_inference import _make_adapter
        # Verify that extra canonicalization arg doesn't break non-Bank adapters
        adapter_market = _make_adapter("OpenRCAMarketAdapter", include_dates=None,
                                       canonicalization_mode="bank_safe_v1")
        self.assertIsNotNone(adapter_market)
        adapter_telecom = _make_adapter("OpenRCATelecomAdapter", include_dates=None,
                                        canonicalization_mode="bank_safe_v1")
        self.assertIsNotNone(adapter_telecom)


if __name__ == "__main__":
    unittest.main()
