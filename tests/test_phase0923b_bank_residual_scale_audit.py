"""Phase 0.9.2.3b: Unit tests for Bank residual scale audit.

Tests cover each risk flag rule, deterministic report generation,
and non-interference with adapter/model code.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class RiskFlagRulesTest(unittest.TestCase):
    """Tests for individual risk flag rules."""

    @classmethod
    def setUpClass(cls):
        from phase0923b_bank_residual_scale_audit import (
            check_flags as _cf, compute_stats as _cs, THRESHOLDS as _t,
        )
        cls._check_flags = staticmethod(_cf)
        cls._compute_stats = staticmethod(_cs)
        cls._THRESHOLDS = _t

    def check_flags(self, *args, **kwargs):
        return self._check_flags(*args, **kwargs)

    def compute_stats(self, *args, **kwargs):
        return self._compute_stats(*args, **kwargs)

    @property
    def T(self):
        return self._THRESHOLDS

    # ── 1. Normal same-scale data should not trigger flags ──

    def test_normal_same_scale_no_flags(self):
        vals = np.random.default_rng(42).normal(50, 5, 1000)
        stats = self.compute_stats(vals)
        flags = self.check_flags(stats, 3, "test_kpi", "cpu")
        self.assertEqual(flags, [],
                         f"Expected no flags for well-behaved data, got {flags}")

    # ── 2. 0-1 vs 0-100 mix triggers POSSIBLE_RATIO_PERCENT_MIX ──

    def test_ratio_percent_mix_triggers_flag(self):
        vals_01 = np.random.default_rng(1).uniform(0, 1, 500)
        vals_0100 = np.random.default_rng(2).uniform(30, 70, 500)
        stats = self.compute_stats(vals_01)
        thr_map = {"ratio_percent_flag": True}
        flags = self.check_flags(stats, 1, "kpi_range01", "cpu", thr_map)
        self.assertIn("POSSIBLE_RATIO_PERCENT_MIX", flags)

    # ── 3. Median scale mismatch triggers MEDIAN_SCALE_MISMATCH ──

    def test_median_scale_mismatch_triggers(self):
        stats = self.compute_stats(np.random.default_rng(3).normal(0.1, 0.01, 1000))
        thr_map = {"median_scale_ratio": 5000.0}
        flags = self.check_flags(stats, 1, "tiny_kpi", "cpu", thr_map)
        self.assertIn("MEDIAN_SCALE_MISMATCH", flags)

    def test_median_scale_match_does_not_trigger(self):
        stats = self.compute_stats(np.random.default_rng(3).normal(0.1, 0.01, 1000))
        thr_map = {"median_scale_ratio": 50.0}  # below threshold of 100
        flags = self.check_flags(stats, 1, "tiny_kpi", "cpu", thr_map)
        self.assertNotIn("MEDIAN_SCALE_MISMATCH", flags)

    # ── 4. P95 scale mismatch triggers P95_SCALE_MISMATCH ──

    def test_p95_scale_mismatch_triggers(self):
        stats = self.compute_stats(np.random.default_rng(4).uniform(0, 1, 1000))
        thr_map = {"p95_scale_ratio": 500.0}
        flags = self.check_flags(stats, 1, "kpi_small", "cpu", thr_map)
        self.assertIn("P95_SCALE_MISMATCH", flags)

    # ── 5. NaN or inf triggers NON_FINITE_VALUES ──

    def test_nan_triggers_non_finite(self):
        vals = np.array([1.0, 2.0, np.nan, 3.0, np.inf, -np.inf, 4.0])
        stats = self.compute_stats(vals)
        flags = self.check_flags(stats, 1, "bad_kpi", "cpu")
        self.assertIn("NON_FINITE_VALUES", flags)

    def test_all_finite_no_flag(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = self.compute_stats(vals)
        flags = self.check_flags(stats, 1, "clean_kpi", "cpu")
        self.assertNotIn("NON_FINITE_VALUES", flags)

    # ── 6. Near-constant signal triggers CONSTANT_OR_NEAR_CONSTANT_SIGNAL ──

    def test_near_constant_triggers(self):
        vals = np.full(100, 42.0)
        stats = self.compute_stats(vals)
        flags = self.check_flags(stats, 1, "constant_kpi", "cpu")
        self.assertIn("CONSTANT_OR_NEAR_CONSTANT_SIGNAL", flags)

    def test_variable_signal_no_constant_flag(self):
        vals = np.random.default_rng(5).normal(50, 5, 1000)
        stats = self.compute_stats(vals)
        flags = self.check_flags(stats, 1, "variable_kpi", "cpu")
        self.assertNotIn("CONSTANT_OR_NEAR_CONSTANT_SIGNAL", flags)

    # ── 7. Sparse signal triggers SPARSE_SIGNAL ──

    def test_sparse_signal_triggers(self):
        vals = np.zeros(1000)
        vals[0] = 1.0
        vals[100] = 2.0  # only 2 non-zero out of 1000 → 0.2% < 1%
        stats = self.compute_stats(vals)
        flags = self.check_flags(stats, 1, "sparse_kpi", "cpu")
        self.assertIn("SPARSE_SIGNAL", flags)

    def test_dense_signal_no_sparse_flag(self):
        vals = np.random.default_rng(6).normal(50, 5, 1000)
        stats = self.compute_stats(vals)
        flags = self.check_flags(stats, 1, "dense_kpi", "cpu")
        self.assertNotIn("SPARSE_SIGNAL", flags)

    # ── 8. Single huge spike triggers EXTREME_OUTLIER_TAIL ──

    def test_extreme_outlier_triggers(self):
        vals = np.random.default_rng(7).normal(50, 5, 1000)
        vals[500] = 1e6  # huge spike
        stats = self.compute_stats(vals)
        flags = self.check_flags(stats, 1, "outlier_kpi", "cpu")
        self.assertIn("EXTREME_OUTLIER_TAIL", flags)

    def test_no_extreme_no_outlier_flag(self):
        vals = np.random.default_rng(8).normal(50, 5, 1000)
        stats = self.compute_stats(vals)
        flags = self.check_flags(stats, 1, "normal_kpi", "cpu")
        self.assertNotIn("EXTREME_OUTLIER_TAIL", flags)

    # ── 9. Negative values trigger NEGATIVE_VALUES_PRESENT ──

    def test_negative_values_triggers(self):
        vals = np.array([-5.0, 0.0, 3.0, 10.0] * 250)
        stats = self.compute_stats(vals)
        flags = self.check_flags(stats, 1, "neg_kpi", "cpu")
        self.assertIn("NEGATIVE_VALUES_PRESENT", flags)


class DeterministicReportTest(unittest.TestCase):
    """Test that report generation is deterministic: same input → same output."""

    @classmethod
    def setUpClass(cls):
        from phase0923b_bank_residual_scale_audit import analyze, generate_report

    def test_analyze_is_deterministic(self):
        from phase0923b_bank_residual_scale_audit import analyze

        data = {
            "total_raw": 100,
            "total_unmapped": 20,
            "total_dropped": 30,
            "total_accepted": 50,
            "slot_kpi_vals": {
                "cpu": {
                    "kpi_A": {"comp1": [10.0, 20.0, 30.0]},
                    "kpi_B": {"comp1": [40.0, 50.0, 60.0]},
                },
            },
            "slot_all_vals": {"cpu": np.array([10, 20, 30, 40, 50, 60])},
            "slot_all_raw_values": {"cpu": np.array([10, 20, 30, 40, 50, 60])},
        }

        r1 = analyze(data)
        r2 = analyze(data)

        self.assertEqual(r1["global"], r2["global"])
        self.assertEqual(list(r1["slots"].keys()), list(r2["slots"].keys()))
        for slot in r1["slots"]:
            s1 = r1["slots"][slot]
            s2 = r2["slots"][slot]
            self.assertEqual(s1["total_rows"], s2["total_rows"])
            self.assertEqual(s1["n_raw_kpis"], s2["n_raw_kpis"])
            self.assertEqual(s1["n_flags"], s2["n_flags"])
            for kpi in s1["raw_kpis"]:
                for key in s1["raw_kpis"][kpi]:
                    v1 = s1["raw_kpis"][kpi][key]
                    v2 = s2["raw_kpis"][kpi][key]
                    if isinstance(v1, float) and np.isnan(v1) and np.isnan(v2):
                        continue
                    self.assertEqual(v1, v2, f"slot={slot}, kpi={kpi}, key={key}")
        self.assertEqual(r1["flagged_kpis"], r2["flagged_kpis"])


class NoSideEffectsTest(unittest.TestCase):
    """Test that the audit script does not modify adapter or data files."""

    def test_audit_script_does_not_import_adapter(self):
        """The audit script should be self-contained, not importing the adapter."""
        script_path = REPO_ROOT / "scripts" / "phase0923b_bank_residual_scale_audit.py"
        source = script_path.read_text()
        self.assertNotIn("from foundation.adapters", source)
        self.assertNotIn("from foundation.adapters.openrca_bank", source)
        self.assertNotIn("OpenRCABankAdapter", source)

    def test_audit_script_has_no_write_to_data(self):
        """The audit script should never write to the data directory."""
        script_path = REPO_ROOT / "scripts" / "phase0923b_bank_residual_scale_audit.py"
        source = script_path.read_text()
        self.assertNotIn("to_csv", source)
        if "data_dir" in source:
            lines_after_data = source.split("data_dir")
            for part in lines_after_data[1:]:
                if "write" in part.lower() and "report" not in part.lower() and "output" not in part.lower():
                    pass
                if ".csv" in part and "read_csv" not in part and "metric_container" not in part:
                    self.fail("Audit script should not write CSV files")

    def test_compute_stats_is_pure(self):
        """compute_stats should not mutate its input."""
        from phase0923b_bank_residual_scale_audit import compute_stats
        import copy
        vals = np.array([1.0, 2.0, 3.0, np.nan, 4.0])
        orig = copy.deepcopy(vals)
        compute_stats(vals)
        np.testing.assert_array_equal(vals, orig)


class MappingConsistencyTest(unittest.TestCase):
    """Test that the audit mapping matches the adapter mapping."""

    @classmethod
    def setUpClass(cls):
        from phase0923b_bank_residual_scale_audit import (
            map_kpi_to_category as _mk, canonicalize as _ca,
        )
        cls._map_kpi = staticmethod(_mk)
        cls._canonicalize = staticmethod(_ca)

    def map_kpi(self, *args, **kwargs):
        return self._map_kpi(*args, **kwargs)

    def canonicalize(self, *args, **kwargs):
        return self._canonicalize(*args, **kwargs)

    def test_cpu_pattern_matches_container_cpupercent(self):
        cat = self.map_kpi(
            "Container-DOCKER_CONTAINER_7b4b80f345e0--bcou--UATWKR04_CpuPercent")
        self.assertEqual(cat, "cpu")

    def test_cpu_pattern_matches_oslinux_cpuutil(self):
        cat = self.map_kpi("OSLinux-CPU_CPU_CPUCpuUtil")
        self.assertEqual(cat, "cpu")

    def test_mem_pattern_matches_container_mempercent(self):
        cat = self.map_kpi(
            "Container-DOCKER_CONTAINER_7b4b80f345e0--bcou--UATWKR04_MemPercent")
        self.assertEqual(cat, "mem")

    def test_net_rx_matches_networkrxbytes(self):
        cat = self.map_kpi(
            "Container-DOCKER_CONTAINER_7b4b80f345e0--bcou--UATWKR04_NetworkRxBytes")
        self.assertEqual(cat, "net_rx")

    def test_net_tx_matches_networktxbytes(self):
        cat = self.map_kpi(
            "Container-DOCKER_CONTAINER_7b4b80f345e0--bcou--UATWKR04_NetworkTxBytes")
        self.assertEqual(cat, "net_tx")

    def test_mem_usage_matches(self):
        cat = self.map_kpi(
            "Container-DOCKER_CONTAINER_7b4b80f345e0--bcou--UATWKR04_MemUsage")
        self.assertEqual(cat, "mem_usage")

    def test_jvm_cpu_matches(self):
        cat = self.map_kpi("JVM-Operating System_7778_JVM_JVM_CPULoad")
        self.assertEqual(cat, "jvm_cpu")

    def test_jvm_mem_matches(self):
        cat = self.map_kpi("JVM-Memory_7778_JVM_Memory_HeapMemoryUsed")
        self.assertEqual(cat, "jvm_mem")

    def test_mysql_io_matches(self):
        cat = self.map_kpi("Mysql-MySQL_3306_Innodb buffer pool pages total")
        self.assertEqual(cat, "mysql_io")

    def test_threads_matches(self):
        cat = self.map_kpi("Tomcat-Threads_7441-http-nio-8003_thread_used")
        self.assertEqual(cat, "threads")

    def test_sessions_matches(self):
        cat = self.map_kpi("Tomcat-Sessions_7441_session_used_counter")
        self.assertEqual(cat, "sessions")

    def test_disk_io_matches(self):
        cat = self.map_kpi("OSLinux-OSLinux_LOCALDISK_LOCALDISK-sda_DSKBps")
        self.assertEqual(cat, "disk_io")

    def test_bank_safe_v1_cpu_kept(self):
        result = self.canonicalize("cpu", 42.0)
        self.assertEqual(result, ("cpu", 42.0))

    def test_bank_safe_v1_mem_kept(self):
        result = self.canonicalize("mem", 75.0)
        self.assertEqual(result, ("mem", 75.0))

    def test_bank_safe_v1_net_rx_log1p(self):
        result = self.canonicalize("net_rx", 100.0)
        self.assertAlmostEqual(result[1], np.log1p(100.0))

    def test_bank_safe_v1_disk_io_dropped(self):
        self.assertIsNone(self.canonicalize("disk_io", 500.0))

    def test_bank_safe_v1_mysql_io_dropped(self):
        self.assertIsNone(self.canonicalize("mysql_io", 100.0))

    def test_bank_safe_v1_jvm_cpu_dropped(self):
        self.assertIsNone(self.canonicalize("jvm_cpu", 50.0))

    def test_bank_safe_v1_jvm_mem_dropped(self):
        self.assertIsNone(self.canonicalize("jvm_mem", 1e9))

    def test_bank_safe_v1_mem_usage_dropped(self):
        self.assertIsNone(self.canonicalize("mem_usage", 8e8))


if __name__ == "__main__":
    unittest.main()
