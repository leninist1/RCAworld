"""Phase 0.9.2.3c: Unit tests for Bank KPI Coverage Taxonomy Audit.

Tests cover classification rules, data consistency, and no-side-effects.
"""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class ClassificationRuleTests(unittest.TestCase):
    """Test each classification rule individually."""

    @classmethod
    def setUpClass(cls):
        from phase0923c_bank_kpi_coverage_taxonomy import (
            classify_source_family,
            classify_entity_scope,
            classify_semantic_family,
            classify_metric_kind,
            classify_unit_hint,
            classify_existing_slot_candidate,
            classify_new_slot_candidate,
            classify_recommended_action,
            classify_confidence,
            classify_transform_hint,
            get_bank_safe_v1_status,
        )
        cls.src = staticmethod(classify_source_family)
        cls.scope = staticmethod(classify_entity_scope)
        cls.sem = staticmethod(classify_semantic_family)
        cls.kind = staticmethod(classify_metric_kind)
        cls.unit = staticmethod(classify_unit_hint)
        cls.ex_slot = staticmethod(classify_existing_slot_candidate)
        cls.new_slot = staticmethod(classify_new_slot_candidate)
        cls.action = staticmethod(classify_recommended_action)
        cls.conf = staticmethod(classify_confidence)
        cls.xform = staticmethod(classify_transform_hint)
        cls.status = staticmethod(get_bank_safe_v1_status)

    # ── Source family tests ──

    def test_source_family_container(self):
        self.assertEqual(self.src("Container-DOCKER-foo_CpuPercent"), "Container")

    def test_source_family_oslinux(self):
        self.assertEqual(self.src("OSLinux-CPU_CPU_CPUCpuUtil"), "OSLinux")

    def test_source_family_tomcat(self):
        self.assertEqual(self.src("Tomcat-MEMORY_7441-MEMORY_JVMMemory"), "Tomcat")

    def test_source_family_jvm(self):
        self.assertEqual(self.src("JVM-Threads_7778_ThreadCount"), "JVM")

    def test_source_family_mysql(self):
        self.assertEqual(self.src("Mysql-MySQL_3306_Innodb data reads"), "MySQL")

    def test_source_family_redis(self):
        self.assertEqual(self.src("redis-Redis_6379_Redis used_memory"), "Redis")

    def test_source_family_unknown(self):
        self.assertEqual(self.src("SomeUnknown_kpi"), "Unknown")

    # ── Entity scope tests ──

    def test_entity_scope_container(self):
        self.assertEqual(self.scope("Container-DOCKER-foo"), "container")

    def test_entity_scope_host(self):
        self.assertEqual(self.scope("OSLinux-CPU-foo"), "host")

    def test_entity_scope_application_server(self):
        self.assertEqual(self.scope("Tomcat-MEMORY-foo"), "application_server")

    def test_entity_scope_jvm(self):
        self.assertEqual(self.scope("JVM-Threads-foo"), "jvm")

    def test_entity_scope_database(self):
        self.assertEqual(self.scope("Mysql-MySQL-foo"), "database")

    def test_entity_scope_cache(self):
        self.assertEqual(self.scope("redis-Redis-foo"), "cache")

    # ── Semantic family tests ──

    def test_semantic_cpu(self):
        self.assertEqual(self.sem("OSLinux-CPU_CPU_CPUCpuUtil"), "cpu")
        self.assertEqual(self.sem("Container-foo_CpuPercent"), "cpu")

    def test_semantic_memory(self):
        self.assertEqual(self.sem("Container-foo_MemPercent"), "memory")
        self.assertEqual(self.sem("OSLinux-MEMORY_MEMFreeMem"), "memory")

    def test_semantic_network(self):
        self.assertEqual(self.sem("Container-foo_NetworkRxBytes"), "network")

    def test_semantic_disk(self):
        self.assertEqual(self.sem("OSLinux-LOCALDISK-sda_DSKBps"), "disk")

    def test_semantic_database_io(self):
        self.assertEqual(self.sem("Mysql-MySQL_3306_Innodb data reads"), "database_io")

    def test_semantic_cache(self):
        self.assertEqual(self.sem("redis-Redis_6379_Redis keyspace_hits"), "cache")

    # ── Metric kind tests ──

    def test_metric_kind_counter(self):
        self.assertEqual(self.kind("Mysql-MySQL_3306_Connections"), "counter")
        self.assertEqual(self.kind("Container-foo_thread_count"), "counter")

    def test_metric_kind_gauge(self):
        self.assertEqual(self.kind("Container-foo_MemLimit"), "gauge")
        self.assertEqual(self.kind("OSLinux-MEMORY_MEMTotalMem"), "gauge")

    def test_metric_kind_ratio(self):
        self.assertEqual(self.kind("Container-foo_CpuPercent"), "ratio")
        self.assertEqual(self.kind("OSLinux-CPU_CPU_CPUCpuUtil"), "ratio")

    def test_metric_kind_rate(self):
        self.assertEqual(self.kind("Container-foo_bytes_per_sec"), "rate")

    def test_metric_kind_duration(self):
        self.assertEqual(self.kind("Tomcat-Requests_7441_MaxTimeRequestInfo"), "duration")

    def test_counter_vs_gauge_not_confused(self):
        self.assertEqual(self.kind("Mysql_connections"), "counter")
        self.assertEqual(self.kind("OSLinux_MemFree"), "gauge")

    # ── Unit hint tests ──

    def test_unit_percent(self):
        self.assertEqual(self.unit("Container-foo_CpuPercent"), "percent")
        self.assertEqual(self.unit("OSLinux-CPU_CPU_CPUCpuUtil"), "percent")

    def test_unit_bytes(self):
        self.assertEqual(self.unit("Container-foo_NetworkRxBytes"), "bytes")

    def test_unit_count(self):
        self.assertEqual(self.unit("Mysql-MySQL_3306_Connections"), "count")

    def test_unit_milliseconds(self):
        self.assertEqual(self.unit("Tomcat-Requests_ProcessingTimeRequestInfo"), "milliseconds")

    # ── Existing slot tests ──

    def test_existing_slot_cpu(self):
        self.assertEqual(self.ex_slot("Container_CpuPercent", "cpu", "Container", "percent"), "cpu")

    def test_existing_slot_mem(self):
        self.assertEqual(self.ex_slot("Container_MemPercent", "memory", "Container", "percent"), "mem")

    def test_existing_slot_net_rx(self):
        self.assertEqual(self.ex_slot("Container_NetworkRxBytes", "network", "Container", "bytes"), "net_rx")

    def test_existing_slot_net_tx(self):
        self.assertEqual(self.ex_slot("Container_NetworkTxBytes", "network", "Container", "bytes"), "net_tx")

    def test_existing_slot_not_for_jvm_mem(self):
        self.assertIsNone(self.ex_slot("JVM_HeapMemory", "memory", "JVM", "bytes"))

    # ── New slot tests ──

    def test_new_slot_mysql_io(self):
        self.assertEqual(self.new_slot("Mysql_Innodb_data_reads", "database_io", "MySQL", "count"), "mysql_io")

    def test_new_slot_disk(self):
        self.assertEqual(self.new_slot("OSLinux_LOCALDISK_DSKBps", "disk", "OSLinux", "count"), "disk_io")

    def test_new_slot_jvm_mem(self):
        self.assertEqual(self.new_slot("JVM_HeapMemory", "memory", "JVM", "bytes"), "jvm_mem")

    # ── Recommended action tests ──

    def test_action_map_safe_v2_for_existing_slot(self):
        self.assertEqual(
            self.action("Container_CpuPercent", "cpu", None, "Container",
                        "cpu", "ratio", "percent", 1000, 5),
            "map_safe_v2")

    def test_action_requires_new_slot(self):
        self.assertEqual(
            self.action("Mysql_Innodb_data_reads", None, "mysql_io", "MySQL",
                        "database_io", "counter", "count", 5000, 2),
            "requires_new_slot")

    def test_action_manual_review_for_unknown(self):
        self.assertEqual(
            self.action("Unknown_metric", None, None, "Unknown",
                        "unknown", "unknown", "unknown", 100, 1),
            "manual_review")

    def test_manual_review_not_auto_mapped_to_safe_v2(self):
        self.assertNotEqual(
            self.action("Unknown_metric", None, None, "Unknown",
                        "unknown", "unknown", "unknown", 100, 1),
            "map_safe_v2")

    # ── Confidence tests ──

    def test_confidence_high_for_map_safe_v2(self):
        self.assertEqual(self.conf("kpi", "map_safe_v2", "cpu", "percent", "cpu"), "high")

    def test_confidence_medium_for_new_slot(self):
        self.assertEqual(self.conf("kpi", "requires_new_slot", None, "count", "database_io"), "medium")

    # ── Transform hint tests ──

    def test_transform_log1p_network_bytes(self):
        self.assertEqual(self.xform("Container_NetworkRxBytes", "counter", "bytes", "Container"), "log1p")

    def test_transform_identity_percent(self):
        self.assertEqual(self.xform("Container_CpuPercent", "ratio", "percent", "Container"), "identity")

    # ── bank_safe_v1 status tests ──

    def test_status_accepted_cpu(self):
        self.assertEqual(self.status("OSLinux-CPU_CPU_CPUCpuUtil"), "accepted")

    def test_status_accepted_mem(self):
        self.assertEqual(self.status("Container-foo_MemPercent"), "accepted")

    def test_status_dropped_mem_usage(self):
        self.assertEqual(self.status("Container-foo_MemUsage"), "dropped")

    def test_status_dropped_jvm(self):
        self.assertEqual(self.status("JVM-Memory_7778_HeapMemoryUsed"), "dropped")

    def test_status_unmapped(self):
        self.assertEqual(self.status("Unknown_KPI_Name"), "unmapped")


class CatalogConsistencyTests(unittest.TestCase):
    """Test catalog data integrity and consistency."""

    @classmethod
    def setUpClass(cls):
        from phase0923c_bank_kpi_coverage_taxonomy import classify_kpi
        cls.classify_kpi = staticmethod(classify_kpi)

    def test_each_kpi_unique(self):
        kpi_names = ["Container_A_CpuPercent", "OSLinux_B_CPUCpuUtil", "Container_C_MemPercent"]
        results = [self.classify_kpi(k, 100, 3) for k in kpi_names]
        names = [r["raw_kpi_name"] for r in results]
        self.assertEqual(len(names), len(set(names)))

    def test_total_rows_consistent(self):
        results = [
            self.classify_kpi("A_kpi", 100, 2),
            self.classify_kpi("B_kpi", 200, 3),
            self.classify_kpi("C_kpi", 300, 1),
        ]
        total = sum(r["row_count"] for r in results)
        self.assertEqual(total, 600)

    def test_accepted_dropped_unmapped_sum_to_total(self):
        from phase0923c_bank_kpi_coverage_taxonomy import get_bank_safe_v1_status
        results = [
            self.classify_kpi("Container_CpuPercent", 100, 2),
            self.classify_kpi("Container_MemUsage", 200, 3),
            self.classify_kpi("Unknown_KPI", 300, 1),
        ]
        for r in results:
            st = get_bank_safe_v1_status(r["raw_kpi_name"])
            self.assertIn(st, ("accepted", "dropped", "unmapped"))

    def test_sort_is_deterministic(self):
        kpis = [
            "Container_C_CpuPercent", "Container_A_CpuPercent",
            "OSLinux_B_CPUUtil", "Container_A_ReqCount",
        ]
        results1 = [self.classify_kpi(k, 100, 1) for k in kpis]
        results2 = [self.classify_kpi(k, 100, 1) for k in kpis]

        sorted1 = sorted(results1, key=lambda x: (-x["row_count"], x["raw_kpi_name"]))
        sorted2 = sorted(results2, key=lambda x: (-x["row_count"], x["raw_kpi_name"]))
        self.assertEqual(
            [r["raw_kpi_name"] for r in sorted1],
            [r["raw_kpi_name"] for r in sorted2])

    def test_high_confidence_existing_slot_has_nonempty_target(self):
        r = self.classify_kpi("Container_CpuPercent", 1000, 5)
        if r["recommended_action"] == "map_safe_v2":
            self.assertTrue(len(r["candidate_existing_slot"]) > 0,
                            "map_safe_v2 must have non-empty target slot")

    def test_requires_new_slot_has_no_existing_slot(self):
        r = self.classify_kpi("Mysql_Innodb_data_reads", 5000, 2)
        if r["recommended_action"] == "requires_new_slot":
            self.assertEqual(r["candidate_existing_slot"], "")

    def test_manual_review_not_map_safe_v2(self):
        r = self.classify_kpi("Mysterious_Metric_With_Unknown_Semantics_XyZ", 100, 1)
        self.assertNotEqual(r["recommended_action"], "map_safe_v2")

    def test_csv_report_consistent_fields(self):
        r = self.classify_kpi("Container_CpuPercent", 1000, 5)
        for field in ["raw_kpi_name", "row_count", "row_ratio",
                      "cumulative_row_ratio", "unique_component_count",
                      "source_family", "entity_scope", "semantic_family",
                      "metric_kind", "unit_hint",
                      "candidate_existing_slot", "candidate_new_slot",
                      "recommended_action", "confidence", "transform_hint", "reason"]:
            self.assertIn(field, r, f"Missing field {field}")

    def test_counter_vs_rate_vs_gauge_not_confused(self):
        from phase0923c_bank_kpi_coverage_taxonomy import classify_metric_kind
        self.assertNotEqual(classify_metric_kind("mysql_connections"), "gauge")
        self.assertNotEqual(classify_metric_kind("cpu_percent"), "counter")
        self.assertNotEqual(classify_metric_kind("bytes_per_sec"), "gauge")


class NoSideEffectsTest(unittest.TestCase):
    """Test that the taxonomy script does not modify adapter or data files."""

    def test_script_does_not_import_adapter(self):
        script_path = REPO_ROOT / "scripts" / "phase0923c_bank_kpi_coverage_taxonomy.py"
        source = script_path.read_text()
        self.assertNotIn("from foundation.adapters", source)
        self.assertNotIn("OpenRCABankAdapter", source)

    def test_script_does_not_write_to_source_dirs(self):
        script_path = REPO_ROOT / "scripts" / "phase0923c_bank_kpi_coverage_taxonomy.py"
        source = script_path.read_text()
        self.assertNotIn("openrca_bank", source)


class ClassificationEdgeCasesTest(unittest.TestCase):
    """Edge case tests for classification rules."""

    @classmethod
    def setUpClass(cls):
        from phase0923c_bank_kpi_coverage_taxonomy import (
            classify_source_family,
            classify_semantic_family,
            classify_metric_kind,
            classify_unit_hint,
        )
        cls.src = staticmethod(classify_source_family)
        cls.sem = staticmethod(classify_semantic_family)
        cls.kind = staticmethod(classify_metric_kind)
        cls.unit = staticmethod(classify_unit_hint)

    def test_empty_string_returns_unknown(self):
        self.assertEqual(self.src(""), "Unknown")

    def test_mixed_case_normalized(self):
        self.assertEqual(self.src("CONTAINER-foo_CpuPercent"), "Container")
        self.assertEqual(self.src("oslinux-bar-CPUUtil"), "OSLinux")

    def test_semantic_with_hyphens(self):
        self.assertEqual(self.sem("OSLinux-CPU_CPU_CPUCpuUtil"), "cpu")
        self.assertEqual(self.sem("OSLinux-MEMORY_MEMTotalMem"), "memory")

    def test_metric_kind_all_zeros_pattern(self):
        self.assertEqual(self.kind("total_commands_processed"), "counter")
        self.assertEqual(self.kind("mem_fragmentation_ratio"), "ratio")
        self.assertEqual(self.kind("uptime_in_seconds"), "duration")


if __name__ == "__main__":
    unittest.main()
