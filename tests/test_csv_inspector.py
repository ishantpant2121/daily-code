"""Tests for CSV profiling, inference, diagnostics, and the CLI."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from csv_inspector import CSVInspectionError, profile_csv
from csv_inspector.cli import main
from csv_inspector.inference import classify_value, resolve_type


class TypeInferenceTests(unittest.TestCase):
    def test_scalar_kinds(self):
        cases = {
            "42": "integer", "-3.5": "number", "TRUE": "boolean",
            "2026-11-04": "date", "2026-11-04T21:00:00": "datetime",
            "plain text": "string",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(classify_value(value), expected)

    def test_type_promotion(self):
        self.assertEqual(resolve_type(["integer", "number"]), "number")
        self.assertEqual(resolve_type(["integer", "string"]), "mixed")
        self.assertEqual(resolve_type([]), "empty")


class ProfileTests(unittest.TestCase):
    def write_csv(self, content: str, encoding: str = "utf-8") -> Path:
        handle = tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline="", suffix=".csv", delete=False)
        with handle:
            handle.write(content)
        path = Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_semicolon_detection_multiline_and_numeric_range(self):
        path = self.write_csv('id;note;price\n1;"first line\nsecond line";2.5\n2;plain;4\n')
        result = profile_csv(path)
        self.assertEqual((result.delimiter, result.row_count, result.column_count), (";", 2, 3))
        self.assertEqual(result.columns[2].inferred_type, "number")
        self.assertEqual((result.columns[2].min_value, result.columns[2].max_value), ("2.5", "4"))
        self.assertFalse(result.has_issues)

    def test_bom_and_missing_values(self):
        path = self.write_csv("name,age\nAda,36\nGrace,\n", encoding="utf-8-sig")
        result = profile_csv(path)
        self.assertEqual(result.columns[0].name, "name")
        self.assertEqual(result.columns[1].null_count, 1)

    def test_header_and_ragged_row_diagnostics(self):
        path = self.write_csv("id,id,\n1,2,3\n4,5\n")
        result = profile_csv(path, delimiter=",")
        codes = {issue.code for issue in result.issues}
        self.assertTrue({"duplicate_header", "blank_header", "row_width_mismatch"} <= codes)
        self.assertEqual(result.issue_count, 3)

    def test_issue_storage_is_bounded(self):
        path = self.write_csv("a,b\n1\n2\n3\n")
        result = profile_csv(path, max_issues=1)
        self.assertEqual((len(result.issues), result.issue_count, result.issues_truncated), (1, 3, 2))

    def test_distinct_tracking_can_be_disabled(self):
        path = self.write_csv("tag\na\na\nb\n")
        result = profile_csv(path, track_distinct=False)
        self.assertIsNone(result.columns[0].distinct_count)

    def test_duplicate_rows_are_opt_in_and_exact(self):
        path = self.write_csv("id,name\n1,Ada\n2,Grace\n1,Ada\n1, Ada\n2,Grace\n")
        self.assertIsNone(profile_csv(path).duplicate_row_count)
        result = profile_csv(path, track_duplicates=True)
        self.assertEqual(result.duplicate_row_count, 2)

    def test_custom_missing_markers_are_additive_and_case_insensitive(self):
        path = self.write_csv("status\nUNKNOWN\n - \nNA\nok\n")
        result = profile_csv(path, missing_values=("unknown", "-"))
        self.assertEqual(result.columns[0].null_count, 3)

    def test_empty_file_and_invalid_delimiter(self):
        with self.assertRaises(CSVInspectionError):
            profile_csv(self.write_csv(""))
        with self.assertRaises(CSVInspectionError):
            profile_csv(self.write_csv("a,b\n1,2\n"), delimiter="||")

    def test_rejects_unclosed_quoted_field(self):
        path = self.write_csv('name,note\nAda,"not closed\n')
        with self.assertRaisesRegex(CSVInspectionError, "malformed CSV"):
            profile_csv(path)


class CLITests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".csv", delete=False)
        with handle:
            handle.write("id,value\n1,10\n2,\n")
        self.path = Path(handle.name)
        self.addCleanup(self.path.unlink, missing_ok=True)

    def test_text_and_json_output(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main([str(self.path)])
        self.assertEqual(status, 0)
        self.assertIn("CSV profile:", output.getvalue())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main([str(self.path), "--json"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["row_count"], 2)

    def test_duplicate_and_custom_missing_cli_options(self):
        sample = self.path.with_name(self.path.stem + "-duplicates.csv")
        sample.write_text("status\nunknown\nunknown\nok\n", encoding="utf-8")
        self.addCleanup(sample.unlink, missing_ok=True)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main([str(sample), "--duplicates", "--missing-value", "unknown", "--json"])
        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["duplicate_row_count"], 1)
        self.assertEqual(report["columns"][0]["null_count"], 2)

    def test_strict_and_error_exit_codes(self):
        broken = self.path.with_name(self.path.stem + "-broken.csv")
        broken.write_text("id,value\n1\n", encoding="utf-8")
        self.addCleanup(broken.unlink, missing_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(broken), "--strict"]), 1)
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertEqual(main([str(self.path) + ".missing"]), 2)
        self.assertIn("csv-inspector: error:", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
