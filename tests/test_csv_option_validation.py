"""Tests for strict CSV profiler option validation."""

import tempfile
import unittest
from pathlib import Path

from csv_inspector import CSVInspectionError, profile_csv


class CSVOptionValidationTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".csv", delete=False)
        with handle:
            handle.write("name,value\nAda,1\n")
        self.path = Path(handle.name)
        self.addCleanup(self.path.unlink, missing_ok=True)

    def test_requires_integer_issue_limit_and_boolean_tracking_flags(self):
        for kwargs in (
            {"max_issues": True},
            {"max_issues": 1.5},
            {"max_issues": 0},
            {"track_distinct": 1},
            {"track_duplicates": "yes"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                profile_csv(self.path, **kwargs)

    def test_requires_string_encoding_and_delimiter(self):
        for encoding in ("", 3):
            with self.subTest(encoding=encoding), self.assertRaises(ValueError):
                profile_csv(self.path, encoding=encoding)
        with self.assertRaises(CSVInspectionError):
            profile_csv(self.path, delimiter=3)

    def test_missing_markers_must_be_an_iterable_of_strings(self):
        for values in ("unknown", (None,), 42):
            with self.subTest(values=values), self.assertRaises(ValueError):
                profile_csv(self.path, missing_values=values)


if __name__ == "__main__":
    unittest.main()
