"""Tests for conservative CSV schema drift detection."""
import tempfile
import unittest
from pathlib import Path
from csv_inspector import profile_csv
from csv_inspector.schema import compare_csv_schemas


class CSVSchemaDiffTests(unittest.TestCase):
    def setUp(self):
        self.paths = []

    def tearDown(self):
        for path in self.paths:
            path.unlink(missing_ok=True)

    def profile(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             newline="", suffix=".csv", delete=False)
        with handle:
            handle.write(text)
        path = Path(handle.name)
        self.paths.append(path)
        return profile_csv(path)

    def test_added_column_is_non_breaking(self):
        diff = compare_csv_schemas(self.profile("id,name\n1,Ada\n"),
                                   self.profile("id,name,active\n1,Ada,true\n"))
        self.assertEqual(diff.added_columns, ("active",))
        self.assertFalse(diff.has_breaking_changes)

    def test_removed_column_is_breaking(self):
        diff = compare_csv_schemas(self.profile("id,name\n1,Ada\n"),
                                   self.profile("id\n1\n"))
        self.assertEqual(diff.removed_columns, ("name",))
        self.assertTrue(diff.has_breaking_changes)

    def test_integer_to_number_is_compatible(self):
        diff = compare_csv_schemas(self.profile("amount\n1\n2\n"),
                                   self.profile("amount\n1.5\n2\n"))
        self.assertTrue(diff.type_changes[0].compatible)
        self.assertFalse(diff.has_breaking_changes)

    def test_number_to_string_is_breaking(self):
        diff = compare_csv_schemas(self.profile("amount\n1.5\n2.5\n"),
                                   self.profile("amount\none\ntwo\n"))
        self.assertFalse(diff.type_changes[0].compatible)
        self.assertTrue(diff.has_breaking_changes)

    def test_reordering_is_breaking(self):
        diff = compare_csv_schemas(self.profile("id,name,age\n1,Ada,36\n"),
                                   self.profile("name,id,age\nAda,1,36\n"))
        self.assertTrue(diff.reordered)
        self.assertTrue(diff.has_breaking_changes)

    def test_duplicate_headers_are_ambiguous(self):
        diff = compare_csv_schemas(self.profile("id,name\n1,Ada\n"),
                                   self.profile("id,name,name\n1,Ada,Other\n"))
        self.assertEqual(diff.ambiguous_names, ("name",))
        self.assertEqual(diff.type_changes, ())
        self.assertTrue(diff.has_breaking_changes)

    def test_identical_schemas_have_no_changes(self):
        diff = compare_csv_schemas(self.profile("id,amount\n1,2.5\n"),
                                   self.profile("id,amount\n2,3.5\n"))
        self.assertFalse(diff.has_changes)

    def test_rejects_non_profiles(self):
        profile = self.profile("id\n1\n")
        with self.assertRaises(TypeError):
            compare_csv_schemas(profile, object())


if __name__ == "__main__":
    unittest.main()
