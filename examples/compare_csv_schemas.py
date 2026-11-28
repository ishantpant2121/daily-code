"""Compare two CSV exports and report potentially breaking schema drift."""
from pathlib import Path
import tempfile
from csv_inspector import profile_csv
from csv_inspector.schema import compare_csv_schemas


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        before = root / "before.csv"
        after = root / "after.csv"
        before.write_text("id,amount\n1,10\n2,20\n", encoding="utf-8")
        after.write_text("id,amount,currency\n1,10.5,INR\n2,20,INR\n",
                         encoding="utf-8")
        diff = compare_csv_schemas(profile_csv(before), profile_csv(after))
        print("Added:", diff.added_columns)
        print("Removed:", diff.removed_columns)
        print("Type changes:", diff.type_changes)
        print("Potentially breaking:", diff.has_breaking_changes)


if __name__ == "__main__":
    main()
