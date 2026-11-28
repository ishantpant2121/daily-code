# CSV schema comparison

Compare two CSV profiles to identify likely schema drift without retaining
either complete source file in memory.

```python
from csv_inspector import profile_csv
from csv_inspector.schema import compare_csv_schemas

before = profile_csv("exports/yesterday.csv")
after = profile_csv("exports/today.csv")
diff = compare_csv_schemas(before, after)

print(diff.added_columns)
print(diff.removed_columns)
print(diff.type_changes)
print(diff.has_breaking_changes)
```

## Classification policy

- Added columns are reported and considered non-breaking by default.
- Removed columns and reordered common columns are breaking because consumers
  may depend on the original schema or positional layout.
- Integer-to-number is the only inferred type promotion considered compatible.
- Other inferred type changes are breaking.
- Duplicate header names are reported in `ambiguous_names`; the comparator
  does not guess which duplicate corresponds to an earlier column.

This is a practical drift signal, not a formal schema contract. Inferred types
depend on observed values, so profile representative data or maintain an
explicit schema for stronger guarantees. Null rates, ranges and business
meaning are intentionally outside this comparison.
