"""Compare inferred schemas from two CSV profiles."""
from collections import Counter
from dataclasses import dataclass
from .models import CSVProfile


@dataclass(frozen=True, slots=True)
class ColumnTypeChange:
    """An inferred type change for a uniquely named column."""
    name: str
    previous: str
    current: str
    compatible: bool


@dataclass(frozen=True, slots=True)
class SchemaDiff:
    """Immutable summary of schema drift between two CSV profiles."""
    added_columns: tuple[str, ...]
    removed_columns: tuple[str, ...]
    type_changes: tuple[ColumnTypeChange, ...]
    reordered: bool
    ambiguous_names: tuple[str, ...]

    @property
    def has_changes(self) -> bool:
        return bool(self.added_columns or self.removed_columns or self.type_changes
                    or self.reordered or self.ambiguous_names)

    @property
    def has_breaking_changes(self) -> bool:
        return bool(self.removed_columns or self.reordered or self.ambiguous_names
                    or any(not item.compatible for item in self.type_changes))


_COMPATIBLE_PROMOTIONS = {("integer", "number")}


def compare_csv_schemas(previous: CSVProfile, current: CSVProfile) -> SchemaDiff:
    """Compare profiles by header name and inferred type.

    Adding columns is non-breaking. Removal, reordering, duplicate names and
    incompatible type changes are breaking. Integer-to-number is compatible.
    Duplicate names are reported as ambiguous instead of guessed by position.
    """
    if not isinstance(previous, CSVProfile) or not isinstance(current, CSVProfile):
        raise TypeError("previous and current must be CSVProfile instances")

    old_counts = Counter(column.name for column in previous.columns)
    new_counts = Counter(column.name for column in current.columns)
    ambiguous = tuple(sorted(
        name for name in old_counts.keys() | new_counts.keys()
        if old_counts[name] > 1 or new_counts[name] > 1
    ))
    old_unique = {c.name: c for c in previous.columns
                  if old_counts[c.name] == 1 and new_counts[c.name] == 1}
    new_unique = {c.name: c for c in current.columns
                  if old_counts[c.name] == 1 and new_counts[c.name] == 1}

    old_names, new_names = set(old_counts), set(new_counts)
    added = tuple(c.name for c in current.columns if c.name not in old_names)
    removed = tuple(c.name for c in previous.columns if c.name not in new_names)

    changes = []
    for column in current.columns:
        name = column.name
        if name in old_unique and name in new_unique:
            before, after = old_unique[name].inferred_type, new_unique[name].inferred_type
            if before != after:
                changes.append(ColumnTypeChange(
                    name, before, after, (before, after) in _COMPATIBLE_PROMOTIONS
                ))

    common = old_names & new_names
    old_order = [c.name for c in previous.columns if c.name in common]
    new_order = [c.name for c in current.columns if c.name in common]
    reordered = not ambiguous and old_order != new_order
    return SchemaDiff(added, removed, tuple(changes), reordered, ambiguous)
