"""A dependency-free CSV profiling utility."""

from .models import CSVProfile, ColumnProfile, RowIssue
from .profiler import CSVInspectionError, profile_csv

__all__ = ["CSVInspectionError", "CSVProfile", "ColumnProfile", "RowIssue", "profile_csv"]
__version__ = "0.1.0"
