"""TF-IDF text weighting built on the dependency-free CountVectorizer."""

import math
from typing import Any, Iterable

from ._validation import validate_bool, validate_documents
from .count import CountVectorizer


class TfidfVectorizer(CountVectorizer):
    """Convert documents to dense term-frequency/inverse-document-frequency rows.

    The default IDF is log((1 + n_documents) / (1 + document_frequency)) + 1.
    Rows are L2-normalized by default. Results are ordinary Python lists and
    are intended for educational use and small corpora.
    """

    def __init__(
        self,
        *,
        norm: str | None = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        **vectorizer_options: Any,
    ) -> None:
        super().__init__(**vectorizer_options)
        if norm not in ("l1", "l2", None):
            raise ValueError("norm must be 'l1', 'l2', or None")
        self.norm = norm
        self.use_idf = validate_bool(use_idf, "use_idf")
        self.smooth_idf = validate_bool(smooth_idf, "smooth_idf")
        self.sublinear_tf = validate_bool(sublinear_tf, "sublinear_tf")

    def fit(self, documents: Iterable[str]) -> "TfidfVectorizer":
        """Learn the vocabulary and inverse-document-frequency weights."""
        corpus = validate_documents(documents)
        super().fit(corpus)
        if self.use_idf:
            if self.smooth_idf:
                self.idf_ = [
                    math.log((1 + self.n_documents_) / (1 + self.document_frequency_[feature])) + 1.0
                    for feature in self.feature_names_out_
                ]
            else:
                self.idf_ = [
                    math.log(self.n_documents_ / self.document_frequency_[feature]) + 1.0
                    for feature in self.feature_names_out_
                ]
        else:
            self.idf_ = [1.0] * self.n_features_in_
        return self

    def transform(self, documents: Iterable[str]) -> list[list[float]]:
        """Transform documents with fitted TF-IDF weights and row normalization."""
        self._check_fitted()
        counts = super().transform(documents)
        weighted_rows: list[list[float]] = []
        for row in counts:
            weighted = []
            for count, inverse_frequency in zip(row, self.idf_):
                if count == 0:
                    weighted.append(0.0)
                    continue
                term_frequency = 1.0 + math.log(count) if self.sublinear_tf else float(count)
                weighted.append(term_frequency * inverse_frequency)

            if self.norm == "l1":
                denominator = math.fsum(abs(value) for value in weighted)
            elif self.norm == "l2":
                denominator = math.sqrt(math.fsum(value * value for value in weighted))
            else:
                denominator = 0.0

            if self.norm is not None and denominator > 0.0:
                weighted = [value / denominator for value in weighted]
            weighted_rows.append(weighted)
        return weighted_rows

    def fit_transform(self, documents: Iterable[str]) -> list[list[float]]:
        """Learn vocabulary and IDF weights, then return normalized TF-IDF rows."""
        corpus = validate_documents(documents)
        return self.fit(corpus).transform(corpus)
