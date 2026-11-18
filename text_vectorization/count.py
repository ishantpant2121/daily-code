"""Dense, dependency-free bag-of-words vectorization for small text corpora."""

from collections import Counter
from numbers import Real
from typing import Iterable

from ._tokenizer import extract_ngrams, tokenize
from ._validation import (
    DEFAULT_TOKEN_PATTERN,
    compile_token_pattern,
    validate_bool,
    validate_df_threshold,
    validate_documents,
    validate_feature_matrix,
    validate_max_features,
    validate_ngram_range,
    validate_stop_words,
)


class CountVectorizer:
    """Convert text documents into deterministic dense token-count vectors.

    The implementation intentionally returns ordinary Python lists and is
    intended for learning and small corpora. It does not allocate sparse
    matrices, read files, or download stop-word lists.
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        token_pattern: str = DEFAULT_TOKEN_PATTERN,
        ngram_range: tuple[int, int] = (1, 1),
        stop_words: Iterable[str] | None = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_features: int | None = None,
        binary: bool = False,
    ) -> None:
        self.lowercase = validate_bool(lowercase, "lowercase")
        self.binary = validate_bool(binary, "binary")
        self.token_pattern = compile_token_pattern(token_pattern)
        self.ngram_range = validate_ngram_range(ngram_range)
        self.stop_words = validate_stop_words(stop_words, lowercase=self.lowercase)
        self.min_df = validate_df_threshold(min_df, "min_df")
        self.max_df = validate_df_threshold(max_df, "max_df")
        self.max_features = validate_max_features(max_features)
        self._fitted = False

    def _document_features(self, document: str) -> list[str]:
        tokens = tokenize(
            document,
            self.token_pattern,
            lowercase=self.lowercase,
            stop_words=self.stop_words,
        )
        return extract_ngrams(tokens, self.ngram_range)

    def fit(self, documents: Iterable[str]) -> "CountVectorizer":
        """Learn a vocabulary from document and term frequencies."""
        corpus = validate_documents(documents)
        if not corpus:
            raise ValueError("at least one training document is required")

        term_frequency: Counter[str] = Counter()
        document_frequency: Counter[str] = Counter()
        for document in corpus:
            features = self._document_features(document)
            term_frequency.update(features)
            document_frequency.update(set(features))

        minimum = (
            self.min_df
            if isinstance(self.min_df, int)
            else self.min_df * len(corpus)
        )
        maximum = (
            self.max_df
            if isinstance(self.max_df, int)
            else self.max_df * len(corpus)
        )
        if minimum > maximum:
            raise ValueError("min_df must not exceed max_df after corpus scaling")

        eligible = [
            feature
            for feature, frequency in document_frequency.items()
            if frequency >= minimum and frequency <= maximum
        ]
        if self.max_features is not None:
            eligible = sorted(
                eligible,
                key=lambda feature: (-term_frequency[feature], feature),
            )[: self.max_features]
        feature_names = sorted(eligible)
        if not feature_names:
            raise ValueError(
                "empty vocabulary; lower min_df, raise max_df, or check tokenization"
            )

        self.feature_names_out_ = feature_names
        self.vocabulary_ = {
            feature: index for index, feature in enumerate(feature_names)
        }
        self.document_frequency_ = {
            feature: document_frequency[feature] for feature in feature_names
        }
        self.term_frequency_ = {
            feature: term_frequency[feature] for feature in feature_names
        }
        self.n_documents_ = len(corpus)
        self.n_features_in_ = len(feature_names)
        self._fitted = True
        return self

    def transform(self, documents: Iterable[str]) -> list[list[int]]:
        """Transform documents using the vocabulary learned during fit."""
        self._check_fitted()
        corpus = validate_documents(documents)
        matrix: list[list[int]] = []
        for document in corpus:
            counts = Counter(self._document_features(document))
            row = [0] * self.n_features_in_
            for feature, count in counts.items():
                index = self.vocabulary_.get(feature)
                if index is not None:
                    row[index] = 1 if self.binary else count
            matrix.append(row)
        return matrix

    def fit_transform(self, documents: Iterable[str]) -> list[list[int]]:
        """Learn the vocabulary and return the corresponding count matrix."""
        corpus = validate_documents(documents)
        return self.fit(corpus).transform(corpus)

    def get_feature_names_out(self) -> list[str]:
        """Return feature names in the same order as matrix columns."""
        self._check_fitted()
        return list(self.feature_names_out_)

    def inverse_transform(self, matrix: Iterable[Iterable[Real]]) -> list[list[str]]:
        """Return the non-zero feature names for each dense matrix row."""
        self._check_fitted()
        rows = validate_feature_matrix(matrix, self.n_features_in_)
        return [
            [
                self.feature_names_out_[index]
                for index, value in enumerate(row)
                if value != 0.0
            ]
            for row in rows
        ]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("vectorizer is not fitted; call fit before transform")
