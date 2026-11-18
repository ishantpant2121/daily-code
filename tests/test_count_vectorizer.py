import unittest

from text_vectorization import CountVectorizer


class CountVectorizerTests(unittest.TestCase):
    def test_fit_transform_uses_stable_feature_order_and_counts(self):
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(
            ["The cat sat sat", "the dog sat", "cat dog"]
        )
        self.assertEqual(
            vectorizer.get_feature_names_out(),
            ["cat", "dog", "sat", "the"],
        )
        self.assertEqual(matrix, [[1, 0, 2, 1], [0, 1, 1, 1], [1, 1, 0, 0]])
        self.assertEqual(vectorizer.vocabulary_["sat"], 2)

    def test_transform_accepts_generators_and_ignores_unknown_terms(self):
        vectorizer = CountVectorizer().fit(["red fox", "blue fox"])
        matrix = vectorizer.transform(document for document in ["fox red", "green bird"])
        self.assertEqual(matrix, [[0, 1, 1], [0, 0, 0]])

    def test_lowercase_can_be_disabled(self):
        vectorizer = CountVectorizer(lowercase=False)
        matrix = vectorizer.fit_transform(["Cat cat"])
        self.assertEqual(vectorizer.get_feature_names_out(), ["Cat", "cat"])
        self.assertEqual(matrix, [[1, 1]])

    def test_binary_counts_only_record_presence(self):
        matrix = CountVectorizer(binary=True).fit_transform(["cat cat dog"])
        self.assertEqual(matrix, [[1, 1]])

    def test_word_ngrams_are_contiguous_and_ordered(self):
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(["red fox jumps"])
        self.assertEqual(
            vectorizer.get_feature_names_out(),
            ["fox", "fox jumps", "jumps", "red", "red fox"],
        )
        self.assertEqual(matrix, [[1, 1, 1, 1, 1]])

    def test_custom_pattern_uses_its_single_capturing_group(self):
        vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b([A-Z]{2,})\b",
        )
        matrix = vectorizer.fit_transform(["ID API api"])
        self.assertEqual(vectorizer.get_feature_names_out(), ["API", "ID"])
        self.assertEqual(matrix, [[1, 1]])

    def test_stop_words_are_removed_before_ngram_generation(self):
        vectorizer = CountVectorizer(
            stop_words=["the"],
            ngram_range=(1, 2),
        )
        matrix = vectorizer.fit_transform(["the red fox"])
        self.assertEqual(
            vectorizer.get_feature_names_out(),
            ["fox", "red", "red fox"],
        )
        self.assertEqual(matrix, [[1, 1, 1]])

    def test_document_frequency_thresholds_filter_vocabulary(self):
        vectorizer = CountVectorizer(min_df=2, max_df=1.0)
        vectorizer.fit(["apple banana apple", "apple banana", "apple cherry"])
        self.assertEqual(vectorizer.get_feature_names_out(), ["apple", "banana"])
        self.assertEqual(vectorizer.document_frequency_, {"apple": 3, "banana": 2})

        capped = CountVectorizer(max_df=0.66).fit(
            ["apple banana", "apple banana", "apple cherry"]
        )
        self.assertEqual(capped.get_feature_names_out(), ["banana", "cherry"])

    def test_max_features_uses_term_frequency_then_lexical_ties(self):
        vectorizer = CountVectorizer(max_features=2).fit(
            ["alpha alpha beta", "beta gamma", "gamma"]
        )
        self.assertEqual(vectorizer.get_feature_names_out(), ["alpha", "beta"])

    def test_inverse_transform_returns_nonzero_features_in_column_order(self):
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(["cat dog", "dog bird"])
        self.assertEqual(
            vectorizer.inverse_transform(matrix),
            [["cat", "dog"], ["bird", "dog"]],
        )

    def test_empty_documents_are_valid_during_transform(self):
        vectorizer = CountVectorizer().fit(["cat dog"])
        self.assertEqual(vectorizer.transform(["", "!!!"]), [[0, 0], [0, 0]])
        self.assertEqual(vectorizer.transform([]), [])


class CountVectorizerValidationTests(unittest.TestCase):
    def test_rejects_unfitted_and_empty_training_corpora(self):
        with self.assertRaises(RuntimeError):
            CountVectorizer().transform(["cat"])
        with self.assertRaises(ValueError):
            CountVectorizer().fit([])
        with self.assertRaises(ValueError):
            CountVectorizer().fit("cat dog")

    def test_failed_refit_invalidates_the_previous_vocabulary(self):
        vectorizer = CountVectorizer().fit(["cat dog"])
        with self.assertRaises(ValueError):
            vectorizer.fit(["a"])
        with self.assertRaises(RuntimeError):
            vectorizer.transform(["cat"])

    def test_rejects_non_string_documents(self):
        with self.assertRaises(ValueError):
            CountVectorizer().fit(["cat", 3])

    def test_rejects_invalid_configuration(self):
        invalid_options = (
            {"lowercase": 1},
            {"binary": "yes"},
            {"ngram_range": (0, 1)},
            {"ngram_range": (2, 1)},
            {"ngram_range": [1, 2]},
            {"token_pattern": "("},
            {"token_pattern": r"(a)(b)"},
            {"token_pattern": r".*"},
            {"stop_words": "english"},
            {"stop_words": [""]},
            {"min_df": -1},
            {"max_df": 1.1},
            {"max_features": 0},
        )
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                CountVectorizer(**options)

    def test_rejects_inconsistent_document_frequency_thresholds(self):
        with self.assertRaises(ValueError):
            CountVectorizer(min_df=0.8, max_df=0.2).fit(
                ["alpha beta", "alpha beta", "alpha beta"]
            )

    def test_rejects_empty_vocabulary_after_tokenization_or_filtering(self):
        with self.assertRaises(ValueError):
            CountVectorizer().fit(["a I"])
        with self.assertRaises(ValueError):
            CountVectorizer(stop_words=["cat"]).fit(["cat"])

    def test_inverse_transform_validates_matrix_shape_and_values(self):
        vectorizer = CountVectorizer().fit(["cat dog"])
        for matrix in ([[1]], [[-1, 0]], [[float("inf"), 0]], [["1", 0]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                vectorizer.inverse_transform(matrix)


if __name__ == "__main__":
    unittest.main()
