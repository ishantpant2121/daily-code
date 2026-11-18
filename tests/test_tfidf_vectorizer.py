import math
import unittest

from text_vectorization import TfidfVectorizer


class TfidfVectorizerTests(unittest.TestCase):
    def test_smooth_idf_matches_document_frequency_formula(self):
        vectorizer = TfidfVectorizer(norm=None).fit(
            ["cat dog", "cat cat", "dog bird"]
        )
        self.assertEqual(vectorizer.get_feature_names_out(), ["bird", "cat", "dog"])
        self.assertAlmostEqual(vectorizer.idf_[0], math.log(4 / 2) + 1)
        self.assertAlmostEqual(vectorizer.idf_[1], math.log(4 / 3) + 1)
        self.assertAlmostEqual(vectorizer.idf_[2], math.log(4 / 3) + 1)

    def test_l2_normalization_produces_unit_length_nonzero_rows(self):
        matrix = TfidfVectorizer().fit_transform(
            ["cat dog", "cat cat", "dog bird", ""]
        )
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in matrix[0])), 1.0)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in matrix[1])), 1.0)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in matrix[2])), 1.0)
        self.assertEqual(matrix[3], [0.0, 0.0, 0.0])

    def test_l1_normalization_and_no_normalization(self):
        l1 = TfidfVectorizer(norm="l1").fit_transform(["cat dog"])
        self.assertAlmostEqual(sum(abs(value) for value in l1[0]), 1.0)

        raw = TfidfVectorizer(norm=None, use_idf=False).fit_transform(["cat cat dog"])
        self.assertEqual(raw, [[2.0, 1.0]])

    def test_unsmoothed_idf_uses_raw_document_frequencies(self):
        vectorizer = TfidfVectorizer(norm=None, smooth_idf=False).fit(
            ["cat dog", "cat cat", "dog bird"]
        )
        self.assertAlmostEqual(vectorizer.idf_[0], math.log(3 / 1) + 1)
        self.assertAlmostEqual(vectorizer.idf_[1], math.log(3 / 2) + 1)

    def test_sublinear_term_frequency_uses_one_plus_log_count(self):
        vectorizer = TfidfVectorizer(
            norm=None,
            use_idf=False,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(["cat cat dog"])
        self.assertAlmostEqual(matrix[0][0], 1.0 + math.log(2.0))
        self.assertEqual(matrix[0][1], 1.0)

    def test_binary_term_frequency_is_available(self):
        matrix = TfidfVectorizer(
            norm=None,
            use_idf=False,
            binary=True,
        ).fit_transform(["cat cat dog"])
        self.assertEqual(matrix, [[1.0, 1.0]])

    def test_fit_transform_and_transform_accept_generators(self):
        vectorizer = TfidfVectorizer().fit(
            document for document in ["red fox", "blue fox"]
        )
        matrix = vectorizer.transform(document for document in ["fox red", "green bird"])
        self.assertEqual(len(matrix), 2)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in matrix[0])), 1.0)
        self.assertEqual(matrix[1], [0.0, 0.0, 0.0])

    def test_inverse_transform_works_with_tfidf_rows(self):
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(["cat dog", "dog bird"])
        self.assertEqual(
            vectorizer.inverse_transform(matrix),
            [["cat", "dog"], ["bird", "dog"]],
        )


class TfidfVectorizerValidationTests(unittest.TestCase):
    def test_rejects_invalid_normalization_and_boolean_options(self):
        with self.assertRaises(ValueError):
            TfidfVectorizer(norm="max")
        for options in (
            {"use_idf": 1},
            {"smooth_idf": "yes"},
            {"sublinear_tf": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TfidfVectorizer(**options)

    def test_transform_requires_a_fitted_vectorizer(self):
        with self.assertRaises(RuntimeError):
            TfidfVectorizer().transform(["cat"])

    def test_empty_training_corpus_is_rejected(self):
        with self.assertRaises(ValueError):
            TfidfVectorizer().fit([])


if __name__ == "__main__":
    unittest.main()
