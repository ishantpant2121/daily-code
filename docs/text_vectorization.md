# Text vectorization

The `text_vectorization` package provides dependency-free word-count and
TF-IDF vectorizers for learning exercises and small text collections. It uses
Python's standard library and returns dense lists of Python numbers; it does not
require NumPy, SciPy, or scikit-learn.

## CountVectorizer

`CountVectorizer` learns a vocabulary from a training corpus and converts each
document into a row of token counts:

~~~python
from text_vectorization import CountVectorizer

documents = [
    "The quick brown fox",
    "The quick blue hare",
    "A blue fox",
]

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(counts)
~~~

Feature names are sorted lexicographically, so every row uses the same stable
column order. The default token pattern is
`(?u)\\b\\w\\w+\\b`: it selects Unicode word tokens containing at least
two word characters. Text is lowercased by default. A custom regular expression
may contain zero or one capturing group; when present, the captured text is the
token.

### Vocabulary controls

- `ngram_range=(1, 1)` extracts unigrams. Use `(1, 2)` for unigrams and
  bigrams, or `(2, 3)` for bigrams and trigrams. N-grams are contiguous words.
- `stop_words` accepts an iterable of explicit tokens. Stop words are removed
  before n-grams are generated; no external stop-word list is downloaded.
- `min_df` removes terms occurring in fewer than the configured number or
  fraction of documents.
- `max_df` removes terms occurring in more than the configured number or
  fraction of documents.
- Integer document-frequency thresholds are absolute document counts. Floating
  thresholds are proportions in the inclusive range `[0, 1]`.
- `max_features` retains the most frequent terms after document-frequency
  filtering. Ties are resolved lexicographically for deterministic output.
- `binary=True` changes each non-zero document-term count to one.

The fitted estimator exposes `vocabulary_`, `document_frequency_`,
`term_frequency_`, `n_documents_`, and `get_feature_names_out()`. Call
`transform()` on later documents to reuse the training vocabulary; unseen terms
are ignored. `inverse_transform()` returns the non-zero feature names for each
matrix row.

## TfidfVectorizer

`TfidfVectorizer` shares the same tokenization and vocabulary options. By
default, it uses smoothed inverse document frequency:

[
\operatorname{idf}(t) = \log\left(\frac{1 + N}{1 + \operatorname{df}(t)}\right) + 1
]

where `N` is the number of training documents and `df(t)` is the number of
training documents containing term `t`. The default term frequency is the raw
count. Set `sublinear_tf=True` to use `1 + log(count)` for non-zero counts.
Set `use_idf=False` to use unit IDF weights, or `smooth_idf=False` to omit the
add-one smoothing in the formula.

Rows are L2-normalized by default. Choose `norm="l1"` for unit absolute sum or
`norm=None` to leave rows unnormalized.

~~~python
from text_vectorization import TfidfVectorizer

train_documents = [
    "graph search finds shortest paths",
    "tree search explores branches",
    "shortest paths use weighted graphs",
]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
train_matrix = vectorizer.fit_transform(train_documents)
new_matrix = vectorizer.transform(["weighted graph search"])
print(vectorizer.get_feature_names_out())
print(new_matrix)
~~~

Fit the vectorizer on training documents only, then call `transform()` for
validation, test, or production documents. Fitting on held-out text leaks
information about that text into the learned vocabulary and IDF weights.

## Scope and complexity

Both estimators return dense matrices with shape
`number_of_documents × vocabulary_size`. Their storage therefore grows with
the full product of those dimensions, even when most entries are zero. They are
intended for small educational corpora, not large-scale text retrieval or
production NLP. The package supports word n-grams only; it does not currently
provide character n-grams, sparse matrices, custom analyzers, stemming,
lemmatization, file loading, or a built-in English stop-word list.

The tokenization, document-frequency, and TF-IDF conventions are informed by
the public API documentation for
[scikit-learn CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
and
[scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
This repository contains an original standard-library implementation; it does
not copy scikit-learn implementation code or add it as a dependency.
