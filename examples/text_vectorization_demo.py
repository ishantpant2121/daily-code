"""Run a small bag-of-words and TF-IDF example from the repository root."""

from text_vectorization import CountVectorizer, TfidfVectorizer


def main() -> None:
    documents = [
        "Graph search finds shortest paths",
        "Tree search explores branches",
        "Shortest paths use weighted graphs",
    ]

    counts = CountVectorizer(ngram_range=(1, 2), stop_words=["use"])
    count_matrix = counts.fit_transform(documents)
    print("Count features:", counts.get_feature_names_out())
    print("Count matrix:", count_matrix)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words=["use"])
    tfidf_matrix = tfidf.fit_transform(documents)
    print("TF-IDF features:", tfidf.get_feature_names_out())
    print("TF-IDF matrix:", tfidf_matrix)


if __name__ == "__main__":
    main()
