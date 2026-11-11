"""Demonstrate expanding-window validation with a leakage gap."""

from advanced_splitting import TimeSeriesSplit

observations = list(range(12))
splitter = TimeSeriesSplit(n_splits=3, test_size=2, gap=1)

for fold, (train_indices, test_indices) in enumerate(splitter.split(len(observations)), 1):
    print(f"Fold {fold}")
    print("  train:", [observations[index] for index in train_indices])
    print("  test: ", [observations[index] for index in test_indices])
