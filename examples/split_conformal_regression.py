"""Demonstrate split conformal regression with separate train/calibration/test sets."""

from decision_tree_regressor import DecisionTreeRegressor
from split_conformal import SplitConformalRegressor, empirical_coverage, mean_interval_width


def main() -> None:
    features = [[float(index), float(index % 3)] for index in range(20)]
    targets = [
        2.5 * index + 1.2 * (index % 3) + ((index % 4) - 1.5)
        for index in range(20)
    ]

    train_features, train_targets = features[:10], targets[:10]
    calibration_features, calibration_targets = features[10:15], targets[10:15]
    test_features, test_targets = features[15:], targets[15:]

    model = DecisionTreeRegressor(
        max_depth=3, min_samples_leaf=2, random_state=7
    ).fit(train_features, train_targets)

    calibration_predictions = model.predict(calibration_features)
    conformal = SplitConformalRegressor(alpha=0.2).fit(
        calibration_targets, calibration_predictions
    )
    test_predictions = model.predict(test_features)
    intervals = conformal.predict(test_predictions)

    print(f"Calibration count: {conformal.n_calibration_}")
    print(f"Conformal rank: {conformal.rank_}")
    print(f"Residual threshold: {conformal.quantile_:.3f}")
    print(f"Empirical test coverage: {empirical_coverage(test_targets, intervals):.1%}")
    print(f"Mean interval width: {mean_interval_width(intervals):.3f}")
    for prediction, interval in zip(test_predictions, intervals):
        print(f"Prediction {prediction:7.3f} -> [{interval[0]:7.3f}, {interval[1]:7.3f}]")


if __name__ == "__main__":
    main()
