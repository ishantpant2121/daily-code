"""Evaluate split-conformal intervals overall and by cohort."""

from interval_diagnostics import interval_diagnostics
from split_conformal import SplitConformalRegressor


calibration_targets = [1.0, 2.0, 2.8, 4.2, 5.1, 6.0, 7.2, 8.1, 9.0, 10.2]
calibration_predictions = [1.1, 1.8, 3.0, 4.0, 5.0, 6.2, 7.0, 8.0, 9.2, 10.0]
test_targets = [1.2, 2.4, 7.5, 8.8]
test_predictions = [1.0, 2.0, 7.0, 8.0]
cohorts = ["A", "A", "B", "B"]

conformal = SplitConformalRegressor(alpha=0.2).fit(
    calibration_targets, calibration_predictions
)
intervals = conformal.predict(test_predictions)
report = interval_diagnostics(
    test_targets, intervals, alpha=conformal.alpha, groups=cohorts
)

print(f"Overall empirical coverage: {report.overall.coverage:.1%}")
print(f"Mean interval width: {report.overall.mean_width:.3f}")
for cohort, summary in report.by_group.items():
    print(f"{cohort}: coverage={summary.coverage:.1%}, n={summary.count}")
