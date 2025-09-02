from crom_efficientllm.drift_estimator.estimator import DriftEstimator, DriftMode

def test_drift_triggers():
    de = DriftEstimator(threshold=0.1, mode=DriftMode.L2)
    alert, dist, ewma = de.update([0, 0, 0])
    assert alert is False
    alert, dist, ewma = de.update([1, 0, 0])
    assert isinstance(alert, bool)
