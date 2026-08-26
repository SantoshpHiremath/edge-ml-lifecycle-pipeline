from src.retrain_trigger import decide_retrain


def test_no_decision_when_window_not_ready():
    result = decide_retrain({"ready": False})
    assert result["trigger_retrain"] is False
    assert "not yet full" in result["reasons"][0]


def test_no_trigger_when_nothing_drifted_and_accuracy_fine():
    drift_result = {
        "ready": True,
        "input_drift_detected": False,
        "confidence_drift_detected": False,
    }
    result = decide_retrain(drift_result, recent_labeled_accuracy=0.9)
    assert result["trigger_retrain"] is False
    assert result["reasons"] == []


def test_trigger_on_input_drift():
    drift_result = {
        "ready": True,
        "input_drift_detected": True,
        "p_value": 0.001,
        "confidence_drift_detected": False,
    }
    result = decide_retrain(drift_result)
    assert result["trigger_retrain"] is True
    assert any("input distribution drift" in r for r in result["reasons"])


def test_trigger_on_confidence_drift():
    drift_result = {
        "ready": True,
        "input_drift_detected": False,
        "confidence_drift_detected": True,
        "confidence_drop": 0.3,
    }
    result = decide_retrain(drift_result)
    assert result["trigger_retrain"] is True
    assert any("confidence" in r for r in result["reasons"])


def test_trigger_on_low_recent_accuracy_even_without_drift():
    drift_result = {
        "ready": True,
        "input_drift_detected": False,
        "confidence_drift_detected": False,
    }
    result = decide_retrain(drift_result, recent_labeled_accuracy=0.4, min_accuracy=0.75)
    assert result["trigger_retrain"] is True
    assert any("accuracy" in r for r in result["reasons"])


def test_multiple_reasons_can_combine():
    drift_result = {
        "ready": True,
        "input_drift_detected": True,
        "p_value": 0.002,
        "confidence_drift_detected": True,
        "confidence_drop": 0.2,
    }
    result = decide_retrain(drift_result, recent_labeled_accuracy=0.3, min_accuracy=0.75)
    assert result["trigger_retrain"] is True
    assert len(result["reasons"]) == 3
