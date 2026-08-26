"""The last step of the lifecycle loop: turning a monitoring signal
into an explicit, explainable retrain-or-not decision.

Deliberately simple, rule-based logic -- the point of this module is
not a clever decision algorithm, it's that the *lifecycle has a
closing step at all*: drift gets detected, and detection actually
leads somewhere (a flagged decision with a stated reason), rather than
being logged and ignored.
"""


def decide_retrain(drift_check_result, recent_labeled_accuracy=None, min_accuracy=0.75):
    """`drift_check_result` is the dict returned by DriftMonitor.check().
    `recent_labeled_accuracy`, if provided, is the accuracy of the
    current production model measured against a small recently-labeled
    sample (a realistic proxy for "someone spot-checked N recent
    predictions") -- optional because in production it usually isn't
    available continuously, only when a labeling batch is run.
    """
    reasons = []

    if not drift_check_result.get("ready", False):
        return {
            "trigger_retrain": False,
            "reasons": ["monitoring window not yet full — no decision made"],
        }

    if drift_check_result.get("input_drift_detected"):
        reasons.append(
            f"input distribution drift detected "
            f"(KS p-value={drift_check_result['p_value']:.4g} < alpha)"
        )

    if drift_check_result.get("confidence_drift_detected"):
        reasons.append(
            f"mean prediction confidence dropped by "
            f"{drift_check_result['confidence_drop']:.3f} vs. baseline"
        )

    if recent_labeled_accuracy is not None and recent_labeled_accuracy < min_accuracy:
        reasons.append(
            f"recent labeled-sample accuracy {recent_labeled_accuracy:.2%} "
            f"below minimum {min_accuracy:.2%}"
        )

    return {"trigger_retrain": len(reasons) > 0, "reasons": reasons}
