"""Error taxonomy for ReXErr-v1.

The 12 native ReXErr error types are the primary label space. The 6-group
grouping merges related types into six clinically motivated groups and is
reported as a secondary view.
"""
import ast
import re

# Native ReXErr-v1 error types (README order).
NATIVE = [
    "add_medical_device",
    "false_prediction",
    "false_negation",
    "add_repetitions",
    "add_contradictions",
    "change_name_of_device",
    "change_position_of_device",
    "change_severity",
    "change_location",
    "change_measurement",
    "change_to_homophone",
    "add_typo",
]

NATIVE_DISPLAY = {
    "add_medical_device": "Add medical device",
    "false_prediction": "False prediction",
    "false_negation": "False negation",
    "add_repetitions": "Add repetitions",
    "add_contradictions": "Add contradictions",
    "change_name_of_device": "Change name of device",
    "change_position_of_device": "Change position of device",
    "change_severity": "Change severity",
    "change_location": "Change location",
    "change_measurement": "Change measurement",
    "change_to_homophone": "Change to homophone",
    "add_typo": "Add typo",
}

# ReXErr's own three-way grouping (README).
REXERR_GROUP = {
    "add_medical_device": "content_addition",
    "false_prediction": "content_addition",
    "false_negation": "content_addition",
    "add_repetitions": "linguistic_quality",
    "add_contradictions": "linguistic_quality",
    "change_to_homophone": "linguistic_quality",
    "add_typo": "linguistic_quality",
    "change_name_of_device": "context_dependent",
    "change_position_of_device": "context_dependent",
    "change_severity": "context_dependent",
    "change_location": "context_dependent",
    "change_measurement": "context_dependent",
}

# Six-group mapping. The linguistic group holds typographical/transcription
# errors (typos, homophones, repetitions); contradictions in ReXErr are not
# restricted to findings versus impression.
GROUP6 = [
    "false_findings",
    "incorrect_location",
    "measurement_error",
    "internal_contradiction",
    "linguistic_error",
    "severity_error",
]

GROUP6_DISPLAY = {
    "false_findings": "False findings / devices",
    "incorrect_location": "Incorrect location / position",
    "measurement_error": "Measurement error",
    "internal_contradiction": "Internal contradiction",
    "linguistic_error": "Linguistic / transcription error",
    "severity_error": "Severity misclassification",
}

NATIVE_TO_GROUP6 = {
    "false_prediction": "false_findings",
    "false_negation": "false_findings",
    "add_medical_device": "false_findings",
    "change_name_of_device": "false_findings",
    "change_location": "incorrect_location",
    "change_position_of_device": "incorrect_location",
    "change_measurement": "measurement_error",
    "add_contradictions": "internal_contradiction",
    "add_repetitions": "linguistic_error",
    "change_to_homophone": "linguistic_error",
    "add_typo": "linguistic_error",
    "change_severity": "severity_error",
}


def normalize_type(name):
    """Map any spelling used in ReXErr files to a NATIVE key (or None)."""
    if name is None:
        return None
    s = str(name).strip().lower()
    if not s or s in {"nan", "not applicable", "none", "n/a"}:
        return None
    s = re.sub(r"[^a-z ]+", " ", s)
    s = re.sub(r"\s+", "_", s.strip())
    # Sentence-level files use singular forms ("Add repetition").
    s = {"add_repetition": "add_repetitions", "add_contradiction": "add_contradictions"}.get(s, s)
    return s if s in NATIVE else None


def parse_sampled(value):
    """Parse the `errors_sampled` column (a stringified Python list)."""
    if value is None or (isinstance(value, float) and value != value):
        return []
    try:
        items = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        items = re.split(r"[,;]", str(value).strip("[]"))
    if isinstance(items, str):
        items = [items]
    return [t for t in (normalize_type(i) for i in items) if t]


def group6_matrix(y_native):
    """Collapse an (n, 12) native matrix into the (n, 6) group matrix (logical OR / max)."""
    import numpy as np

    y_native = np.asarray(y_native)
    out = np.zeros((y_native.shape[0], len(GROUP6)), dtype=y_native.dtype)
    for j, name in enumerate(NATIVE):
        g = GROUP6.index(NATIVE_TO_GROUP6[name])
        out[:, g] = np.maximum(out[:, g], y_native[:, j])
    return out
