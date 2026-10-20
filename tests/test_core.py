import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from rexerr import metrics as M
from rexerr.data import build_report_task, build_sentence_task, normalize_text
from rexerr.labels import GROUP6, NATIVE, group6_matrix, normalize_type, parse_sampled


def test_normalize_type_variants():
    assert normalize_type("Add repetition") == "add_repetitions"
    assert normalize_type("Change Severity") == "change_severity"
    assert normalize_type("Not applicable") is None
    assert normalize_type(float("nan")) is None


def test_parse_sampled():
    assert parse_sampled("['Change severity', 'Add repetitions', 'Add medical device']") == [
        "change_severity", "add_repetitions", "add_medical_device"]


def test_fold_contradictions():
    from rexerr.io import fold_contradictions
    p = np.zeros((2, len(NATIVE)), dtype=np.float32)
    p[0, NATIVE.index("add_contradictions")] = 1
    p[1, NATIVE.index("add_typo")] = 1
    q = fold_contradictions(p)
    assert q[0, NATIVE.index("false_prediction")] == 1 and q[:, NATIVE.index("add_contradictions")].sum() == 0
    assert q[1].sum() == 1 and p[0, NATIVE.index("add_contradictions")] == 1  # input left unchanged


def test_group6_is_logical_or():
    y = np.zeros((1, len(NATIVE)))
    y[0, NATIVE.index("add_typo")] = 1
    y[0, NATIVE.index("change_to_homophone")] = 1
    g = group6_matrix(y)
    assert g[0, GROUP6.index("linguistic_error")] == 1 and g.sum() == 1


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.random(500) + 0.3 * y
    assert M._auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_summary_simple_case():
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.4], [0.1, 0.2]])
    s = M.summary(y, p, 0.5)
    assert s["f1_macro"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert s["exact_match_accuracy"] == pytest.approx(0.75)


def test_prevalence_analysis_ppv():
    row = M.prevalence_analysis(0.9, 0.9, prevalences=(0.05,))[0]
    assert row["ppv"] == pytest.approx(0.9 * 0.05 / (0.9 * 0.05 + 0.1 * 0.95))
    assert row["alerts_per_1000"] == pytest.approx(1000 * (0.9 * 0.05 + 0.1 * 0.95))


def test_threshold_for_sensitivity():
    y = np.array([1, 1, 1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.65, 0.1])
    t = M.threshold_for_sensitivity(y, s, 0.75)
    assert ((s >= t) & (y == 1)).sum() / y.sum() >= 0.75


def _write_fake_rexerr(root):
    (root / "ReXErr-report-level").mkdir()
    (root / "ReXErr-sentence-level").mkdir()
    reports = pd.DataFrame({
        "dicom_id": ["a", "b"], "study_id": [1, 2], "subject_id": [10, 20],
        "original_report": ["Findings:  No pneumothorax.  Heart size normal. ", "Impression: Clear lungs."],
        "error_report": ["Findings: No pneumothorax. Heart size enlarged.", "Impression: Clear lugns. A line is present."],
        "errors_sampled": ["['Change severity', 'Add typo', 'False prediction']",
                           "['Add typo', 'Add medical device', 'Change location']"],
    })
    sentences = pd.DataFrame({
        "dicom_id": ["a", "a", "b", "b"], "study_id": [1, 1, 2, 2], "subject_id": [10, 10, 20, 20],
        "original_sentence": ["Findings: No pneumothorax.", "Heart size normal.", "Impression: Clear lungs.", None],
        "error_sentence": ["Findings: No pneumothorax.", "Heart size enlarged.", "Impression: Clear lugns.",
                           "A line is present."],
        "error_present": [0, 1, 1, 1],
        "error_type": ["Not applicable", "Change severity", "Add typo", "Add medical device"],
    })
    for split in ("train", "val", "test"):
        reports.to_csv(root / "ReXErr-report-level" / f"ReXErr-report-level_{split}.csv", index=False)
        sentences.to_csv(root / "ReXErr-sentence-level" / f"ReXErr-sentence-level_{split}.csv", index=False)


def test_report_task_with_controls(tmp_path):
    _write_fake_rexerr(tmp_path)
    frame, y = build_report_task(str(tmp_path), "test")
    assert len(frame) == 4 and set(frame["variant"]) == {"error", "original"}
    assert y[frame["variant"] == "original"].sum() == 0
    err1 = y[(frame["study_id"] == 1).to_numpy() & (frame["variant"] == "error").to_numpy()][0]
    assert err1[NATIVE.index("change_severity")] == 1 and err1.sum() == 1  # sentence labels, not sampled ones
    assert "  " not in " ".join(frame["text"])  # whitespace normalized
    frame, _ = build_report_task(str(tmp_path), "test", include_clean=False)
    assert len(frame) == 2
    _, y_s = build_report_task(str(tmp_path), "test", label_source="sampled", include_clean=False)
    assert y_s.sum() == 6


def test_sentence_task(tmp_path):
    _write_fake_rexerr(tmp_path)
    frame, y = build_sentence_task(str(tmp_path), "test")
    assert (frame["variant"] == "error").sum() == 4 and (frame["variant"] == "original").sum() == 3
    assert y.sum() == 3


def test_normalize_text():
    assert normalize_text("  a \n b  ") == "a b"


def test_llm_parse_ignores_reasoning():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from llm_zeroshot import parse
    assert parse('<think>laterality?</think>\n{"errors": ["Change location", "Add typo"]}') == (
        {"change_location", "add_typo"}, True)
    assert parse('{"errors": []}') == (set(), True)
    assert parse("<think> unfinished reasoning") == (set(), False)


def test_perturb_rules_change_text_once():
    from rexerr.perturb import RULES, apply, build
    text = ("Findings: There is a small left pleural effusion measuring 2.5 cm. No pneumothorax is seen. "
            "The right lung base is clear. Impression: Mild cardiomegaly.")
    for name in RULES:
        new = apply(text, name, 0)
        assert new is not None and new != text, name
    assert apply(text, "false_negation", 0) == text.replace("There is a small", "There is no small")
    assert apply("No change.", "change_measurement", 0) is None
    kinds = [k for _, k in build([text] * 90, seed=0)]
    assert max(kinds.count(k) for k in RULES) - min(kinds.count(k) for k in RULES) <= 1
