"""
Tests for app.evaluator.calculate_cross_model_agreement.

Writes small synthetic prediction CSVs under a dedicated test prompt_version
("test_cma") so nothing here ever touches real export data or the real
human-annotated dataset — cleaned up after every test.

Usage:
    python -m tools.test_cross_model_agreement
"""
from pathlib import Path

import pandas as pd

from app.evaluator import calculate_cross_model_agreement

OUTPUT_DIR = Path("outputs")
TEST_PROMPT_VERSION = "test_cma"


def _write_predictions(model, ids, action, consequence):
    path = OUTPUT_DIR / f"output_{model.replace('/', '_')}_{TEST_PROMPT_VERSION}_entire_dataset.csv"
    df = pd.DataFrame({
        "ID": ids,
        "Scenario": [f"scenario {i}" for i in ids],
        f"{model}_action": action,
        f"{model}_consequences": consequence,
        "action_reasoning": [""] * len(ids),
        "consequences_reasoning": [""] * len(ids),
    })
    df.to_csv(path, index=False)
    return path


def _cleanup(paths):
    for p in paths:
        if p.exists():
            p.unlink()


def test_perfect_agreement():
    ids = [1, 2, 3, 4, 5]
    vals = [-0.8, -0.2, 0.1, 0.5, 0.9]
    paths = [
        _write_predictions("gemini", ids, vals, vals),
        _write_predictions("claude", ids, vals, vals),
    ]
    try:
        result = calculate_cross_model_agreement(["gemini", "claude"], prompt_version=TEST_PROMPT_VERSION)
        assert result["n_scenarios"] == 5
        assert round(result["action"]["ccc_matrix"]["gemini"]["claude"], 4) == 1.0
        assert round(result["action"]["spearman_matrix"]["gemini"]["claude"], 4) == 1.0
        assert round(result["consequence"]["ccc_matrix"]["gemini"]["claude"], 4) == 1.0
        assert result["action"]["ccc_matrix"]["gemini"]["gemini"] == 1.0
        print("✅ Perfect Agreement Test Passed")
    finally:
        _cleanup(paths)


def test_perfect_inverse():
    """
    b_vals = -a_vals gives Pearson exactly -1.0 (pure linear inverse
    relationship). CCC is expected to be strongly negative but NOT exactly
    -1.0 here — unlike Pearson, CCC also penalises the two series having
    different means (mean(a)=0.1, mean(b)=-0.1), which is the whole point
    of using CCC instead of Pearson for this analysis (see
    calculate_cross_model_agreement's docstring). Independently
    hand-verified: CCC = -0.9551 for these exact values.
    """
    ids = [1, 2, 3, 4, 5]
    a_vals = [-0.8, -0.2, 0.1, 0.5, 0.9]
    b_vals = [0.8, 0.2, -0.1, -0.5, -0.9]  # exactly -a_vals
    paths = [
        _write_predictions("gemini", ids, a_vals, a_vals),
        _write_predictions("claude", ids, b_vals, b_vals),
    ]
    try:
        result = calculate_cross_model_agreement(["gemini", "claude"], prompt_version=TEST_PROMPT_VERSION)
        assert abs(result["action"]["ccc_matrix"]["gemini"]["claude"] - (-0.9550561797752808)) < 1e-9
        assert result["action"]["spearman_matrix"]["gemini"]["claude"] < -0.98
        print("✅ Perfect Inverse Test Passed")
    finally:
        _cleanup(paths)


def test_known_small_dataset():
    """
    Cross-checked against an independent from-scratch implementation of
    Lin's CCC formula (numpy/scipy.stats.pearsonr only — no shared code
    with app.metric.calculate_ccc), plus scipy's own spearmanr, both
    computed outside app code entirely.
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    def _manual_ccc(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        rho, _ = pearsonr(x, y)
        mx, my = x.mean(), y.mean()
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        return (2 * rho * np.sqrt(vx) * np.sqrt(vy)) / (vx + vy + (mx - my) ** 2)

    ids = [1, 2, 3, 4]
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 4.0, 5.0, 8.0]
    paths = [
        _write_predictions("gemini", ids, a, a),
        _write_predictions("claude", ids, b, b),
    ]
    try:
        result = calculate_cross_model_agreement(["gemini", "claude"], prompt_version=TEST_PROMPT_VERSION)
        expected_ccc = _manual_ccc(a, b)
        expected_spearman, _ = spearmanr(a, b)
        assert abs(result["action"]["ccc_matrix"]["gemini"]["claude"] - expected_ccc) < 1e-9
        assert abs(result["action"]["spearman_matrix"]["gemini"]["claude"] - expected_spearman) < 1e-9
        print(f"✅ Known Small Dataset Test Passed (ccc={expected_ccc:.4f}, spearman={expected_spearman:.4f})")
    finally:
        _cleanup(paths)


def test_row_order_robustness():
    ids = [1, 2, 3, 4, 5]
    a = [-0.8, -0.2, 0.1, 0.5, 0.9]
    b = [0.3, -0.1, 0.6, 0.2, 0.9]
    paths = [
        _write_predictions("gemini", ids, a, a),
        _write_predictions("claude", ids, b, b),
    ]
    try:
        result1 = calculate_cross_model_agreement(["gemini", "claude"], prompt_version=TEST_PROMPT_VERSION)

        claude_path = paths[1]
        shuffled = pd.read_csv(claude_path).sample(frac=1, random_state=123).reset_index(drop=True)
        shuffled.to_csv(claude_path, index=False)

        result2 = calculate_cross_model_agreement(["gemini", "claude"], prompt_version=TEST_PROMPT_VERSION)

        assert result1["action"]["ccc_matrix"]["gemini"]["claude"] == result2["action"]["ccc_matrix"]["gemini"]["claude"]
        assert result1["action"]["spearman_matrix"]["gemini"]["claude"] == result2["action"]["spearman_matrix"]["gemini"]["claude"]
        assert result1["n_scenarios"] == result2["n_scenarios"] == 5
        print("✅ Row-Order Robustness Test Passed (results unchanged after shuffling rows)")
    finally:
        _cleanup(paths)


def test_missing_scenario():
    a_ids = [1, 2, 3, 4, 5]
    a = [-0.8, -0.2, 0.1, 0.5, 0.9]
    b_ids = [1, 2, 3, 4]  # scenario 5 missing from claude
    b = [0.3, -0.1, 0.6, 0.2]
    paths = [
        _write_predictions("gemini", a_ids, a, a),
        _write_predictions("claude", b_ids, b, b),
    ]
    try:
        result = calculate_cross_model_agreement(["gemini", "claude"], prompt_version=TEST_PROMPT_VERSION)
        assert result["n_scenarios"] == 4
        print("✅ Missing Scenario Test Passed (N correctly reduced to 4, not padded with zeros)")
    finally:
        _cleanup(paths)


def test_duplicate_scenario_id():
    ids = [1, 2, 2, 4]  # duplicate ID 2
    a = [-0.8, -0.2, 0.15, 0.5]
    paths = [
        _write_predictions("gemini", ids, a, a),
        _write_predictions("claude", [1, 2, 3, 4], a, a),
    ]
    try:
        try:
            calculate_cross_model_agreement(["gemini", "claude"], prompt_version=TEST_PROMPT_VERSION)
            raised = False
        except ValueError as e:
            raised = True
            assert "duplicate" in str(e).lower()
        assert raised, "Expected ValueError for duplicate scenario ID, none was raised"
        print("✅ Duplicate Scenario ID Test Passed (clear ValueError raised)")
    finally:
        _cleanup(paths)


def test_human_annotation_independence():
    """
    The critical test: proves calculate_cross_model_agreement() never reads
    human annotations at all. Rather than mutating the real annotated
    dataset on disk (risky — that file is hand-annotated, irreplaceable),
    monkeypatch app.evaluator.load_dataset to raise if called, and confirm
    the function still succeeds — a stronger, structural guarantee than
    just checking the numeric result doesn't change.
    """
    import app.evaluator as evaluator_module

    ids = [1, 2, 3, 4, 5]
    a = [-0.8, -0.2, 0.1, 0.5, 0.9]
    b = [0.3, -0.1, 0.6, 0.2, 0.9]
    paths = [
        _write_predictions("gemini", ids, a, a),
        _write_predictions("claude", ids, b, b),
    ]

    original_load_dataset = evaluator_module.load_dataset

    def _poisoned_load_dataset(*args, **kwargs):
        raise AssertionError(
            "calculate_cross_model_agreement called load_dataset() — it must "
            "NEVER read human annotations, only model-vs-model predictions."
        )

    evaluator_module.load_dataset = _poisoned_load_dataset
    try:
        result = calculate_cross_model_agreement(["gemini", "claude"], prompt_version=TEST_PROMPT_VERSION)
        assert result["n_scenarios"] == 5
        print("✅ Human Annotation Independence Test Passed (load_dataset() was never called)")
    finally:
        evaluator_module.load_dataset = original_load_dataset
        _cleanup(paths)


if __name__ == "__main__":
    test_perfect_agreement()
    test_perfect_inverse()
    test_known_small_dataset()
    test_row_order_robustness()
    test_missing_scenario()
    test_duplicate_scenario_id()
    test_human_annotation_independence()
    print("\n🎉 All Cross-Model Agreement tests passed!")
