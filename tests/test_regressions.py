import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import f1_score

from api import main as api
import predict
from src.data_loader import TripData, load_trip_from_dir
from src.hmm_viterbi import (
    HMMConfig, _precompute_node_reachability,
    compute_route_continuity, compute_sequence_route_metrics,
)
from src.paths import safe_child, validate_trip_id
from src.ui_state import request_evaluation, select_evaluation
from src.visualization import generate_folium_map_for_trip


@pytest.mark.parametrize("trip_id", ["../trip", "..", "/tmp/trip", r"C:\data\trip", r"..\trip", "a/b", "a\nb", "", "a" * 65])
def test_unsafe_trip_ids_rejected_before_loading(trip_id, tmp_path):
    with pytest.raises(ValueError):
        validate_trip_id(trip_id)
    with pytest.raises(ValueError):
        load_trip_from_dir(str(tmp_path), trip_id)
    response = TestClient(api.app).post("/evaluate", json={"trip_id": trip_id})
    assert response.status_code == 422


def test_output_paths_reject_escape_and_symlink(tmp_path):
    with pytest.raises(ValueError):
        safe_child(str(tmp_path), "../reference.route")
    target = tmp_path.parent / "reference.route"
    target.write_text("42\n")
    try:
        (tmp_path / "trip.route").symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer mode or elevated permissions.")
    with pytest.raises(ValueError):
        safe_child(str(tmp_path), "trip.route")
    assert target.read_text() == "42\n"


class ConstantModel:
    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (len(X), 1))


@pytest.fixture
def small_trip():
    return TripData(
        trip_id="demo", partition="student_test",
        track=np.array([[0.00001, 0., 0.], [0.00009, 0., 1.]]),
        nodes=np.array([[0., 0.], [0.0001, 0.]]),
        arcs=np.array([[0, 1], [1, 0]]), route=np.array([0]),
    )


def test_api_output_cannot_overwrite_reference_and_does_not_expose_paths(tmp_path, monkeypatch, small_trip):
    student = tmp_path / "student"
    trip_dir = student / "demo"
    trip_dir.mkdir(parents=True)
    for suffix, array in [("track", small_trip.track), ("nodes", small_trip.nodes),
                          ("arcs", small_trip.arcs), ("route", small_trip.route)]:
        np.savetxt(trip_dir / f"demo.{suffix}", array, fmt="%d" if suffix in {"arcs", "route"} else "%.12f")
    reference = trip_dir / "demo.route"
    before = reference.read_bytes()
    monkeypatch.setattr(api, "STUDENT_TEST_DIR", str(student))
    monkeypatch.setattr(api, "EVALUATOR_TEST_DIR", str(tmp_path / "missing_eval"))
    monkeypatch.setattr(api, "DATA_ROOT", str(tmp_path / "Data" / "Student folder"))
    monkeypatch.setattr(api, "TRAIN_DIR", str(student))
    monkeypatch.setattr(api, "_get_artifacts", lambda: (ConstantModel(), HMMConfig(), {}))
    monkeypatch.setattr(predict, "PREDICTIONS_OUT_DIR", str(tmp_path / "predictions"))
    client = TestClient(api.app)
    # Reproduce the former absolute-stem overwrite attack.
    bad = client.post("/evaluate", json={"trip_id": str(trip_dir / "demo")})
    assert bad.status_code == 422
    response = client.post("/evaluate", json={"trip_id": "demo", "partition": "student_test"})
    assert response.status_code == 200
    result = response.json()
    assert reference.read_bytes() == before
    assert (tmp_path / "predictions" / "demo.route").exists()
    assert "Reference Ground-Truth" not in result["folium_map_html"]
    assert "output_route_file" not in result["metrics"]
    assert "map_html_path" not in result["metrics"]
    assert result["metrics"]["evaluation_protocol"] == "wta-lcs-v2"
    # A partition mismatch must not silently fall back to training data.
    assert client.post("/evaluate", json={"trip_id": "demo", "partition": "evaluator_test"}).status_code == 404
    assert client.post("/evaluate", json={"trip_id": "demo", "partition": "train", "include_map_html": False}).status_code == 200


def test_reference_layer_can_be_disabled_without_modifying_trip(small_trip):
    hidden = generate_folium_map_for_trip(small_trip, include_reference=False).get_root().render()
    visible = generate_folium_map_for_trip(small_trip).get_root().render()
    assert "Reference Ground-Truth" not in hidden
    assert "Reference Ground-Truth" in visible
    assert small_trip.route.tolist() == [0]


@pytest.mark.parametrize("pred,ref,expected", [
    ([1, 2, 3], [1, 2, 3], 1.),
    ([3, 2, 1], [1, 2, 3], 1 / 3),
    ([1, 2, 1], [1, 2], 4 / 5),
    ([1, 1], [1], 2 / 3),
    ([], [1], 0.), ([1], [], 0.),
])
def test_sequence_metric_respects_order_and_repetitions(pred, ref, expected):
    result = compute_sequence_route_metrics(pred, ref)
    assert result["route_seq_f1"] == pytest.approx(expected)
    if set(pred) == set(ref) and pred:
        assert result["route_set_f1"] == 1.


@pytest.mark.parametrize("max_hops,expected", [(4, 0.), (5, 1.), (6, 1.)])
def test_continuity_uses_same_limit_with_or_without_cached_reachability(max_hops, expected):
    trip = SimpleNamespace(arcs=np.array([[i, i + 1] for i in range(7)]), arc_lengths=np.ones(7))
    sequence = np.array([0, 6])
    reach = _precompute_node_reachability(trip, np.array([1]), max_hops=max_hops)
    direct = compute_route_continuity(trip, sequence, max_hops=max_hops)
    cached = compute_route_continuity(trip, sequence, reach_map=reach, max_hops=max_hops)
    assert direct == cached
    assert direct["connected_continuity"] == expected


def test_evaluation_uses_fixed_winner_takes_all_even_when_threshold_scores_better(monkeypatch):
    from src import models
    monkeypatch.setattr(models, "compute_route_continuity", lambda *args: {"connected_continuity": 1., "strict_1hop_continuity": 1.})
    trip = SimpleNamespace(trip_id="demo", obs_ref_arcs=np.array([-1]))
    candidates = SimpleNamespace(cand_arcs=np.array([[0, 1]]))
    result = models.evaluate_predictions_on_trips([trip], {"demo": candidates}, np.array([0.1, 0.1]), np.array([0, 0]))
    assert result["macro_f1"] == f1_score([0, 0], [1, 0], average="macro", zero_division=0)
    assert result["macro_f1"] < 1.


def test_tuning_records_measured_scores_without_baseline_offset(monkeypatch):
    from src import tuning
    class Model(ConstantModel):
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            return self
    monkeypatch.setattr(tuning, "LGBMClassifier", Model)
    monkeypatch.setattr(tuning, "N_CV_FOLDS", 2)
    metrics = {key: 0.5 for key in ["top1_accuracy", "macro_f1", "log_loss", "route_continuity"]}
    monkeypatch.setattr(tuning, "evaluate_predictions_on_trips", lambda *args: metrics)
    untuned = {key: 0.12 for key in ["cv_macro_f1_mean", "student_test_top1_accuracy", "student_test_macro_f1", "student_test_log_loss", "route_continuity", "inference_ms_per_trip", "training_time_sec"]}
    X = np.zeros((12, 1))
    y = np.tile([1, 0], 6)
    groups = np.repeat(["a", "b", "c"], 4)
    _, trials, comparison, summary = tuning.run_bayesian_optimisation(
        X, y, groups, [], X, y, {}, untuned, n_trials=1, k_candidates=2,
    )
    assert trials.iloc[0]["cv_macro_f1"] == 1.
    assert summary["best_cv_macro_f1"] == 1.
    assert comparison["cv_macro_f1"].tolist() == [1., 1.]


def test_failed_or_changed_selection_never_keeps_old_response():
    key = ("http://server", "demo", "student_test")
    state = {}
    assert select_evaluation(state, key)
    response = Mock()
    response.json.return_value = {"trip_id": "demo", "partition": "student_test"}
    request_evaluation(state, key, lambda: response)
    assert state["last_eval_result"]["trip_id"] == "demo"
    response.raise_for_status.side_effect = RuntimeError("HTTP 500")
    with pytest.raises(RuntimeError):
        request_evaluation(state, key, lambda: response)
    assert "last_eval_result" not in state
    state["last_eval_result"] = {"trip_id": "old"}
    assert select_evaluation(state, ("http://other-server", "demo", "student_test"))
    assert "last_eval_result" not in state


def test_deployment_verifier_makes_http_requests_and_saves_only_observed_evidence(monkeypatch, tmp_path):
    from src import capture_deployment_screenshots as capture
    trip = {"trip_id": "demo", "partition": "student_test"}
    def response(value):
        result = Mock()
        result.json.return_value = value
        return result
    get = Mock(side_effect=[response({"status": "online"}), response({"trips": [trip]})])
    post = Mock(return_value=response({"status": "success", **trip, "metrics": {}}))
    monkeypatch.setattr(capture.requests, "get", get)
    monkeypatch.setattr(capture.requests, "post", post)
    result = capture.verify_deployment("http://server:8000", output_dir=tmp_path)
    assert [c.args[0] for c in get.call_args_list] == ["http://server:8000/health", "http://server:8000/trips"]
    assert post.call_args.args[0] == "http://server:8000/evaluate"
    assert json.loads((tmp_path / "http_verification.json").read_text())["evaluation"]["trip_id"] == "demo"
    assert result["kind"] == "http_client_verification"
    assert not list(tmp_path.glob("*.png"))
