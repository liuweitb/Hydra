import json
import os
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage
from scipy.spatial.transform import Rotation

from layer_attribution.injection import inject
from layer_attribution.injection.synthetic import make_dataset

FAULTS = sorted(inject.FAULT_META)
# params that make randomness visible on a 30-frame set
EXTRA = {"frame_drop": {"chunk_len": 3}}
N = 30
INJECT_PY = Path(inject.__file__)


@pytest.fixture(scope="session")
def src(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("src"), n=N)


@pytest.fixture(scope="session")
def src16(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("src16"), n=8, depth_dtype="uint16", label_dtype="uint16")


def run(src, tmp_path, fault, sev, seed=0, name="out", **params):
    p = {**EXTRA.get(fault, {}), **params}
    return inject.inject(src, tmp_path / name, fault, sev, seed, p)


def load(root):
    """Same read path as spark_dataset_interfaces.FileDataLoader: poses.csv rows index the frame files."""
    poses = pd.read_csv(Path(root) / "poses.csv")
    out = []
    for i in range(len(poses)):
        f = {}
        for ch, pat in inject.CHANNELS.items():
            f[ch] = iio.imread(Path(root) / ch / pat.format(i))
        out.append(f)
    return poses, out


def tree_bytes(root):
    root = Path(root)
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "injection_manifest.json"}


# ---------------------------------------------------------------- generic contract

@pytest.mark.parametrize("fault", FAULTS)
def test_same_seed_is_deterministic(src, tmp_path, fault):
    m1 = run(src, tmp_path, fault, 0.5, seed=7, name="a")
    m2 = run(src, tmp_path, fault, 0.5, seed=7, name="b")
    assert m1 == m2
    assert tree_bytes(tmp_path / "a") == tree_bytes(tmp_path / "b")
    run(src, tmp_path, fault, 0.5, seed=8, name="c")
    assert tree_bytes(tmp_path / "a") != tree_bytes(tmp_path / "c")


@pytest.mark.parametrize("fault", FAULTS)
def test_severity_zero_is_identity(src, tmp_path, fault):
    m = run(src, tmp_path, fault, 0.0)
    assert tree_bytes(tmp_path / "out") == tree_bytes(src)
    assert m["n_units_modified"] == 0
    assert m["n_frames_out"] == N
    changed_keys = {"n_changed", "n_newly_invalid", "trans_error_m", "yaw_error_deg", "dropped_before"}
    for k, v in m["summary"].items():
        if k in changed_keys:
            assert v["max"] == 0.0


@pytest.mark.parametrize("fault", FAULTS)
def test_manifest_schema(src, tmp_path, fault):
    m = run(src, tmp_path, fault, 0.4, seed=3)
    on_disk = json.loads((tmp_path / "out" / "injection_manifest.json").read_text(),
                         parse_constant=lambda c: pytest.fail(f"non-finite {c} in manifest"))
    assert on_disk["fault"] == fault and on_disk["severity"] == 0.4 and on_disk["seed"] == 3
    assert on_disk["expected_first_layer"] in inject.LAYER_ORDER + ["all"]
    assert on_disk["layer_order"] == inject.LAYER_ORDER
    assert len(on_disk["frame_stats"]) == on_disk["n_frames_out"]
    assert on_disk["params"].keys() == inject.DEFAULTS[fault].keys()
    assert on_disk["n_frames_in"] == N
    assert m["frame_stats"] == on_disk["frame_stats"]


@pytest.mark.parametrize("fault", FAULTS)
def test_shapes_dtypes_preserved_and_loader_reads(src, tmp_path, fault):
    run(src, tmp_path, fault, 0.7, seed=1)
    p0, f0 = load(src)
    p1, f1 = load(tmp_path / "out")
    assert list(p1.columns) == list(p0.columns) == ["timestamp_ns", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
    assert len(f1) == len(p1) <= len(f0)
    for a, b in zip(f0, f1):
        for ch in a:
            assert a[ch].shape == b[ch].shape and a[ch].dtype == b[ch].dtype
    q = p1[["qx", "qy", "qz", "qw"]].to_numpy()
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-9)


def test_only_target_channel_changes(src, tmp_path):
    base = tree_bytes(src)
    for fault, keep in [("label_confusion", {"depth", "color"}), ("depth_noise", {"labels", "color"}),
                        ("pose_drift", {"depth", "labels", "color"})]:
        run(src, tmp_path, fault, 0.5, name=fault)
        out = tree_bytes(tmp_path / fault)
        for k, v in out.items():
            if k.split(os.sep)[0] in keep or k == "camera_info.yaml" or (k == "poses.csv" and fault != "pose_drift"):
                assert v == base[k], k


def test_unchanged_files_are_hard_links_or_copies(src, tmp_path):
    run(src, tmp_path, "depth_noise", 0.5, name="link")
    a, b = src / "color" / "rgb_0000003.png", tmp_path / "link" / "color" / "rgb_0000003.png"
    assert os.path.samefile(a, b) or a.read_bytes() == b.read_bytes()
    inject.inject(src, tmp_path / "cp", "depth_noise", 0.5, 0, copy=True)
    assert not os.path.samefile(a, tmp_path / "cp" / "color" / "rgb_0000003.png")


def test_input_validation(src, tmp_path):
    with pytest.raises(ValueError, match="severity"):
        inject.inject(src, tmp_path / "x", "depth_noise", 1.5)
    with pytest.raises(ValueError, match="unknown fault"):
        inject.inject(src, tmp_path / "x", "nope", 0.5)
    with pytest.raises(ValueError, match="unknown params"):
        inject.inject(src, tmp_path / "x", "depth_noise", 0.5, params={"bogus": 1})
    with pytest.raises(ValueError, match="overlap"):
        inject.inject(src, src / "sub", "depth_noise", 0.5)
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "f").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        inject.inject(src, tmp_path / "full", "depth_noise", 0.5)


def test_cli_roundtrip(src, tmp_path):
    r = subprocess.run([sys.executable, str(INJECT_PY), str(src), str(tmp_path / "cli"),
                        "--fault", "label_dropout", "--severity", "0.3", "--seed", "5",
                        "--param", "granularity=pixel"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "cli" / "injection_manifest.json").read_text())
    assert m["params"]["granularity"] == "pixel" and m["seed"] == 5


# ---------------------------------------------------------------- label faults

def _blobs_all_or_nothing(a, changed):
    for c in np.unique(a):
        lab, n = ndimage.label(a == c)
        for j in range(1, n + 1):
            blob = changed[lab == j]
            assert blob.all() or not blob.any()


def test_label_confusion_matches_manifest(src, tmp_path):
    m = run(src, tmp_path, "label_confusion", 0.5, seed=2)
    _, f0 = load(src)
    _, f1 = load(tmp_path / "out")
    cmap, objs = inject.CONFUSION_MAP, set(inject.OBJECT_LABELS)
    assert set(cmap) == objs and set(cmap.values()) <= objs and all(k != v for k, v in cmap.items())
    for st, a, b in zip(m["frame_stats"], f0, f1):
        a, b = a["labels"], b["labels"]
        ch = a != b
        assert st["n_changed"] == ch.sum() and st["frac_changed"] == ch.sum() / a.size
        assert st["n_object_pixels"] == np.isin(a, list(objs)).sum()
        assert set(np.unique(a[ch])) <= objs
        assert all(b[ch & (a == k)].tolist() == [cmap[k]] * int((ch & (a == k)).sum()) for k in objs)
        assert (a[~ch] == b[~ch]).all()
        assert st["n_changed"] >= 0.5 * st["n_eligible_pixels"]
        _blobs_all_or_nothing(a, ch)
    assert 0.5 <= m["summary"]["frac_object_changed"]["mean"] <= 0.75


def test_label_confusion_pixel_mode_and_full_severity(src, tmp_path):
    m = run(src, tmp_path, "label_confusion", 0.3, seed=2, granularity="pixel")
    assert abs(m["summary"]["frac_object_changed"]["mean"] - 0.3) < 0.03
    m = run(src, tmp_path, "label_confusion", 1.0, name="full")
    assert all(s["frac_object_changed"] == 1.0 for s in m["frame_stats"] if s["n_object_pixels"])


def test_label_confusion_custom_map(src, tmp_path):
    m = run(src, tmp_path, "label_confusion", 1.0, confusion_map={"3": 5})
    _, f0 = load(src)
    _, f1 = load(tmp_path / "out")
    for a, b in zip(f0, f1):
        a, b = a["labels"], b["labels"]
        assert (b[a == 3] == 5).all() and ((a != b) == (a == 3)).all()
    assert m["params"]["confusion_map"] == {"3": 5}


def test_label_dropout_matches_manifest(src, tmp_path):
    m = run(src, tmp_path, "label_dropout", 0.4, seed=4)
    _, f0 = load(src)
    _, f1 = load(tmp_path / "out")
    objs = np.array(inject.OBJECT_LABELS)
    for st, a, b in zip(m["frame_stats"], f0, f1):
        a, b = a["labels"], b["labels"]
        ch = a != b
        assert st["n_changed"] == ch.sum()
        assert np.isin(a[ch], objs).all() and (b[ch] == 0).all()
        assert (a[~ch] == b[~ch]).all()
        assert st["n_changed"] >= 0.4 * st["n_object_pixels"]
        _blobs_all_or_nothing(a, ch)


def test_label_faults_uint16_labels(src16, tmp_path):
    for fault in ("label_confusion", "label_dropout"):
        run(src16, tmp_path, fault, 0.5, name=fault)
        _, f0 = load(src16)
        _, f1 = load(tmp_path / fault)
        assert all(b["labels"].dtype == np.uint16 for b in f1)
        assert any((a["labels"] != b["labels"]).any() for a, b in zip(f0, f1))


# ---------------------------------------------------------------- depth faults

def test_depth_noise_matches_manifest(src, tmp_path):
    m = run(src, tmp_path, "depth_noise", 0.6, seed=9)
    _, f0 = load(src)
    _, f1 = load(tmp_path / "out")
    rels = []
    for st, a, b in zip(m["frame_stats"], f0, f1):
        a, b = a["depth"], b["depth"]
        assert (np.isnan(a) == np.isnan(b)).all()
        ok = np.isfinite(a)
        err = np.abs(b[ok].astype(np.float64) - a[ok])
        assert st["n_changed"] == (a[ok] != b[ok]).sum()
        assert np.isclose(st["mean_abs_err_m"], err[err > 0].mean(), rtol=1e-9)
        assert np.isclose(st["rms_err_m"], np.sqrt((err[err > 0] ** 2).mean()), rtol=1e-9)
        rels.append(((b[ok] - a[ok]) / a[ok]).astype(np.float64))
    # gaussian, sigma proportional to depth: relative std = severity * rel_sigma_max
    assert abs(np.concatenate(rels).std() - 0.6 * 0.05) < 0.002


def test_depth_noise_scales_with_depth_and_uint16(src16, tmp_path):
    m = run(src16, tmp_path, "depth_noise", 1.0, seed=1)
    _, f0 = load(src16)
    _, f1 = load(tmp_path / "out")
    for a, b in zip(f0, f1):
        a, b = a["depth"], b["depth"]
        assert b.dtype == np.uint16 and ((a == 0) == (b == 0)).all()
    far = np.concatenate([a["depth"][a["depth"] > 4000] for a in f0]).size
    assert far > 0
    a_all = np.concatenate([a["depth"].ravel() for a in f0]).astype(float)
    b_all = np.concatenate([b["depth"].ravel() for b in f1]).astype(float)
    v = a_all > 0
    near, farm = v & (a_all < 2500), v & (a_all > 4000)
    assert (b_all - a_all)[farm].std() > 1.3 * (b_all - a_all)[near].std()
    assert m["summary"]["mean_abs_err_m"]["mean"] > 0


def test_depth_holes_matches_manifest(src, tmp_path):
    m = run(src, tmp_path, "depth_holes", 0.8, seed=5)
    _, f0 = load(src)
    _, f1 = load(tmp_path / "out")
    n_dropped = 0
    for st, a, b in zip(m["frame_stats"], f0, f1):
        a, b = a["depth"], b["depth"]
        va, vb = np.isfinite(a) & (a > 0), np.isfinite(b) & (b > 0)
        assert (a[vb] == b[vb]).all()
        assert not (vb & ~va).any()
        assert st["n_newly_invalid"] == (va & ~vb).sum()
        assert st["frac_pixels_newly_invalid"] == (va & ~vb).sum() / a.size
        assert st["n_valid_before"] == va.sum()
        if st["frame_dropped"]:
            n_dropped += 1
            assert not vb.any()
        else:
            assert st["n_rects"] > 0
            assert st["n_newly_invalid"] > 0.8 * 0.5 * a.size * 0.9 * 0.8
    assert n_dropped == sum(s["frame_dropped"] for s in m["frame_stats"])


def test_depth_holes_severity_monotone_and_uint16_zero_fill(src, src16, tmp_path):
    lo = run(src, tmp_path, "depth_holes", 0.2, seed=1, name="lo")["summary"]["frac_pixels_newly_invalid"]["mean"]
    hi = run(src, tmp_path, "depth_holes", 0.9, seed=1, name="hi")["summary"]["frac_pixels_newly_invalid"]["mean"]
    assert hi > 2 * lo > 0
    run(src16, tmp_path, "depth_holes", 1.0, seed=1, name="u16")
    _, f0 = load(src16)
    _, f1 = load(tmp_path / "u16")
    assert any((a["depth"] != b["depth"]).any() for a, b in zip(f0, f1))
    assert all(set(np.unique(b["depth"][a["depth"] != b["depth"]])) <= {0} for a, b in zip(f0, f1))


# ---------------------------------------------------------------- pose drift

def _pose_arrays(root):
    df = pd.read_csv(Path(root) / "poses.csv")
    return (df[["tx", "ty", "tz"]].to_numpy(), df[["qx", "qy", "qz", "qw"]].to_numpy(),
            df["timestamp_ns"].to_numpy())


@pytest.mark.parametrize("mode", ["random_walk", "bias"])
def test_pose_drift_matches_manifest(src, tmp_path, mode):
    m = run(src, tmp_path, "pose_drift", 0.8, seed=11, mode=mode)
    p0, q0, t0 = _pose_arrays(src)
    p1, q1, t1 = _pose_arrays(tmp_path / "out")
    assert (t0 == t1).all()
    assert (p0[0] == p1[0]).all() and (q0[0] == q1[0]).all()
    rel = Rotation.from_quat(q1) * Rotation.from_quat(q0).inv()
    yaw = np.degrees(rel.as_rotvec())
    assert np.abs(yaw[:, :2]).max() < 1e-6, "drift must be pure yaw about world z"
    for i, st in enumerate(m["frame_stats"]):
        assert np.isclose(st["trans_error_m"], np.linalg.norm(p1[i] - p0[i]), atol=1e-12)
        assert np.isclose(st["yaw_error_deg"], yaw[i, 2], atol=1e-9)
        assert np.isclose(st["rot_error_deg"], np.degrees(rel[i].magnitude()), atol=1e-9)
    assert m["extra"]["final_trans_error_m"] == m["frame_stats"][-1]["trans_error_m"]
    assert m["frame_stats"][-1]["trans_error_m"] > 0


def test_pose_drift_bias_is_linear_in_yaw_and_monotone(src, tmp_path):
    m = run(src, tmp_path, "pose_drift", 1.0, seed=3, mode="bias")
    yaw = np.array([s["yaw_error_deg"] for s in m["frame_stats"]])
    assert np.allclose(yaw, yaw[1] * np.arange(N), atol=1e-6) and abs(yaw[1]) == pytest.approx(0.1)
    err = np.array([s["trans_error_m"] for s in m["frame_stats"]])
    assert (np.diff(err) > 0).all()


def test_pose_drift_random_walk_grows_with_severity(src, tmp_path):
    lo = run(src, tmp_path, "pose_drift", 0.2, seed=6, name="lo")["summary"]["trans_error_m"]["mean"]
    hi = run(src, tmp_path, "pose_drift", 1.0, seed=6, name="hi")["summary"]["trans_error_m"]["mean"]
    assert hi == pytest.approx(5 * lo, rel=1e-3)  # same draws scaled; yaw coupling is only second order


# ---------------------------------------------------------------- frame drop

def test_frame_drop_matches_manifest(src, tmp_path):
    m = run(src, tmp_path, "frame_drop", 0.8, seed=4)
    ex = m["extra"]
    n_drop = round(0.8 * 0.5 * N)
    assert ex["n_dropped"] == n_drop == len(ex["dropped_src_indices"]) == N - m["n_frames_out"]
    chunks = ex["dropped_chunks_src"]
    assert sum(e - s for s, e in chunks) == n_drop and all(e - s <= 3 for s, e in chunks)
    assert all(chunks[i][1] < chunks[i + 1][0] for i in range(len(chunks) - 1)), "chunks must not touch"
    assert sorted(ex["kept_src_indices"] + ex["dropped_src_indices"]) == list(range(N))
    for ch, pat in inject.CHANNELS.items():
        names = sorted(p.name for p in (tmp_path / "out" / ch).iterdir())
        assert names == [pat.format(j) for j in range(N - n_drop)]
        for j, s in enumerate(ex["kept_src_indices"]):
            assert (tmp_path / "out" / ch / pat.format(j)).read_bytes() == (src / ch / pat.format(s)).read_bytes()
    src_rows = (src / "poses.csv").read_text().splitlines()
    out_rows = (tmp_path / "out" / "poses.csv").read_text().splitlines()
    assert out_rows == [src_rows[0]] + [src_rows[1 + s] for s in ex["kept_src_indices"]]
    for st in m["frame_stats"]:
        assert st["timestamp_ns"] == st["src_frame"] * 200_000_000
    assert sum(s["dropped_before"] for s in m["frame_stats"]) + (N - 1 - ex["kept_src_indices"][-1]) == n_drop
