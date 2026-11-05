import json
import sys

import pytest

from layer_attribution.sweep import orchestrate as orch

SPEC = {
    "scene": "uh2",
    "source_dataset": "D:/hydra_work/data/uhumans2_office_mp3d",
    "dataset_root": "D:/hydra_work/data/faults",
    "label_space": "D:/Hydra/config/label_spaces/uhumans2_office_label_space.yaml",
    "confusion_map": {"5": 6, "6": 5},
    "faults": ["label_confusion", "depth_noise"],
    "groups": [
        {"role": "benign", "severities": [0.03], "seeds": [0, 1]},
        {"role": "main", "severities": [0.2, 0.5], "seeds": [0]},
        {"role": "main", "severities": [0.2, 0.5], "seeds": [0, 1]},  # overlaps the previous group
    ],
}


def test_job_names_and_legacy_names():
    jobs = orch.enumerate_jobs(SPEC)
    names = [j["name"] for j in jobs]
    assert "uh2_label_confusion_s03_r1" in names and "uh2_depth_noise_s50_r0" in names
    j = next(j for j in jobs if j["name"] == "uh2_depth_noise_s50_r0")
    assert j["legacy"] == "uh2_depth_noise_s50" and j["type"] == "sensor"
    assert next(j for j in jobs if j["name"] == "uh2_depth_noise_s50_r1")["legacy"] is None


def test_overlapping_groups_do_not_duplicate_jobs():
    names = [j["name"] for j in orch.enumerate_jobs(SPEC)]
    assert len(names) == len(set(names)) == 2 * 2 + 2 * 2 + 2 * 2  # benign 2 seeds, s20/s50 seed 0, s20/s50 seed 1


def test_config_jobs():
    spec = dict(SPEC, groups=[], config_faults={"faults": ["rooms_min_size"], "severities": [0.3, 0.6, 1.0]})
    jobs = orch.enumerate_jobs(spec)
    assert [j["name"] for j in jobs] == ["uh2_cfg_rooms_min_size_s30", "uh2_cfg_rooms_min_size_s60",
                                        "uh2_cfg_rooms_min_size_s100"]
    assert all(j["type"] == "config" and j["seed"] == 0 and j["legacy"] is None for j in jobs)


def test_config_jobs_per_fault_severities_come_first():
    plan = [{"fault": "a", "severities": [0.2]}, {"fault": "b", "severities": [0.5, 1.0]}]
    jobs = orch.enumerate_jobs(dict(SPEC, config_faults=plan))
    assert [j["type"] for j in jobs[:3]] == ["config"] * 3 and jobs[3]["type"] == "sensor"
    assert [j["name"] for j in jobs[:3]] == ["uh2_cfg_a_s20", "uh2_cfg_b_s50", "uh2_cfg_b_s100"]


def test_phase2_spec_matches_the_config_launch_plan():
    from layer_attribution.pipeline import config_faults as cf

    spec = json.loads((orch.PROJECT / "configs" / "sweep_phase2.json").read_text())
    from_spec = {(e["fault"], s) for e in spec["config_faults"] for s in e["severities"]}
    assert from_spec == {(r["fault"], r["severity"]) for r in cf.launch_plan()}


def test_inject_command_for_label_faults():
    job = {"fault": "label_confusion", "severity": 0.2, "seed": 1}
    cmd = orch.inject_cmd(SPEC, job, "dst")
    assert cmd[cmd.index("--fault") + 1] == "label_confusion" and cmd[cmd.index("--seed") + 1] == "1"
    assert cmd[cmd.index("--label-space") + 1] == SPEC["label_space"]
    param = cmd[cmd.index("--param") + 1]
    assert json.loads(param.split("=", 1)[1]) == SPEC["confusion_map"]


def test_inject_command_for_geometry_faults_has_no_label_args():
    cmd = orch.inject_cmd(SPEC, {"fault": "depth_noise", "severity": 0.5, "seed": 0}, "dst")
    assert "--label-space" not in cmd and "--param" not in cmd


@pytest.mark.skipif(sys.platform != "win32", reason="windows drive paths")
def test_to_wsl_path():
    assert orch.to_wsl("D:/hydra_work/data/x") == "/mnt/d/hydra_work/data/x"
    script = orch.to_wsl(orch.PROJECT / "scripts" / "run_job.sh")  # location of the checkout is not fixed
    assert script.startswith("/mnt/") and script.endswith("/scripts/run_job.sh") and "\\" not in script
