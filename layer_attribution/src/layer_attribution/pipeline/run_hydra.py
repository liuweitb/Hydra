#!/usr/bin/env python3
"""Run Hydra on one MP3D-format dataset folder with a chosen dataset config + label space."""

import argparse
import pathlib
import shutil
import signal
import sys

import hydra_python as hydra
import spark_dataset_interfaces as sdi


def _decompose_pose(pose):
    q_xyzw = pose.rotation.as_quat()
    return [q_xyzw[i] for i in [3, 0, 1, 2]], pose.translation


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=pathlib.Path)
    ap.add_argument("output", type=pathlib.Path)
    ap.add_argument("--dataset-config", default="uhumans2")
    ap.add_argument("--label-space", default="uhumans2_office")
    ap.add_argument("--config-path", type=pathlib.Path, default=None, help="dir with datasets/ and label_spaces/")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = args.output.expanduser().absolute()
    if out.exists():
        if not args.force:
            sys.exit(f"output exists (use --force): {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    hydra.set_glog_level(0, 0)
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # an import above disables SIGINT

    loader = sdi.FileDataLoader(args.dataset.expanduser().resolve())
    sensor = hydra.make_camera(**loader.intrinsics)
    pipeline = hydra.load_pipeline(
        sensor,
        args.dataset_config,
        args.label_space,
        config_path=args.config_path,
        output_path=out,
        freeze_global_info=False,
    )
    if pipeline is None:
        sys.exit("failed to load pipeline")

    def step(packet):
        rotation, translation = _decompose_pose(packet.pose)
        pipeline.step(
            packet.timestamp, translation, rotation, packet.depth, packet.labels, packet.color, **packet.extras
        )

    sdi.DataLoader.run(loader, step, max_steps=args.max_steps, step_mode=False, show_progress=True)
    pipeline.save(str(out))


if __name__ == "__main__":
    main()
