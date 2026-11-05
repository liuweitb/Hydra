#!/usr/bin/env python3
"""Convert a ROS1 bag (uHumans2 / TESSE) into the folder layout read by Hydra's FileDataLoader.

Output: camera_info.yaml, color/rgb_%07d.png, depth/depth_%07d.tiff (float32 m),
labels/labels_%07d.png (single-channel ids), poses.csv (world_T_camera, optical axes).
Read the bag only with `python -I` from a directory that does not contain the bag.
"""
import argparse
import csv
import json
import struct
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation, Slerp

IMAGE_T = "sensor_msgs/msg/Image"
INFO_T = "sensor_msgs/msg/CameraInfo"
ODOM_T = "nav_msgs/msg/Odometry"
TF_TYPES = ("tf2_msgs/msg/TFMessage", "tf/msg/tfMessage")
DEFAULT_COLOR_MAP = Path(__file__).with_name("uhumans2_office_color_map.csv")
# x_opt = -y_flu, y_opt = -z_flu, z_opt = x_flu; same matrix as Hydra's habitat b_R_c
FLU_R_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def stamp_ns(raw):
    # every image/odom message starts with Header{seq u32, sec u32, nsec u32}
    sec, nsec = struct.unpack_from("<II", raw, 4)
    return sec * 1_000_000_000 + nsec


def print_topics(reader, bag_path):
    dur = (reader.end_time - reader.start_time) / 1e9
    size_gb = Path(bag_path).stat().st_size / 1e9
    print(f"bag: {bag_path}  size={size_gb:.3f} GB  duration={dur:.1f} s  messages={reader.message_count}")
    print(f"{'topic':<48}{'type':<36}{'count':>9}{'rate_hz':>9}")
    for c in sorted(reader.connections, key=lambda c: c.topic):
        rate = c.msgcount / dur if dur > 0 else float("nan")
        print(f"{c.topic:<48}{c.msgtype:<36}{c.msgcount:>9}{rate:>9.2f}")


def pick(cands, what, override):
    if override:
        return override
    if len(cands) == 1:
        return cands[0]
    raise SystemExit(f"cannot auto-detect {what} topic, candidates={cands}; pass it explicitly")


def detect_topics(reader, args):
    by_topic = {c.topic: c for c in reader.connections}
    imgs = [c.topic for c in reader.connections if c.msgtype == IMAGE_T]
    low = lambda t: t.lower()
    depth = [t for t in imgs if "depth" in low(t)]
    label = [t for t in imgs if any(k in low(t) for k in ("seg", "label", "semantic")) and t not in depth]
    rgb = [t for t in imgs if t not in depth and t not in label and "right" not in low(t)]
    topics = {
        "rgb": pick(rgb, "rgb", args.rgb_topic),
        "depth": pick(depth, "depth", args.depth_topic),
        "label": pick(label, "label", args.label_topic),
    }
    for k, t in topics.items():
        if t not in by_topic:
            raise SystemExit(f"{k} topic {t} not in bag")
    infos = [c.topic for c in reader.connections if c.msgtype == INFO_T]
    ns = lambda t: t.rsplit("/", 2)[0]
    same_ns = [t for t in infos if ns(t) == ns(topics["depth"])] or [t for t in infos if ns(t) == ns(topics["rgb"])]
    topics["info"] = pick(same_ns if len(same_ns) == 1 else infos, "camera_info", args.info_topic)
    odoms = [c.topic for c in reader.connections if c.msgtype == ODOM_T]
    topics["odom"] = args.odom_topic or (odoms[0] if len(odoms) == 1 else None)
    topics["tf"] = [c.topic for c in reader.connections if c.msgtype in TF_TYPES]
    return topics


class TfBuffer:
    """Minimal time-interpolating tf tree; transforms are T_parent_child."""

    def __init__(self):
        self.raw = defaultdict(list)
        self.static = set()
        self.adj = defaultdict(set)

    def add(self, parent, child, t, p, q, static):
        self.raw[(parent, child)].append((t, p, q))
        self.adj[parent].add(child)
        self.adj[child].add(parent)
        if static:
            self.static.add((parent, child))

    def finalize(self):
        self.tab = {}
        for key, rows in self.raw.items():
            rows.sort(key=lambda r: r[0])
            self.tab[key] = (
                np.array([r[0] for r in rows], dtype=np.int64),
                np.array([r[1] for r in rows]),
                Rotation.from_quat([r[2] for r in rows]),
            )

    def frames(self):
        return set(self.adj)

    def _edge(self, key, t, tol):
        times, pos, rots = self.tab[key]
        T = np.eye(4)
        if key in self.static or len(times) == 1:
            if key not in self.static and abs(int(times[0]) - t) > tol:
                return None
            T[:3, :3] = rots[0].as_matrix()
            T[:3, 3] = pos[0]
            return T
        if t < times[0] - tol or t > times[-1] + tol:
            return None
        tc = min(max(t, int(times[0])), int(times[-1]))
        i1 = min(max(int(np.searchsorted(times, tc)), 1), len(times) - 1)
        i0 = i1 - 1
        a = (tc - times[i0]) / max(times[i1] - times[i0], 1)
        T[:3, :3] = Slerp([0.0, 1.0], Rotation.concatenate([rots[i0], rots[i1]]))([a]).as_matrix()[0]
        T[:3, 3] = (1 - a) * pos[i0] + a * pos[i1]
        return T

    def _path(self, a, b):
        prev = {a: None}
        dq = deque([a])
        while dq:
            n = dq.popleft()
            if n == b:
                break
            for m in self.adj[n]:
                if m not in prev:
                    prev[m] = n
                    dq.append(m)
        if b not in prev:
            return None
        path, n = [], b
        while n is not None:
            path.append(n)
            n = prev[n]
        return path[::-1]

    def lookup(self, target, source, t, tol=100_000_000):
        """T_target_source at time t (ns) or None."""
        if target == source:
            return np.eye(4)
        if target not in self.adj or source not in self.adj:
            return None
        if not hasattr(self, "_paths"):
            self._paths = {}
        path = self._paths.setdefault((target, source), self._path(target, source))
        if path is None:
            return None
        T = np.eye(4)
        for a, b in zip(path[:-1], path[1:]):
            if (a, b) in self.tab:
                E = self._edge((a, b), t, tol)
            else:
                E = self._edge((b, a), t, tol)
                if E is not None:
                    E = np.linalg.inv(E)
            if E is None:
                return None
            T = T @ E
        return T


class OdomTrack:
    def __init__(self):
        self.t, self.p, self.q = [], [], []
        self.world = self.body = None

    def add(self, t, p, q):
        self.t.append(t)
        self.p.append(p)
        self.q.append(q)

    def finalize(self):
        order = np.argsort(self.t)
        self.t = np.array(self.t, dtype=np.int64)[order]
        self.p = np.array(self.p)[order]
        self.rot = Rotation.from_quat(np.array(self.q)[order])

    def at(self, t, tol=100_000_000):
        if len(self.t) == 0 or t < self.t[0] - tol or t > self.t[-1] + tol:
            return None
        tc = min(max(t, int(self.t[0])), int(self.t[-1]))
        i1 = min(max(int(np.searchsorted(self.t, tc)), 1), len(self.t) - 1)
        i0 = i1 - 1
        a = (tc - self.t[i0]) / max(self.t[i1] - self.t[i0], 1)
        T = np.eye(4)
        T[:3, :3] = Slerp([0.0, 1.0], Rotation.concatenate([self.rot[i0], self.rot[i1]]))([a]).as_matrix()[0]
        T[:3, 3] = (1 - a) * self.p[i0] + a * self.p[i1]
        return T


class Stream:
    """Header stamps of one topic; lookups return the message ordinal in bag order."""

    def __init__(self, stamps):
        s = np.asarray(stamps, dtype=np.int64)
        self.order = np.argsort(s, kind="stable")  # header stamps need not follow record order
        self.sorted = s[self.order]

    def nearest(self, t):
        i = int(np.searchsorted(self.sorted, t))
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.sorted)]
        j = min(cands, key=lambda j: abs(int(self.sorted[j]) - t))
        return int(self.order[j]), abs(int(self.sorted[j]) - t)


def load_color_map(path):
    cmap = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["red"]) << 16) | (int(row["green"]) << 8) | int(row["blue"])
            if cmap.setdefault(key, int(row["id"])) != int(row["id"]):
                raise SystemExit(f"color map conflict at rgb key {key}")
    return cmap


def load_remap(path):
    with open(path) as f:
        return {int(x["sub_id"]): int(x["super_id"]) for x in yaml.safe_load(f)}


def image_array(msg):
    """Decode sensor_msgs/Image into (array, encoding) without cv_bridge."""
    h, w, enc = int(msg.height), int(msg.width), msg.encoding
    buf = np.asarray(msg.data, dtype=np.uint8)
    spec = {
        "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
        "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1), "mono16": (np.uint16, 1), "16UC1": (np.uint16, 1),
        "32FC1": (np.float32, 1), "32SC1": (np.int32, 1), "16SC1": (np.int16, 1),
    }
    if enc not in spec:
        raise SystemExit(f"unsupported image encoding {enc}")
    dt, ch = spec[enc]
    dt = np.dtype(dt).newbyteorder(">" if msg.is_bigendian else "<")
    row = buf[: h * msg.step].reshape(h, msg.step)[:, : w * ch * dt.itemsize]
    arr = np.ascontiguousarray(row).view(dt).reshape(h, w, ch)
    arr = arr.astype(dt.newbyteorder("="), copy=False)
    return (arr[..., 0] if ch == 1 else arr), enc


class Converter:
    def __init__(self, args):
        self.a = args
        self.unmapped = defaultdict(int)
        self.label_max = 0

    def to_rgb(self, arr, enc):
        if enc in ("bgr8", "bgra8"):
            arr = arr[..., [2, 1, 0]]
        if arr.ndim == 3:
            return np.ascontiguousarray(arr[..., :3])
        raise SystemExit(f"rgb topic has single-channel encoding {enc}")

    def to_depth(self, arr, enc):
        if enc == "32FC1":
            return arr.astype(np.float32, copy=False)
        if enc in ("16UC1", "mono16"):
            d = arr.astype(np.float32) * 1e-3
            d[arr == 0] = np.nan  # mirrors Hydra's uint16 path
            return d
        raise SystemExit(f"unsupported depth encoding {enc}")

    def to_labels(self, arr, enc):
        if arr.ndim == 3:
            if enc in ("bgr8", "bgra8"):
                arr = arr[..., [2, 1, 0]]
            px = arr[..., :3].astype(np.uint32)
            key = (px[..., 0] << 16) | (px[..., 1] << 8) | px[..., 2]
            uniq, inv = np.unique(key, return_inverse=True)
            ids = np.empty(len(uniq), dtype=np.int32)
            for i, k in enumerate(uniq):
                v = self.cmap.get(int(k))
                if v is None:
                    self.unmapped[int(k)] += int((key == k).sum())
                    v = 0
                ids[i] = v
            out = ids[inv.reshape(key.shape)]
        else:
            out = arr.astype(np.int32)
            if self.remap is not None:
                uniq, inv = np.unique(out, return_inverse=True)
                ids = np.empty(len(uniq), dtype=np.int32)
                for i, k in enumerate(uniq):
                    ids[i] = self.remap.get(int(k), 0)
                    if int(k) not in self.remap:
                        self.unmapped[int(k)] += int((out == k).sum())
                out = ids[inv.reshape(out.shape)]
        self.label_max = max(self.label_max, int(out.max()))
        if self.label_max > 255:
            raise SystemExit(f"label id {self.label_max} > 255; raw ids need --label-remap")  # keep dtype uint8 for all frames
        return out.astype(np.uint8)


def decide_cam_axes(poses, mode):
    """Return 3x3 R_cam_opt so that world_T_opt = world_T_cam @ R; pick by heading alignment when auto."""
    if mode == "optical":
        return np.eye(3), {}
    if mode == "flu":
        return FLU_R_OPT, {}
    P = np.array([T[:3, 3] for T in poses])
    v = np.diff(P, axis=0)
    speed = np.linalg.norm(v, axis=1)
    mv = speed > 1e-3
    if mv.sum() < 5:
        raise SystemExit("too little motion to infer camera axes; pass --cam-axes optical|flu")
    v = v[mv] / speed[mv, None]
    score = {}
    for name, col in (("optical", 2), ("flu", 0)):
        fwd = np.array([T[:3, col] for T in poses[:-1]])[mv]
        score[name] = float(np.mean(np.sum(fwd * v, axis=1)))
    best = max(score, key=score.get)
    other = min(score, key=score.get)
    print(f"cam-axes alignment (mean cos(forward, heading)): {score}")
    if score[best] < 0.3 or score[best] - score[other] < 0.2:
        raise SystemExit("camera-axes inference ambiguous; pass --cam-axes optical|flu explicitly")
    return (np.eye(3) if best == "optical" else FLU_R_OPT), score


def convert(args):
    bag = Path(args.bag)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} not empty; use --overwrite")
    tol_ns = int(args.tol_ms * 1e6)
    cv = Converter(args)
    cv.cmap = None
    cv.remap = load_remap(args.label_remap) if args.label_remap else None

    with AnyReader([bag]) as reader:
        print_topics(reader, bag)
        topics = detect_topics(reader, args)
        print("selected topics:", json.dumps(topics, indent=1))
        conn = {c.topic: c for c in reader.connections}

        # intrinsics + frame ids from first messages
        c_info, _, raw = next(reader.messages(connections=[conn[topics["info"]]]))
        info = reader.deserialize(raw, c_info.msgtype)
        K = np.array(info.K).reshape(3, 3)
        cam = dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                   width=int(info.width), height=int(info.height))
        first = {}
        for k in ("rgb", "depth", "label"):
            c, _, raw = next(reader.messages(connections=[conn[topics[k]]]))
            m = reader.deserialize(raw, c.msgtype)
            arr, enc = image_array(m)
            first[k] = (arr.shape, enc, m.header.frame_id)
            print(f"first {k}: shape={arr.shape} enc={enc} frame_id={m.header.frame_id!r}")
        for k in first:
            if first[k][0][:2] != (cam["height"], cam["width"]):
                raise SystemExit(f"{k} image shape {first[k][0]} != camera_info {cam['height']}x{cam['width']}")
        if first["label"][0] and len(first["label"][0]) == 3:
            cv.cmap = load_color_map(args.color_map or DEFAULT_COLOR_MAP)
        cam_frame = args.cam_frame or first["depth"][2]
        if first["rgb"][2] != first["depth"][2] or first["label"][2] != first["depth"][2]:
            print("NOTE: rgb/label/depth frame_ids differ; pose is for the depth frame, other streams assumed co-located")

        # pass 1: header stamps + pose sources
        scan = [conn[topics[k]] for k in ("rgb", "depth", "label")]
        stamps = {k: [] for k in ("rgb", "depth", "label")}
        k_of = {topics[k]: k for k in stamps}
        tfb, odom = TfBuffer(), OdomTrack()
        pose_conns = [conn[t] for t in topics["tf"]] + ([conn[topics["odom"]]] if topics["odom"] else [])
        need = None if args.max_frames is None else args.max_frames * args.stride
        t_stop = None
        for c, t_rec, raw in reader.messages(connections=scan + pose_conns):
            if t_stop is not None and t_rec > t_stop:
                break
            if c.topic in k_of:
                stamps[k_of[c.topic]].append(stamp_ns(raw))
                if need is not None and k_of[c.topic] == "depth" and len(stamps["depth"]) == need:
                    t_stop = t_rec + 1_000_000_000
            elif c.msgtype in TF_TYPES:
                msg = reader.deserialize(raw, c.msgtype)
                static = "static" in c.topic
                for tr in msg.transforms:
                    ts = tr.header.stamp.sec * 10**9 + tr.header.stamp.nanosec
                    p = [tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z]
                    q = [tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w]
                    tfb.add(tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/"), ts, p, q, static)
            elif c.msgtype == ODOM_T:
                m = reader.deserialize(raw, c.msgtype)
                pp, qq = m.pose.pose.position, m.pose.pose.orientation
                odom.world, odom.body = m.header.frame_id.lstrip("/"), m.child_frame_id.lstrip("/")
                odom.add(m.header.stamp.sec * 10**9 + m.header.stamp.nanosec, [pp.x, pp.y, pp.z], [qq.x, qq.y, qq.z, qq.w])
        if tfb.raw:
            tfb.finalize()
        if len(odom.t):
            odom.finalize()
        print({k: len(v) for k, v in stamps.items()}, "tf edges:", sorted(tfb.raw), "odom msgs:", len(odom.t))

        # association: depth is the reference stream (Hydra back-projects with depth)
        ref = stamps["depth"]
        streams = {k: Stream(v) for k, v in stamps.items()}
        frames = tfb.frames() if tfb.raw else set()
        world_frame = args.world_frame or odom.world or next((f for f in ("world", "map", "odom") if f in frames), None)
        src = args.pose_source
        if src == "auto":
            tf_ok = bool(world_frame) and world_frame in frames and cam_frame in frames and tfb._path(world_frame, cam_frame) is not None
            src = "tf" if tf_ok else ("odom" if len(odom.t) else None)
            if src is None:
                raise SystemExit(f"no pose source: tf frames={sorted(frames)}, odom msgs={len(odom.t)}; check --world-frame/--cam-frame")
        if world_frame is None:
            raise SystemExit(f"cannot guess world frame from {sorted(frames)}; pass --world-frame")
        print(f"pose source={src} world_frame={world_frame} cam_frame={cam_frame} odom_body={odom.body}")

        def world_T_cam(t):
            if src == "tf":
                return tfb.lookup(world_frame, cam_frame, t)
            Tb = odom.at(t)
            if Tb is None:
                return None
            Tc = tfb.lookup(odom.body, cam_frame, t) if tfb.raw else None
            return Tb @ (Tc if Tc is not None else np.eye(4))

        plan, poses, dts, skipped = [], [], [], 0
        for ri in range(0, len(ref), args.stride):
            if args.max_frames is not None and len(plan) >= args.max_frames:
                break
            t = ref[ri]
            ci, dc = streams["rgb"].nearest(t)
            li, dl = streams["label"].nearest(t)
            T = world_T_cam(t)
            if dc > tol_ns or dl > tol_ns or T is None:
                skipped += 1
                continue
            plan.append({"depth": ri, "rgb": ci, "label": li, "t": t})
            poses.append(T)
            dts.append((dc, dl))
        if not plan:
            raise SystemExit("no frames could be associated; check topics, tolerance and pose source")
        print(f"frames planned={len(plan)} skipped={skipped} max|dt| rgb={max(d[0] for d in dts)/1e6:.2f} ms label={max(d[1] for d in dts)/1e6:.2f} ms")

        R_opt, axes_score = decide_cam_axes(poses, args.cam_axes)
        for T in poses:
            T[:3, :3] = T[:3, :3] @ R_opt

        # pass 2: decode + write only the planned messages
        for sub in ("color", "depth", "labels"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        want = {k: {p[k]: i for i, p in enumerate(plan)} for k in ("rgb", "depth", "label")}
        last = {k: max(want[k]) for k in want}
        seen = {k: -1 for k in want}
        done = {k: 0 for k in want}
        for c, _, raw in reader.messages(connections=scan):
            k = k_of[c.topic]
            seen[k] += 1
            idx = want[k].get(seen[k])
            if idx is not None:
                arr, enc = image_array(reader.deserialize(raw, c.msgtype))
                if k == "rgb":
                    Image.fromarray(cv.to_rgb(arr, enc)).save(out / "color" / f"rgb_{idx:07d}.png")
                elif k == "depth":
                    tifffile.imwrite(out / "depth" / f"depth_{idx:07d}.tiff", cv.to_depth(arr, enc))
                else:
                    Image.fromarray(cv.to_labels(arr, enc)).save(out / "labels" / f"labels_{idx:07d}.png")
                done[k] += 1
                if k == "depth" and done[k] % 200 == 0:
                    print(f"  wrote {done[k]}/{len(plan)} depth frames", flush=True)
            if all(seen[x] >= last[x] for x in want):
                break
        if any(done[k] != len(plan) for k in done):
            raise SystemExit(f"write incomplete: {done} of {len(plan)}")

    t0 = plan[0]["t"]
    quats = Rotation.from_matrix(np.array([T[:3, :3] for T in poses])).as_quat()  # xyzw
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]  # keep sign continuous for downstream slerp
    with open(out / "poses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "tx", "ty", "tz", "qw", "qx", "qy", "qz"])
        for p, T, q in zip(plan, poses, quats):
            w.writerow([p["t"] - t0, *(repr(float(x)) for x in T[:3, 3]), *(repr(float(x)) for x in (q[3], q[0], q[1], q[2]))])
    extra = {k: v for k, v in (("min_range", args.min_range), ("max_range", args.max_range)) if v is not None}
    with open(out / "camera_info.yaml", "w") as f:
        yaml.safe_dump({**cam, **extra}, f, sort_keys=True)
    meta = dict(bag=str(bag), topics=topics, cam_frame=cam_frame, world_frame=world_frame, pose_source=src,
                cam_axes=args.cam_axes, cam_axes_scores=axes_score, stride=args.stride, max_frames=args.max_frames,
                tol_ms=args.tol_ms, frames=len(plan), skipped=skipped, first_stamp_ns=int(t0),
                label_unmapped={str(k): v for k, v in cv.unmapped.items()}, label_max=cv.label_max,
                label_remap=args.label_remap, color_map=str(args.color_map or (DEFAULT_COLOR_MAP if cv.cmap else None)))
    with open(out / "conversion_info.json", "w") as f:
        json.dump(meta, f, indent=1)
    if cv.unmapped:
        print(f"WARNING unmapped label values -> 0: {dict(list(cv.unmapped.items())[:10])}")
    print(f"done: {len(plan)} frames -> {out}")


def verify(out, label_space=None, sample=200):
    """Sanity-check a converted folder the way FileDataLoader will read it."""
    import imageio.v3 as iio
    import pandas as pd

    out = Path(out)
    cam = yaml.safe_load((out / "camera_info.yaml").read_text())
    df = pd.read_csv(out / "poses.csv")
    n = len(df)
    print(f"camera_info: {cam}")
    print(f"poses.csv: {n} rows, columns={list(df.columns)}, t=[{df.timestamp_ns.iloc[0]}, {df.timestamp_ns.iloc[-1]}] ns")
    for sub, pat in (("color", "rgb_{:07d}.png"), ("depth", "depth_{:07d}.tiff"), ("labels", "labels_{:07d}.png")):
        miss = [i for i in range(n) if not (out / sub / pat.format(i)).exists()]
        print(f"{sub}: missing {len(miss)} of {n}")
    dt = np.diff(df.timestamp_ns.to_numpy())
    print(f"dt ms: min={dt.min()/1e6:.1f} median={np.median(dt)/1e6:.1f} max={dt.max()/1e6:.1f} (monotonic={bool((dt > 0).all())})")
    P = df[["tx", "ty", "tz"]].to_numpy()
    Q = df[["qx", "qy", "qz", "qw"]].to_numpy()
    print(f"quat norm err max={np.abs(np.linalg.norm(Q, axis=1) - 1).max():.2e}")
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    R = Rotation.from_quat(Q)
    rstep = np.degrees((R[:-1].inv() * R[1:]).magnitude())
    print(f"trans step m: median={np.median(step):.3f} p99={np.percentile(step, 99):.3f} max={step.max():.3f}; "
          f"rot step deg: median={np.median(rstep):.2f} max={rstep.max():.2f}; path length={step.sum():.1f} m")
    print(f"bbox min={P.min(0).round(2)} max={P.max(0).round(2)}")
    mv = step > 1e-3
    fwd = R.apply([0.0, 0.0, 1.0])[:-1][mv]  # optical z in world
    hdg = np.diff(P, axis=0)[mv]
    hdg /= np.linalg.norm(hdg, axis=1, keepdims=True)
    print(f"optical-z vs heading cos: mean={np.mean(np.sum(fwd * hdg, axis=1)):.3f} (moving frames={int(mv.sum())})")

    idxs = np.unique(np.linspace(0, n - 1, min(sample, n)).astype(int))
    shapes, dtypes = defaultdict(set), defaultdict(set)
    dmin, dmax, nanfrac, zerofrac = np.inf, -np.inf, [], []
    hist = defaultdict(int)
    for i in idxs:
        c = iio.imread(out / "color" / f"rgb_{i:07d}.png")
        d = iio.imread(out / "depth" / f"depth_{i:07d}.tiff")
        lb = iio.imread(out / "labels" / f"labels_{i:07d}.png")
        for k, a in (("color", c), ("depth", d), ("labels", lb)):
            shapes[k].add(a.shape)
            dtypes[k].add(str(a.dtype))
        fin = np.isfinite(d) & (d > 0)
        nanfrac.append(1 - np.isfinite(d).mean())
        zerofrac.append((d == 0).mean())
        if fin.any():
            dmin, dmax = min(dmin, float(d[fin].min())), max(dmax, float(d[fin].max()))
        for v, cnt in zip(*np.unique(lb, return_counts=True)):
            hist[int(v)] += int(cnt)
    print(f"sampled {len(idxs)} frames; shapes={dict(shapes)} dtypes={dict(dtypes)}")
    want = (cam["height"], cam["width"])
    ok = all(s[:2] == want for ss in shapes.values() for s in ss)
    print(f"all image sizes match camera_info {want}: {ok}")
    print(f"depth valid range [{dmin:.3f}, {dmax:.3f}] m; nan frac mean={np.mean(nanfrac):.4f}; zero frac mean={np.mean(zerofrac):.4f}")
    names = {}
    if label_space and Path(label_space).exists():
        names = {x["label"]: x["name"] for x in yaml.safe_load(Path(label_space).read_text())["label_names"]}
    tot = sum(hist.values())
    print("label ids (pixel share):", ", ".join(f"{k}:{names.get(k, '?')}={100 * v / tot:.2f}%" for k, v in sorted(hist.items())))
    unknown = sorted(k for k in hist if names and k not in names)
    if unknown:
        print(f"WARNING label ids not in label space: {unknown}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--list", action="store_true", help="print the topic table and exit")
    ap.add_argument("--verify-only", action="store_true", help="only sanity-check an existing output folder (pass it as `bag`)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="keep every k-th depth frame")
    ap.add_argument("--tol-ms", type=float, default=25.0, help="max |dt| between depth and rgb/label stamps")
    ap.add_argument("--rgb-topic")
    ap.add_argument("--depth-topic")
    ap.add_argument("--label-topic")
    ap.add_argument("--info-topic")
    ap.add_argument("--odom-topic")
    ap.add_argument("--pose-source", choices=["auto", "tf", "odom"], default="auto")
    ap.add_argument("--world-frame")
    ap.add_argument("--cam-frame", help="defaults to the depth image header frame_id")
    ap.add_argument("--cam-axes", choices=["auto", "optical", "flu"], default="auto",
                    help="axis convention of --cam-frame; Hydra needs optical (x right, y down, z forward)")
    ap.add_argument("--color-map", help="CSV name,red,green,blue,alpha,id for RGB label images")
    ap.add_argument("--label-remap", help="Hydra label_remaps yaml (sub_id->super_id) for single-channel label images")
    ap.add_argument("--min-range", type=float)
    ap.add_argument("--max-range", type=float)
    ap.add_argument("--label-space", default=r"D:\Hydra\config\label_spaces\uhumans2_office_label_space.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.verify_only:
        verify(args.bag, args.label_space)
        return
    if args.list:
        with AnyReader([Path(args.bag)]) as reader:
            print_topics(reader, args.bag)
        return
    if not args.out:
        ap.error("out directory required")
    convert(args)
    verify(args.out, args.label_space)


if __name__ == "__main__":
    sys.exit(main())
