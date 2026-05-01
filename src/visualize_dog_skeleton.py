#!/usr/bin/env python3
"""
Visualize a simplified dog skeleton from project-defined IMU files.

The script produces a torso-centered relative skeleton:
- root is the stern sensor
- all joint positions are expressed in the stern frame
- the first frame is treated as the neutral pose to absorb sensor mounting offsets

This is intended for data sanity checks and relative pose modeling, not for
absolute trajectory reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


SENSOR_TO_JOINT = {
    "00B49A9E": "head",
    "00B49A94": "stern",
    "00B49A9B": "upper_arm_left",
    "00B49702": "upper_arm_right",
    "00B49AA0": "left_hand",
    "00B49A9F": "right_hand",
    "00B49AA1": "upper_leg_left",
    "00B49A98": "upper_leg_right",
    "00B49A9D": "left_foot",
    "00B49A9C": "right_foot",
}

JOINT_ORDER = [
    "stern",
    "head",
    "upper_arm_left",
    "left_hand",
    "upper_arm_right",
    "right_hand",
    "upper_leg_left",
    "left_foot",
    "upper_leg_right",
    "right_foot",
]

BONES = [
    ("stern", "head"),
    ("stern", "upper_arm_left"),
    ("upper_arm_left", "left_hand"),
    ("stern", "upper_arm_right"),
    ("upper_arm_right", "right_hand"),
    ("stern", "upper_leg_left"),
    ("upper_leg_left", "left_foot"),
    ("stern", "upper_leg_right"),
    ("upper_leg_right", "right_foot"),
]

ANCHORS = {
    "neck": np.array([0.18, 0.0, 0.10], dtype=np.float64),
    "left_shoulder": np.array([0.08, 0.12, 0.05], dtype=np.float64),
    "right_shoulder": np.array([0.08, -0.12, 0.05], dtype=np.float64),
    "left_hip": np.array([-0.12, 0.10, -0.08], dtype=np.float64),
    "right_hip": np.array([-0.12, -0.10, -0.08], dtype=np.float64),
}

REST_VECTORS = {
    "head": np.array([0.18, 0.0, 0.10], dtype=np.float64),
    "upper_arm": np.array([0.06, 0.0, -0.18], dtype=np.float64),
    "forearm": np.array([0.03, 0.0, -0.18], dtype=np.float64),
    "upper_leg": np.array([0.00, 0.0, -0.20], dtype=np.float64),
    "lower_leg": np.array([0.03, 0.0, -0.20], dtype=np.float64),
}

GLOBAL_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
GLOBAL_RIGHT = np.array([0.0, 1.0, 0.0], dtype=np.float64)
IMU_RATE_HZ = 40.0
DEFAULT_GYR_MOTION_SCALE = 0.8
DEFAULT_MAX_HEADING_STEP_DEG = 10.0
FILLED_PKL_COLUMNS = (
    "packet_counter",
    "freeacc_x",
    "freeacc_y",
    "freeacc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "is_interpolated",
)

LOCAL_BODY_POINTS = {
    "neck": ANCHORS["neck"],
    "left_shoulder": ANCHORS["left_shoulder"],
    "right_shoulder": ANCHORS["right_shoulder"],
    "left_hip": ANCHORS["left_hip"],
    "right_hip": ANCHORS["right_hip"],
    "withers": np.array([0.10, 0.0, 0.17], dtype=np.float64),
    "rib_top": np.array([-0.01, 0.0, 0.16], dtype=np.float64),
    "pelvis_top": np.array([-0.18, 0.0, 0.09], dtype=np.float64),
    "chest_lower": np.array([0.08, 0.0, -0.10], dtype=np.float64),
    "belly_mid": np.array([-0.02, 0.0, -0.13], dtype=np.float64),
    "groin": np.array([-0.15, 0.0, -0.10], dtype=np.float64),
    "tail_base": np.array([-0.23, 0.0, 0.04], dtype=np.float64),
    "tail_tip": np.array([-0.36, 0.0, 0.10], dtype=np.float64),
}

LOCAL_BACK_LINE = np.asarray(
    [
        [0.18, 0.0, 0.13],
        [0.11, 0.0, 0.18],
        [0.02, 0.0, 0.17],
        [-0.08, 0.0, 0.15],
        [-0.18, 0.0, 0.10],
        [-0.28, 0.0, 0.07],
        [-0.36, 0.0, 0.10],
    ],
    dtype=np.float64,
)

LOCAL_BELLY_LINE = np.asarray(
    [
        [0.14, 0.0, -0.08],
        [0.06, 0.0, -0.13],
        [-0.03, 0.0, -0.14],
        [-0.12, 0.0, -0.11],
        [-0.19, 0.0, -0.08],
    ],
    dtype=np.float64,
)

DEFAULT_SMAL_SOURCE_ALIASES = {
    "stern": {
        "stern",
        "chest",
        "thorax",
        "torso",
        "body",
        "root",
        "spine",
        "spine0",
        "spine_0",
    },
    "neck": {"neck", "throat"},
    "head": {"head", "skull", "muzzle", "nose", "snout"},
    "withers": {"withers", "shoulder_center", "shouldercentre"},
    "pelvis_top": {"pelvis", "pelvis_top", "hip_center", "hipcentre"},
    "tail_base": {"tail_base", "tailroot", "tail_root", "tail0", "tail_0"},
    "tail_tip": {"tail_tip", "tail_end", "tail1", "tail_1", "tail"},
    "left_shoulder": {
        "left_shoulder",
        "l_shoulder",
        "front_left_shoulder",
        "frontleftshoulder",
        "left_front_shoulder",
        "lf_shoulder",
        "lfrontshoulder",
    },
    "right_shoulder": {
        "right_shoulder",
        "r_shoulder",
        "front_right_shoulder",
        "frontrightshoulder",
        "right_front_shoulder",
        "rf_shoulder",
        "rfrontshoulder",
    },
    "upper_arm_left": {
        "left_elbow",
        "l_elbow",
        "front_left_elbow",
        "frontleftelbow",
        "left_front_elbow",
        "lf_elbow",
        "left_foreleg",
    },
    "upper_arm_right": {
        "right_elbow",
        "r_elbow",
        "front_right_elbow",
        "frontrightelbow",
        "right_front_elbow",
        "rf_elbow",
        "right_foreleg",
    },
    "left_hand": {
        "left_paw",
        "l_paw",
        "front_left_paw",
        "frontleftpaw",
        "left_front_paw",
        "lf_paw",
        "left_wrist",
        "left_carpus",
    },
    "right_hand": {
        "right_paw",
        "r_paw",
        "front_right_paw",
        "frontrightpaw",
        "right_front_paw",
        "rf_paw",
        "right_wrist",
        "right_carpus",
    },
    "left_hip": {
        "left_hip",
        "l_hip",
        "back_left_hip",
        "backlefthip",
        "left_back_hip",
        "lh_hip",
    },
    "right_hip": {
        "right_hip",
        "r_hip",
        "back_right_hip",
        "backrighthip",
        "right_back_hip",
        "rh_hip",
    },
    "upper_leg_left": {
        "left_knee",
        "l_knee",
        "back_left_knee",
        "backleftknee",
        "left_back_knee",
        "lh_knee",
        "left_hind_knee",
    },
    "upper_leg_right": {
        "right_knee",
        "r_knee",
        "back_right_knee",
        "backrightknee",
        "right_back_knee",
        "rh_knee",
        "right_hind_knee",
    },
    "left_foot": {
        "left_back_paw",
        "back_left_paw",
        "left_hind_paw",
        "lh_paw",
        "left_ankle",
        "left_hock",
        "left_foot",
    },
    "right_foot": {
        "right_back_paw",
        "back_right_paw",
        "right_hind_paw",
        "rh_paw",
        "right_ankle",
        "right_hock",
        "right_foot",
    },
}


@dataclass(frozen=True)
class TemplateMeshModel:
    vertices: np.ndarray
    faces: np.ndarray
    control_points: np.ndarray
    control_labels: tuple[str, ...]
    source_path: Path


@dataclass(frozen=True)
class BoundTemplateMeshModel:
    vertices: np.ndarray
    faces: np.ndarray
    control_points: np.ndarray
    control_labels: tuple[str, ...]
    source_labels: tuple[str, ...]
    source_path: Path


@dataclass(frozen=True)
class OfficialSMALModel:
    vertices_template: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    posedirs: np.ndarray
    shapedirs: np.ndarray
    j_regressor: object
    kintree_table: np.ndarray
    parents: np.ndarray
    betas: np.ndarray
    rest_vertices: np.ndarray
    rest_joints: np.ndarray
    joint_names: tuple[str, ...]
    source_path: Path
    data_path: Path


@dataclass(frozen=True)
class SMALPreset:
    beta: np.ndarray
    pose: np.ndarray
    trans: np.ndarray
    source_path: Path


SMAL_JOINT_NAMES = (
    "root",
    "pelvis",
    "spine_0",
    "spine_1",
    "spine_2",
    "spine_3",
    "chest",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_front_paw",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_front_paw",
    "neck",
    "head",
    "left_hip",
    "left_knee",
    "left_hock",
    "left_back_paw",
    "right_hip",
    "right_knee",
    "right_hock",
    "right_back_paw",
    "tail_1",
    "tail_2",
    "tail_3",
    "tail_4",
    "tail_5",
    "tail_6",
    "tail_7",
    "snout",
)

SMAL_REFERENCE_JOINT_INDICES = (
    0,
    6,
    7,
    8,
    10,
    11,
    12,
    14,
    15,
    16,
    17,
    18,
    20,
    21,
    22,
    24,
    25,
    31,
)

SMAL_DISABLED_RESIDUAL_JOINT_INDICES = (
    32,  # snout: no direct observation, so avoid unstable mouth-specific deformation
)


class ProgressPrinter:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(int(total), 0)
        self.last_reported = -1
        self.step = 1 if self.total <= 25 else max(self.total // 20, 1)
        print(f"{self.label}: 0/{self.total}", flush=True)

    def update(self, processed: int) -> None:
        processed = max(0, min(int(processed), self.total))
        should_report = (
            processed == self.total
            or processed == 0
            or processed - self.last_reported >= self.step
        )
        if not should_report:
            return
        if processed == self.last_reported:
            return
        self.last_reported = processed
        print(f"{self.label}: {processed}/{self.total}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a stern-centered dog skeleton animation from IMU txt, filled pkl, or aligned action pkl files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data"),
        help="Folder containing MT_*.txt / MT_*.pkl IMU files, or a single aligned action .pkl file. Default: Data",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "txt", "filled-pkl", "aligned-pkl"),
        default="auto",
        help="How to interpret --data-dir. Default: auto",
    )
    parser.add_argument(
        "--target-rate-hz",
        type=float,
        default=None,
        help="Optional target sampling rate in Hz. Use 20 for the IMU-only pipeline.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        help="Optional CSV output path for a single preview audit row.",
    )
    parser.add_argument(
        "--neutral-pose-mode",
        choices=("first-frame", "sequence-median"),
        default="sequence-median",
        help="How to estimate the neutral pose used to absorb mounting offsets. Default: sequence-median",
    )
    parser.add_argument(
        "--gyr-motion-scale",
        type=float,
        default=DEFAULT_GYR_MOTION_SCALE,
        help=f"Scale for gyro-driven dynamic pose enhancement. Use 0 to disable. Default: {DEFAULT_GYR_MOTION_SCALE}",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="Optional video output path (.mp4 or .gif).",
    )
    parser.add_argument(
        "--coords-output",
        type=Path,
        default=None,
        help="Optional coordinate output path (.npz or .csv).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Start frame index for export. If omitted, the script can auto-pick an active window.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to export. Default: until sequence end",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Frame stride for export. Default: 2",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video FPS. Default: 20",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=32.0,
        help="Scatter point size. Default: 32",
    )
    parser.add_argument(
        "--window-selection",
        choices=("manual", "most-active"),
        default="most-active",
        help=(
            "How to choose the exported frame range when --start-frame is not supplied. "
            "Default: most-active"
        ),
    )
    parser.add_argument(
        "--search-step",
        type=int,
        default=20,
        help="Step size used when searching for the most active window. Default: 20",
    )
    parser.add_argument(
        "--root-motion-mode",
        choices=("fixed", "approximate"),
        default="approximate",
        help="Whether to keep the root fixed or add approximate root motion. Default: approximate",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=0.11,
        help="Scale factor applied to the approximate root translation. Default: 0.11",
    )
    parser.add_argument(
        "--vertical-motion-scale",
        type=float,
        default=0.28,
        help="Additional scale factor for vertical root motion. Default: 0.28",
    )
    parser.add_argument(
        "--body-model",
        choices=("shell", "smal"),
        default="smal",
        help=(
            "Body surface renderer. Default: smal, using the official SMAL model with the "
            "wolf_alph3 preset unless overridden."
        ),
    )
    parser.add_argument(
        "--show-skeleton-overlay",
        action="store_true",
        help=(
            "Render skeleton lines, joints, and guide curves on top of the body mesh. "
            "For SMAL this is off by default."
        ),
    )
    parser.add_argument(
        "--smal-model",
        type=Path,
        default=Path("SMAL/wolf_alph3.pkl"),
        help=(
            "Path to a SMAL asset. Supports .npz template meshes, official SMAL .pkl files, "
            "and preset .pkl files such as SMAL/wolf_alph3.pkl. Default: SMAL/wolf_alph3.pkl"
        ),
    )
    parser.add_argument(
        "--smal-mapping",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping from model control labels or indices to current source points. "
            "If omitted, the script tries name-based auto-matching."
        ),
    )
    parser.add_argument(
        "--list-smal-controls",
        action="store_true",
        help="List control labels available in the SMAL template and exit.",
    )
    parser.add_argument(
        "--smal-data",
        type=Path,
        default=None,
        help=(
            "Path to SMAL auxiliary data pkl. When omitted for official SMAL .pkl files, "
            "the script uses the sibling smal_CVPR2017_data.pkl."
        ),
    )
    parser.add_argument(
        "--smal-family-index",
        type=int,
        default=1,
        help="Family mean beta index from smal_CVPR2017_data.pkl. Canidae/dog is 1. Default: 1",
    )
    return parser.parse_args()


def normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return quat / norm


def quaternion_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_inverse(quat: np.ndarray) -> np.ndarray:
    conj = quaternion_conjugate(quat)
    squared_norm = np.sum(quat * quat, axis=-1, keepdims=True)
    squared_norm = np.where(squared_norm == 0.0, 1.0, squared_norm)
    return conj / squared_norm


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(lhs, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def rotate_vectors(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    tiled_vector = np.broadcast_to(vector, (quat.shape[0], 3))
    pure_vector = np.concatenate(
        [np.zeros((quat.shape[0], 1), dtype=np.float64), tiled_vector],
        axis=1,
    )
    rotated = quaternion_multiply(
        quaternion_multiply(quat, pure_vector),
        quaternion_conjugate(quat),
    )
    return rotated[:, 1:]


def quaternion_to_rotation_matrices(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quaternions(quat)
    w, x, y, z = np.moveaxis(quat, -1, 0)

    return np.stack(
        [
            np.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], axis=-1),
            np.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], axis=-1),
            np.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )


def transform_static_points(
    rotation_matrices: np.ndarray,
    translation: np.ndarray,
    local_points: np.ndarray,
) -> np.ndarray:
    rotated = np.einsum("tij,nj->tni", rotation_matrices, local_points)
    return rotated + translation[:, None, :]


def transform_dynamic_points(
    rotation_matrices: np.ndarray,
    translation: np.ndarray,
    local_points: np.ndarray,
) -> np.ndarray:
    rotated = np.einsum("tij,tnj->tni", rotation_matrices, local_points)
    return rotated + translation[:, None, :]


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.shape[0] <= 1:
        return values.copy()

    radius = window_size // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    smoothed = np.empty_like(values, dtype=np.float64)
    for dim in range(values.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed


def normalize_vector_rows(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = vectors / np.maximum(norms, 1e-8)

    if fallback.ndim == 1:
        fallback_values = np.broadcast_to(fallback, vectors.shape)
    else:
        fallback_values = fallback

    invalid = norms[:, 0] < 1e-8
    if np.any(invalid):
        safe[invalid] = fallback_values[invalid]
    return safe


def normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return fallback.copy()
    return vector / norm


def orthonormal_basis(
    axis: np.ndarray,
    lateral_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    primary = normalize_vector(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))

    if lateral_hint is None:
        hint = GLOBAL_UP
    else:
        hint = lateral_hint

    lateral = hint - np.dot(hint, primary) * primary
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm < 1e-8:
        fallback_hint = GLOBAL_RIGHT if abs(primary[2]) > 0.9 else GLOBAL_UP
        lateral = fallback_hint - np.dot(fallback_hint, primary) * primary
        lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm < 1e-8:
        lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        lateral = lateral / lateral_norm

    vertical = normalize_vector(np.cross(primary, lateral), GLOBAL_UP)
    return primary, lateral, vertical


def build_ring(
    center: np.ndarray,
    axis: np.ndarray,
    radius_lateral: float,
    radius_vertical: float,
    sides: int,
    lateral_hint: np.ndarray | None = None,
) -> list[np.ndarray]:
    _, lateral, vertical = orthonormal_basis(axis, lateral_hint=lateral_hint)
    ring = []
    for angle in np.linspace(0.0, 2.0 * np.pi, num=sides, endpoint=False):
        point = (
            center
            + np.cos(angle) * radius_lateral * lateral
            + np.sin(angle) * radius_vertical * vertical
        )
        ring.append(point)
    return ring


def connect_rings(
    ring_a: list[np.ndarray],
    ring_b: list[np.ndarray],
) -> list[list[np.ndarray]]:
    faces: list[list[np.ndarray]] = []
    side_count = min(len(ring_a), len(ring_b))
    for side_index in range(side_count):
        next_index = (side_index + 1) % side_count
        faces.append(
            [
                ring_a[side_index],
                ring_a[next_index],
                ring_b[next_index],
                ring_b[side_index],
            ]
        )
    return faces


def close_ring_cap(ring: list[np.ndarray], reverse: bool = False) -> list[list[np.ndarray]]:
    return [list(reversed(ring)) if reverse else ring]


def build_segment_tube(
    start: np.ndarray,
    end: np.ndarray,
    radius_start: float,
    radius_end: float,
    sides: int = 8,
    lateral_hint: np.ndarray | None = None,
    close_start: bool = False,
    close_end: bool = False,
) -> list[list[np.ndarray]]:
    axis = end - start
    if float(np.linalg.norm(axis)) < 1e-8:
        return []

    ring_start = build_ring(
        center=start,
        axis=axis,
        radius_lateral=radius_start,
        radius_vertical=radius_start,
        sides=sides,
        lateral_hint=lateral_hint,
    )
    ring_end = build_ring(
        center=end,
        axis=axis,
        radius_lateral=radius_end,
        radius_vertical=radius_end,
        sides=sides,
        lateral_hint=lateral_hint,
    )

    faces = connect_rings(ring_start, ring_end)
    if close_start:
        faces.extend(close_ring_cap(ring_start, reverse=False))
    if close_end:
        faces.extend(close_ring_cap(ring_end, reverse=True))
    return faces


def build_local_torso_faces() -> list[list[np.ndarray]]:
    shoulder_mid = 0.5 * (LOCAL_BODY_POINTS["left_shoulder"] + LOCAL_BODY_POINTS["right_shoulder"])
    hip_mid = 0.5 * (LOCAL_BODY_POINTS["left_hip"] + LOCAL_BODY_POINTS["right_hip"])
    spine_axis = shoulder_mid - hip_mid
    lateral_hint = LOCAL_BODY_POINTS["left_shoulder"] - LOCAL_BODY_POINTS["right_shoulder"]

    chest_ring = build_ring(
        center=np.array([0.10, 0.0, 0.05], dtype=np.float64),
        axis=spine_axis,
        radius_lateral=0.16,
        radius_vertical=0.12,
        sides=12,
        lateral_hint=lateral_hint,
    )
    rib_ring = build_ring(
        center=np.array([0.00, 0.0, 0.05], dtype=np.float64),
        axis=spine_axis,
        radius_lateral=0.19,
        radius_vertical=0.15,
        sides=12,
        lateral_hint=lateral_hint,
    )
    belly_ring = build_ring(
        center=np.array([-0.10, 0.0, 0.02], dtype=np.float64),
        axis=spine_axis,
        radius_lateral=0.15,
        radius_vertical=0.12,
        sides=12,
        lateral_hint=lateral_hint,
    )
    pelvis_ring = build_ring(
        center=np.array([-0.20, 0.0, 0.03], dtype=np.float64),
        axis=spine_axis,
        radius_lateral=0.12,
        radius_vertical=0.10,
        sides=12,
        lateral_hint=lateral_hint,
    )

    faces: list[list[np.ndarray]] = []
    faces.extend(connect_rings(chest_ring, rib_ring))
    faces.extend(connect_rings(rib_ring, belly_ring))
    faces.extend(connect_rings(belly_ring, pelvis_ring))
    faces.extend(close_ring_cap(chest_ring, reverse=False))
    faces.extend(close_ring_cap(pelvis_ring, reverse=True))
    faces.extend(
        build_segment_tube(
            start=LOCAL_BODY_POINTS["tail_base"],
            end=LOCAL_BODY_POINTS["tail_tip"],
            radius_start=0.028,
            radius_end=0.010,
            sides=7,
            lateral_hint=lateral_hint,
            close_end=True,
        )
    )
    return faces


def build_body_meshes(
    current: np.ndarray,
    current_body_points: dict[str, np.ndarray],
    joint_index: dict[str, int],
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[list[np.ndarray]]]:
    limb_faces: list[list[np.ndarray]] = []
    head_faces: list[list[np.ndarray]] = []
    ear_faces: list[list[np.ndarray]] = []

    left_lateral_hint = (
        current_body_points["left_shoulder"] - current_body_points["right_shoulder"]
    )
    right_lateral_hint = -left_lateral_hint

    neck_point = current_body_points["neck"]
    head_tip = current[joint_index["head"]]
    rough_head_axis = head_tip - neck_point
    rough_head_axis = normalize_vector(rough_head_axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    head_base = neck_point + 0.02 * rough_head_axis + 0.02 * GLOBAL_UP
    head_axis, head_lateral, head_vertical = orthonormal_basis(
        head_tip - head_base,
        lateral_hint=left_lateral_hint,
    )
    muzzle_tip = head_tip + 0.09 * head_axis - 0.01 * head_vertical
    head_faces.extend(
        build_segment_tube(
            start=head_base,
            end=head_tip,
            radius_start=0.075,
            radius_end=0.055,
            sides=10,
            lateral_hint=head_lateral,
            close_start=True,
            close_end=False,
        )
    )
    head_faces.extend(
        build_segment_tube(
            start=head_tip,
            end=muzzle_tip,
            radius_start=0.042,
            radius_end=0.016,
            sides=8,
            lateral_hint=head_lateral,
            close_start=False,
            close_end=True,
        )
    )

    left_ear_base = head_base - 0.02 * head_axis + 0.045 * head_vertical + 0.055 * head_lateral
    right_ear_base = head_base - 0.02 * head_axis + 0.045 * head_vertical - 0.055 * head_lateral
    left_ear_peak = head_base - 0.005 * head_axis + 0.125 * head_vertical + 0.070 * head_lateral
    right_ear_peak = head_base - 0.005 * head_axis + 0.125 * head_vertical - 0.070 * head_lateral
    left_ear_back = head_base - 0.060 * head_axis + 0.050 * head_vertical + 0.035 * head_lateral
    right_ear_back = head_base - 0.060 * head_axis + 0.050 * head_vertical - 0.035 * head_lateral
    ear_faces.append([left_ear_base, left_ear_peak, left_ear_back])
    ear_faces.append([right_ear_base, right_ear_peak, right_ear_back])

    forelegs = [
        (
            current_body_points["left_shoulder"],
            "upper_arm_left",
            "left_hand",
            left_lateral_hint,
        ),
        (
            current_body_points["right_shoulder"],
            "upper_arm_right",
            "right_hand",
            right_lateral_hint,
        ),
    ]
    hindlegs = [
        (
            current_body_points["left_hip"],
            "upper_leg_left",
            "left_foot",
            left_lateral_hint,
        ),
        (
            current_body_points["right_hip"],
            "upper_leg_right",
            "right_foot",
            right_lateral_hint,
        ),
    ]

    for anchor, upper_joint_name, lower_joint_name, lateral_hint in forelegs:
        upper_joint = current[joint_index[upper_joint_name]]
        lower_joint = current[joint_index[lower_joint_name]]
        limb_faces.extend(
            build_segment_tube(
                start=anchor,
                end=upper_joint,
                radius_start=0.05,
                radius_end=0.035,
                sides=7,
                lateral_hint=lateral_hint,
            )
        )
        limb_faces.extend(
            build_segment_tube(
                start=upper_joint,
                end=lower_joint,
                radius_start=0.03,
                radius_end=0.02,
                sides=7,
                lateral_hint=lateral_hint,
                close_end=True,
            )
        )

    for anchor, upper_joint_name, lower_joint_name, lateral_hint in hindlegs:
        upper_joint = current[joint_index[upper_joint_name]]
        lower_joint = current[joint_index[lower_joint_name]]
        limb_faces.extend(
            build_segment_tube(
                start=anchor,
                end=upper_joint,
                radius_start=0.055,
                radius_end=0.038,
                sides=7,
                lateral_hint=lateral_hint,
            )
        )
        limb_faces.extend(
            build_segment_tube(
                start=upper_joint,
                end=lower_joint,
                radius_start=0.032,
                radius_end=0.022,
                sides=7,
                lateral_hint=lateral_hint,
                close_end=True,
            )
        )

    return limb_faces, head_faces, ear_faces


def read_sensor_txt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    packet_counter = []
    quats = []
    freeacc = []
    gyr = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or "PacketCounter" in stripped:
                continue

            parts = [part.strip() for part in stripped.split(",")]
            if len(parts) != 18:
                raise ValueError(f"Unexpected column count in {path}: {len(parts)}")

            packet_counter.append(int(parts[0]))
            freeacc.append([float(value) for value in parts[4:7]])
            gyr.append([float(value) for value in parts[7:10]])
            quats.append([float(value) for value in parts[14:18]])

    if not quats:
        raise ValueError(f"No IMU data rows found in {path}")

    return (
        np.asarray(packet_counter, dtype=np.int64),
        normalize_quaternions(np.asarray(quats, dtype=np.float64)),
        np.asarray(freeacc, dtype=np.float64),
        np.asarray(gyr, dtype=np.float64),
    )


def read_sensor_pkl(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if not hasattr(payload, "columns"):
        raise ValueError(f"Expected pandas DataFrame in {path}, got {type(payload)!r}")

    missing_columns = [column for column in FILLED_PKL_COLUMNS if column not in payload.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")

    packet_counter = payload["packet_counter"].to_numpy(dtype=np.int64)
    quats = normalize_quaternions(
        payload[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(dtype=np.float64)
    )
    freeacc = payload[["freeacc_x", "freeacc_y", "freeacc_z"]].to_numpy(dtype=np.float64)
    gyr = payload[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=np.float64)
    is_interpolated = payload["is_interpolated"].to_numpy(dtype=bool)

    return packet_counter, quats, freeacc, gyr, is_interpolated


def find_sensor_file(data_dir: Path, device_id: str) -> Path:
    matches = sorted(data_dir.glob(f"*_{device_id}.txt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one txt file for device {device_id} in {data_dir}, found {len(matches)}"
        )
    return matches[0]


def find_sensor_pickle(data_dir: Path, device_id: str) -> Path:
    matches = sorted(data_dir.glob(f"*_{device_id}.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one pkl file for device {device_id} in {data_dir}, found {len(matches)}"
        )
    return matches[0]


def load_project_motion_data(data_dir: Path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    packet_counter = None
    sensor_data_by_joint: dict[str, dict[str, np.ndarray]] = {}

    for device_id, joint_name in SENSOR_TO_JOINT.items():
        sensor_path = find_sensor_file(data_dir, device_id)
        current_packet_counter, current_quats, current_freeacc, current_gyr = read_sensor_txt(sensor_path)

        if packet_counter is None:
            packet_counter = current_packet_counter
        else:
            if current_packet_counter.shape != packet_counter.shape or not np.array_equal(
                current_packet_counter, packet_counter
            ):
                raise ValueError(
                    f"PacketCounter mismatch detected for {joint_name} ({sensor_path.name}). "
                    "Run synchronization before visualization."
                )

        sensor_data_by_joint[joint_name] = {
            "quat": current_quats,
            "freeacc": current_freeacc,
            "gyr": current_gyr,
            "is_interpolated": np.zeros(current_quats.shape[0], dtype=bool),
        }

    assert packet_counter is not None
    return packet_counter, sensor_data_by_joint


def load_filled_motion_data(data_dir: Path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    raw_sensor_data_by_joint: dict[str, dict[str, np.ndarray]] = {}
    common_start = None
    common_end = None

    for device_id, joint_name in SENSOR_TO_JOINT.items():
        sensor_path = find_sensor_pickle(data_dir, device_id)
        (
            current_packet_counter,
            current_quats,
            current_freeacc,
            current_gyr,
            current_is_interpolated,
        ) = read_sensor_pkl(sensor_path)
        raw_sensor_data_by_joint[joint_name] = {
            "packet_counter": current_packet_counter,
            "quat": current_quats,
            "freeacc": current_freeacc,
            "gyr": current_gyr,
            "is_interpolated": current_is_interpolated,
        }

        current_start = int(current_packet_counter[0])
        current_end = int(current_packet_counter[-1])
        common_start = current_start if common_start is None else max(common_start, current_start)
        common_end = current_end if common_end is None else min(common_end, current_end)

    assert common_start is not None and common_end is not None
    if common_end < common_start:
        raise ValueError(f"No common packet interval across required sensors in {data_dir}")

    packet_counter = np.arange(common_start, common_end + 1, dtype=np.int64)
    sensor_data_by_joint: dict[str, dict[str, np.ndarray]] = {}

    for joint_name, raw_values in raw_sensor_data_by_joint.items():
        source_packet_counter = raw_values["packet_counter"]
        in_common_range = (source_packet_counter >= common_start) & (source_packet_counter <= common_end)
        aligned_packet_counter = source_packet_counter[in_common_range]
        if aligned_packet_counter.shape != packet_counter.shape or not np.array_equal(aligned_packet_counter, packet_counter):
            raise ValueError(
                f"Filled pkl packet range mismatch for joint {joint_name} in {data_dir}. "
                "Expected dense common interval after fill."
            )
        sensor_data_by_joint[joint_name] = {
            "quat": raw_values["quat"][in_common_range],
            "freeacc": raw_values["freeacc"][in_common_range],
            "gyr": raw_values["gyr"][in_common_range],
            "is_interpolated": raw_values["is_interpolated"][in_common_range],
        }

    return packet_counter, sensor_data_by_joint


def load_aligned_action_data(
    aligned_pkl_path: Path,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray, dict[str, object]]:
    with aligned_pkl_path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Aligned action payload must be a dict: {aligned_pkl_path}")

    required_keys = {
        "packet_counter",
        "joint_names",
        "quat",
        "freeacc",
        "gyr",
        "is_interpolated",
        "labels_40hz",
        "label_vocab",
    }
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise ValueError(f"Aligned action payload missing required keys {missing}: {aligned_pkl_path}")

    packet_counter = np.asarray(payload["packet_counter"], dtype=np.int64)
    joint_names = [str(name) for name in np.asarray(payload["joint_names"]).tolist()]
    quat = normalize_quaternions(np.asarray(payload["quat"], dtype=np.float64))
    freeacc = np.asarray(payload["freeacc"], dtype=np.float64)
    gyr = np.asarray(payload["gyr"], dtype=np.float64)
    is_interpolated = np.asarray(payload["is_interpolated"], dtype=bool)
    labels_40hz = np.asarray(payload["labels_40hz"]) > 0
    label_vocab = np.asarray(payload["label_vocab"], dtype=object)
    meta = payload.get("meta", {})

    if quat.shape[0] != packet_counter.shape[0]:
        raise ValueError(f"quat length does not match packet_counter in {aligned_pkl_path}")
    if (
        freeacc.shape[:2] != quat.shape[:2]
        or gyr.shape[:2] != quat.shape[:2]
        or is_interpolated.shape[:2] != quat.shape[:2]
    ):
        raise ValueError(f"Aligned IMU arrays have inconsistent shapes in {aligned_pkl_path}")
    if labels_40hz.shape[0] != packet_counter.shape[0]:
        raise ValueError(f"labels_40hz length does not match packet_counter in {aligned_pkl_path}")

    joint_to_index = {joint_name: index for index, joint_name in enumerate(joint_names)}
    missing_joints = [joint_name for joint_name in JOINT_ORDER if joint_name not in joint_to_index]
    if missing_joints:
        raise ValueError(f"Aligned action payload missing required joints {missing_joints}: {aligned_pkl_path}")

    sensor_data_by_joint: dict[str, dict[str, np.ndarray]] = {}
    for joint_name in JOINT_ORDER:
        joint_index = joint_to_index[joint_name]
        sensor_data_by_joint[joint_name] = {
            "quat": quat[:, joint_index],
            "freeacc": freeacc[:, joint_index],
            "gyr": gyr[:, joint_index],
            "is_interpolated": is_interpolated[:, joint_index],
        }

    return packet_counter, sensor_data_by_joint, labels_40hz, label_vocab, meta


def infer_input_format(data_dir: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    if data_dir.is_file():
        if data_dir.suffix.lower() == ".pkl":
            return "aligned-pkl"
        raise FileNotFoundError(f"Unsupported input file for auto mode: {data_dir}")
    if any(data_dir.glob("*.pkl")):
        return "filled-pkl"
    if any(data_dir.glob("*.txt")):
        return "txt"
    raise FileNotFoundError(f"Could not infer input format in {data_dir}; no .txt or .pkl files found")


def quaternion_slerp(quat_a: np.ndarray, quat_b: np.ndarray, fraction: float) -> np.ndarray:
    quat_a = normalize_quaternions(quat_a)
    quat_b = normalize_quaternions(quat_b)
    dot = np.sum(quat_a * quat_b, axis=1, keepdims=True)
    negative = dot < 0.0
    quat_b = np.where(negative, -quat_b, quat_b)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    close = dot[:, 0] > 0.9995
    result = np.empty_like(quat_a)
    if np.any(close):
        result[close] = normalize_quaternions(
            (1.0 - fraction) * quat_a[close] + fraction * quat_b[close]
        )
    if np.any(~close):
        theta_0 = np.arccos(dot[~close])
        sin_theta_0 = np.sin(theta_0)
        theta = fraction * theta_0
        scale_a = np.sin(theta_0 - theta) / np.maximum(sin_theta_0, 1e-8)
        scale_b = np.sin(theta) / np.maximum(sin_theta_0, 1e-8)
        result[~close] = scale_a * quat_a[~close] + scale_b * quat_b[~close]
        result[~close] = normalize_quaternions(result[~close])
    return result


def quaternion_slerp_midpoint(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    return quaternion_slerp(quat_a, quat_b, 0.5)


def align_quaternion_hemisphere(
    quat: np.ndarray,
    reference_quat: np.ndarray | None = None,
) -> np.ndarray:
    aligned = quat.copy()
    if reference_quat is None:
        reference_quat = aligned[0:1]
    dot = np.sum(aligned * np.broadcast_to(reference_quat, aligned.shape), axis=1, keepdims=True)
    aligned = np.where(dot < 0.0, -aligned, aligned)
    return aligned


def estimate_neutral_reference_quaternion(
    segment_in_root: np.ndarray,
    neutral_pose_mode: str,
) -> np.ndarray:
    if neutral_pose_mode == "first-frame":
        return segment_in_root[0:1]
    if neutral_pose_mode == "sequence-median":
        aligned = align_quaternion_hemisphere(segment_in_root)
        reference = np.median(aligned, axis=0, keepdims=True)
        return normalize_quaternions(reference)
    raise ValueError(f"Unsupported neutral pose mode: {neutral_pose_mode}")


def downsample_motion_data(
    packet_counter: np.ndarray,
    sensor_data_by_joint: dict[str, dict[str, np.ndarray]],
    source_rate_hz: float,
    target_rate_hz: float | None,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], float]:
    if target_rate_hz is None or np.isclose(target_rate_hz, source_rate_hz):
        return packet_counter, sensor_data_by_joint, source_rate_hz
    if not np.isclose(source_rate_hz, 40.0) or not np.isclose(target_rate_hz, 20.0):
        raise ValueError(
            f"Unsupported rate conversion: source={source_rate_hz}, target={target_rate_hz}. "
            "Current implementation supports 40Hz -> 20Hz only."
        )

    trimmed_frame_count = packet_counter.shape[0] - (packet_counter.shape[0] % 2)
    if trimmed_frame_count < 2:
        raise ValueError("Need at least 2 frames for 40Hz -> 20Hz downsampling")

    packet_counter = packet_counter[:trimmed_frame_count]
    downsampled_packet_counter = packet_counter[::2].copy()
    downsampled_sensor_data_by_joint: dict[str, dict[str, np.ndarray]] = {}

    for joint_name, sensor_values in sensor_data_by_joint.items():
        quat = sensor_values["quat"][:trimmed_frame_count]
        freeacc = sensor_values["freeacc"][:trimmed_frame_count]
        gyr = sensor_values["gyr"][:trimmed_frame_count]
        is_interpolated = sensor_values["is_interpolated"][:trimmed_frame_count]

        downsampled_sensor_data_by_joint[joint_name] = {
            "quat": quaternion_slerp_midpoint(quat[::2], quat[1::2]),
            "freeacc": 0.5 * (freeacc[::2] + freeacc[1::2]),
            "gyr": 0.5 * (gyr[::2] + gyr[1::2]),
            "is_interpolated": is_interpolated[::2] | is_interpolated[1::2],
        }

    return downsampled_packet_counter, downsampled_sensor_data_by_joint, float(target_rate_hz)


def downsample_label_matrix(
    label_matrix: np.ndarray | None,
    source_rate_hz: float,
    target_rate_hz: float | None,
) -> np.ndarray | None:
    if label_matrix is None:
        return None
    if target_rate_hz is None or np.isclose(target_rate_hz, source_rate_hz):
        return label_matrix
    if not np.isclose(source_rate_hz, 40.0) or not np.isclose(target_rate_hz, 20.0):
        raise ValueError(
            f"Unsupported label rate conversion: source={source_rate_hz}, target={target_rate_hz}. "
            "Current implementation supports 40Hz -> 20Hz only."
        )

    trimmed_frame_count = label_matrix.shape[0] - (label_matrix.shape[0] % 2)
    if trimmed_frame_count < 2:
        raise ValueError("Need at least 2 label frames for 40Hz -> 20Hz downsampling")
    label_matrix = label_matrix[:trimmed_frame_count]
    return label_matrix[::2] | label_matrix[1::2]


def wrap_action_label_text(active_labels: list[str], max_labels_per_line: int = 2) -> str:
    if not active_labels:
        return "Dog action: NONE"
    chunks = [
        " | ".join(active_labels[index : index + max_labels_per_line])
        for index in range(0, len(active_labels), max_labels_per_line)
    ]
    return "Dog action: " + "\n".join(chunks)


def build_frame_action_labels(
    label_matrix: np.ndarray | None,
    label_vocab: np.ndarray | None,
) -> np.ndarray | None:
    if label_matrix is None or label_vocab is None:
        return None
    frame_labels: list[str] = []
    for row in label_matrix:
        active_labels = [str(label_vocab[index]) for index in np.flatnonzero(row)]
        frame_labels.append(wrap_action_label_text(active_labels))
    return np.asarray(frame_labels, dtype=object)


def load_motion_data(
    data_dir: Path,
    input_format: str,
    target_rate_hz: float | None,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], float, str, np.ndarray | None]:
    resolved_input_format = infer_input_format(data_dir, input_format)
    label_matrix = None
    label_vocab = None
    if resolved_input_format == "txt":
        packet_counter, sensor_data_by_joint = load_project_motion_data(data_dir)
        source_rate_hz = IMU_RATE_HZ
    elif resolved_input_format == "filled-pkl":
        packet_counter, sensor_data_by_joint = load_filled_motion_data(data_dir)
        source_rate_hz = IMU_RATE_HZ
    else:
        packet_counter, sensor_data_by_joint, label_matrix, label_vocab, _ = load_aligned_action_data(data_dir)
        source_rate_hz = IMU_RATE_HZ

    packet_counter, sensor_data_by_joint, sample_rate_hz = downsample_motion_data(
        packet_counter=packet_counter,
        sensor_data_by_joint=sensor_data_by_joint,
        source_rate_hz=source_rate_hz,
        target_rate_hz=target_rate_hz,
    )
    label_matrix = downsample_label_matrix(
        label_matrix=label_matrix,
        source_rate_hz=source_rate_hz,
        target_rate_hz=target_rate_hz,
    )
    frame_action_labels = build_frame_action_labels(
        label_matrix=label_matrix,
        label_vocab=label_vocab,
    )
    return packet_counter, sensor_data_by_joint, sample_rate_hz, resolved_input_format, frame_action_labels


def relative_segment_delta(
    root_quat: np.ndarray,
    segment_quat: np.ndarray,
    neutral_pose_mode: str,
) -> np.ndarray:
    root_inverse = quaternion_inverse(root_quat)
    segment_in_root = quaternion_multiply(root_inverse, segment_quat)
    neutral_reference = estimate_neutral_reference_quaternion(
        segment_in_root=segment_in_root,
        neutral_pose_mode=neutral_pose_mode,
    )
    neutral_inverse = quaternion_inverse(neutral_reference)
    return normalize_quaternions(
        quaternion_multiply(segment_in_root, np.broadcast_to(neutral_inverse, segment_in_root.shape))
    )


def build_relative_positions(
    sensor_data_by_joint: dict[str, dict[str, np.ndarray]],
    neutral_pose_mode: str,
    gyr_motion_scale: float,
) -> np.ndarray:
    frame_count = sensor_data_by_joint["stern"]["quat"].shape[0]
    positions = np.zeros((frame_count, len(JOINT_ORDER), 3), dtype=np.float64)
    joint_index = {name: idx for idx, name in enumerate(JOINT_ORDER)}

    stern_quat = sensor_data_by_joint["stern"]["quat"]
    delta = {
        joint_name: relative_segment_delta(
            stern_quat,
            sensor_values["quat"],
            neutral_pose_mode=neutral_pose_mode,
        )
        for joint_name, sensor_values in sensor_data_by_joint.items()
        if joint_name != "stern"
    }
    lower_arm_left_delta = quaternion_slerp(
        delta["upper_arm_left"],
        delta["left_hand"],
        fraction=0.30,
    )
    lower_arm_right_delta = quaternion_slerp(
        delta["upper_arm_right"],
        delta["right_hand"],
        fraction=0.30,
    )
    lower_leg_left_delta = quaternion_slerp(
        delta["upper_leg_left"],
        delta["left_foot"],
        fraction=0.22,
    )
    lower_leg_right_delta = quaternion_slerp(
        delta["upper_leg_right"],
        delta["right_foot"],
        fraction=0.22,
    )

    positions[:, joint_index["stern"]] = 0.0
    positions[:, joint_index["head"]] = (
        ANCHORS["neck"] + rotate_vectors(delta["head"], REST_VECTORS["head"])
    )

    positions[:, joint_index["upper_arm_left"]] = (
        ANCHORS["left_shoulder"]
        + rotate_vectors(delta["upper_arm_left"], REST_VECTORS["upper_arm"])
    )
    positions[:, joint_index["left_hand"]] = (
        positions[:, joint_index["upper_arm_left"]]
        + rotate_vectors(lower_arm_left_delta, REST_VECTORS["forearm"])
    )

    positions[:, joint_index["upper_arm_right"]] = (
        ANCHORS["right_shoulder"]
        + rotate_vectors(delta["upper_arm_right"], REST_VECTORS["upper_arm"])
    )
    positions[:, joint_index["right_hand"]] = (
        positions[:, joint_index["upper_arm_right"]]
        + rotate_vectors(lower_arm_right_delta, REST_VECTORS["forearm"])
    )

    positions[:, joint_index["upper_leg_left"]] = (
        ANCHORS["left_hip"]
        + rotate_vectors(delta["upper_leg_left"], REST_VECTORS["upper_leg"])
    )
    positions[:, joint_index["left_foot"]] = (
        positions[:, joint_index["upper_leg_left"]]
        + rotate_vectors(lower_leg_left_delta, REST_VECTORS["lower_leg"])
    )

    positions[:, joint_index["upper_leg_right"]] = (
        ANCHORS["right_hip"]
        + rotate_vectors(delta["upper_leg_right"], REST_VECTORS["upper_leg"])
    )
    positions[:, joint_index["right_foot"]] = (
        positions[:, joint_index["upper_leg_right"]]
        + rotate_vectors(lower_leg_right_delta, REST_VECTORS["lower_leg"])
    )

    return regularize_relative_positions(
        raw_positions=positions,
        sensor_data_by_joint=sensor_data_by_joint,
        gyr_motion_scale=gyr_motion_scale,
    )


def enforce_side_constraint(
    point: np.ndarray,
    parent: np.ndarray,
    side_sign: float,
    desired_length: float,
    min_abs_y: float,
    max_abs_y: float,
    compression: float,
    min_drop: float,
    max_drop: float,
    min_x: float | None = None,
    max_x: float | None = None,
    motion_gate: np.ndarray | None = None,
    relax_min_x: float = 0.0,
    relax_max_x: float = 0.0,
) -> np.ndarray:
    raw_delta = point - parent
    smoothed_x = moving_average(raw_delta[:, [0]], window_size=9)[:, 0]
    vertical_drop = np.clip(np.abs(raw_delta[:, 2]), min_drop, max_drop)
    vertical_drop = np.minimum(vertical_drop, max(desired_length - 1e-4, 1e-4))

    requested_abs_y = np.abs(raw_delta[:, 1]) * compression
    requested_abs_y = np.clip(requested_abs_y, 0.0, max_abs_y)
    feasible_abs_y = np.sqrt(np.maximum(desired_length**2 - vertical_drop**2, 0.0))
    delta_abs_y = np.minimum(requested_abs_y, feasible_abs_y)
    soft_min_mask = (delta_abs_y < min_abs_y) & (np.abs(raw_delta[:, 1]) > 0.02)
    delta_abs_y[soft_min_mask] = np.minimum(min_abs_y, feasible_abs_y[soft_min_mask])

    dominant_sign_x = -1.0 if float(np.median(smoothed_x)) < 0.0 else 1.0
    delta_sign_x = np.where(smoothed_x < -1e-4, -1.0, np.where(smoothed_x > 1e-4, 1.0, dominant_sign_x))

    if min_x is not None or max_x is not None:
        if motion_gate is None:
            motion_gate = np.zeros_like(smoothed_x, dtype=np.float64)
        motion_gate = np.clip(motion_gate, 0.0, 1.0)

        effective_min_x = None if min_x is None else (min_x - relax_min_x * motion_gate)
        effective_max_x = None if max_x is None else (max_x + relax_max_x * motion_gate)
        positive_abs_x_limit = (
            np.full_like(smoothed_x, desired_length, dtype=np.float64)
            if effective_max_x is None
            else np.maximum(effective_max_x, 0.0)
        )
        negative_abs_x_limit = (
            np.full_like(smoothed_x, desired_length, dtype=np.float64)
            if effective_min_x is None
            else np.maximum(-effective_min_x, 0.0)
        )
        target_abs_x_limit = np.where(delta_sign_x >= 0.0, positive_abs_x_limit, negative_abs_x_limit)
        max_abs_x_from_drop = np.sqrt(np.maximum(desired_length**2 - vertical_drop**2, 0.0))
        target_abs_x_limit = np.minimum(target_abs_x_limit, max_abs_x_from_drop)

        remaining_x_sq = np.maximum(desired_length**2 - delta_abs_y**2 - vertical_drop**2, 0.0)
        delta_abs_x = np.sqrt(remaining_x_sq)
        over_limit_mask = delta_abs_x > (target_abs_x_limit + 1e-6)
        if np.any(over_limit_mask):
            required_abs_y_for_x_limit = np.sqrt(
                np.maximum(
                    desired_length**2
                    - vertical_drop[over_limit_mask] ** 2
                    - target_abs_x_limit[over_limit_mask] ** 2,
                    0.0,
                )
            )
            delta_abs_y[over_limit_mask] = np.maximum(
                delta_abs_y[over_limit_mask],
                required_abs_y_for_x_limit,
            )
            delta_abs_y[over_limit_mask] = np.minimum(
                delta_abs_y[over_limit_mask],
                feasible_abs_y[over_limit_mask],
            )

    remaining_x_sq = np.maximum(desired_length**2 - delta_abs_y**2 - vertical_drop**2, 0.0)
    delta_abs_x = np.sqrt(remaining_x_sq)

    delta = np.stack(
        [
            delta_sign_x * delta_abs_x,
            side_sign * delta_abs_y,
            -vertical_drop,
        ],
        axis=1,
    )
    return parent + delta


def smooth_segment_with_fixed_length(
    point: np.ndarray,
    parent: np.ndarray,
    desired_length: float,
    window_size: int,
) -> np.ndarray:
    smoothed_point = moving_average(point, window_size=window_size)
    direction = smoothed_point - parent
    fallback = point - parent
    direction = normalize_vector_rows(
        direction,
        normalize_vector_rows(fallback, np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    )
    return parent + desired_length * direction


def compute_stern_relative_angular_velocity(
    sensor_data_by_joint: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    stern_rotation_matrices = quaternion_to_rotation_matrices(sensor_data_by_joint["stern"]["quat"])
    stern_global_angular_velocity = np.einsum(
        "tij,tj->ti",
        stern_rotation_matrices,
        sensor_data_by_joint["stern"]["gyr"],
    )

    stern_relative_angular_velocity: dict[str, np.ndarray] = {}
    for joint_name, sensor_values in sensor_data_by_joint.items():
        joint_rotation_matrices = quaternion_to_rotation_matrices(sensor_values["quat"])
        joint_global_angular_velocity = np.einsum(
            "tij,tj->ti",
            joint_rotation_matrices,
            sensor_values["gyr"],
        )
        joint_relative_global_angular_velocity = (
            joint_global_angular_velocity - stern_global_angular_velocity
        )
        stern_relative_angular_velocity[joint_name] = moving_average(
            np.einsum(
                "tji,tj->ti",
                stern_rotation_matrices,
                joint_relative_global_angular_velocity,
            ),
            window_size=7,
        )

    return stern_relative_angular_velocity


def project_segment_with_fixed_length(
    point: np.ndarray,
    parent: np.ndarray,
    desired_length: float,
) -> np.ndarray:
    fallback_direction = normalize_vector_rows(
        point - parent,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    direction = normalize_vector_rows(point - parent, fallback_direction)
    return parent + desired_length * direction


def build_gyr_dynamic_gate(
    angular_velocity: np.ndarray,
    threshold: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.linalg.norm(angular_velocity, axis=1, keepdims=True)
    gate = np.tanh(np.maximum(magnitude - threshold, 0.0) / max(scale, 1e-6))
    pitch_signal = np.tanh(angular_velocity[:, 1] / max(scale, 1e-6))
    return gate[:, 0], pitch_signal


def build_motion_gate(
    angular_velocity: np.ndarray,
    threshold: float,
    scale: float,
) -> np.ndarray:
    magnitude = np.linalg.norm(angular_velocity, axis=1)
    return np.tanh(np.maximum(magnitude - threshold, 0.0) / max(scale, 1e-6))


def apply_gyr_dynamic_segment(
    reference_point: np.ndarray,
    corrected_point: np.ndarray,
    parent: np.ndarray,
    desired_length: float,
    angular_velocity: np.ndarray,
    blend_gain_x: float,
    blend_gain_z: float,
    pitch_gain_x: float,
    lift_gain_z: float,
    gate_threshold: float,
    gate_scale: float,
) -> np.ndarray:
    gate, pitch_signal = build_gyr_dynamic_gate(
        angular_velocity=angular_velocity,
        threshold=gate_threshold,
        scale=gate_scale,
    )
    reference_delta = reference_point - parent
    corrected_delta = corrected_point - parent
    enhanced_delta = corrected_delta.copy()

    reference_x_clip = min(max(desired_length * 1.05, 0.12), 0.22)
    reference_z_clip = min(max(desired_length * 0.75, 0.08), 0.14)
    reference_delta_x = np.clip(
        reference_delta[:, 0] - corrected_delta[:, 0],
        -reference_x_clip,
        reference_x_clip,
    )
    reference_delta_z = np.clip(
        reference_delta[:, 2] - corrected_delta[:, 2],
        -reference_z_clip,
        reference_z_clip,
    )
    enhanced_delta[:, 0] += blend_gain_x * gate * reference_delta_x + pitch_gain_x * gate * pitch_signal
    enhanced_delta[:, 2] += blend_gain_z * gate * reference_delta_z + lift_gain_z * gate * np.abs(pitch_signal)

    return project_segment_with_fixed_length(
        point=parent + enhanced_delta,
        parent=parent,
        desired_length=desired_length,
    )


def enhance_relative_positions_with_gyr(
    reference_positions: np.ndarray,
    corrected_positions: np.ndarray,
    sensor_data_by_joint: dict[str, dict[str, np.ndarray]],
    gyr_motion_scale: float,
) -> np.ndarray:
    if gyr_motion_scale <= 0.0:
        return corrected_positions

    joint_index = {name: idx for idx, name in enumerate(JOINT_ORDER)}
    stern_relative_angular_velocity = compute_stern_relative_angular_velocity(sensor_data_by_joint)
    enhanced = corrected_positions.copy()

    upper_arm_length = float(np.linalg.norm(REST_VECTORS["upper_arm"]))
    forearm_length = float(np.linalg.norm(REST_VECTORS["forearm"]))
    upper_leg_length = float(np.linalg.norm(REST_VECTORS["upper_leg"]))
    lower_leg_length = float(np.linalg.norm(REST_VECTORS["lower_leg"]))

    limb_specs = [
        (
            "upper_arm_left",
            "left_hand",
            ANCHORS["left_shoulder"],
            upper_arm_length,
            forearm_length,
            0.34,
            0.26,
            0.022,
            0.008,
            0.82,
            0.56,
            0.070,
            0.026,
            0.45,
            1.05,
            0.65,
            1.35,
        ),
        (
            "upper_arm_right",
            "right_hand",
            ANCHORS["right_shoulder"],
            upper_arm_length,
            forearm_length,
            0.34,
            0.26,
            0.022,
            0.008,
            0.82,
            0.56,
            0.070,
            0.026,
            0.45,
            1.05,
            0.65,
            1.35,
        ),
        (
            "upper_leg_left",
            "left_foot",
            ANCHORS["left_hip"],
            upper_leg_length,
            lower_leg_length,
            0.42,
            0.32,
            0.024,
            0.009,
            0.94,
            0.66,
            0.082,
            0.034,
            0.50,
            1.10,
            0.70,
            1.40,
        ),
        (
            "upper_leg_right",
            "right_foot",
            ANCHORS["right_hip"],
            upper_leg_length,
            lower_leg_length,
            0.42,
            0.32,
            0.024,
            0.009,
            0.94,
            0.66,
            0.082,
            0.034,
            0.50,
            1.10,
            0.70,
            1.40,
        ),
    ]

    for (
        upper_name,
        lower_name,
        anchor,
        upper_length,
        lower_length,
        upper_blend_x,
        upper_blend_z,
        upper_pitch_x,
        upper_lift_z,
        lower_blend_x,
        lower_blend_z,
        lower_pitch_x,
        lower_lift_z,
        upper_gate_threshold,
        upper_gate_scale,
        lower_gate_threshold,
        lower_gate_scale,
    ) in limb_specs:
        upper_parent = np.broadcast_to(anchor, enhanced[:, joint_index[upper_name], :].shape)
        upper_corrected = enhanced[:, joint_index[upper_name], :]
        upper_reference = reference_positions[:, joint_index[upper_name], :]
        upper_enhanced = apply_gyr_dynamic_segment(
            reference_point=upper_reference,
            corrected_point=upper_corrected,
            parent=upper_parent,
            desired_length=upper_length,
            angular_velocity=stern_relative_angular_velocity[upper_name],
            blend_gain_x=upper_blend_x * gyr_motion_scale,
            blend_gain_z=upper_blend_z * gyr_motion_scale,
            pitch_gain_x=upper_pitch_x * gyr_motion_scale,
            lift_gain_z=upper_lift_z * gyr_motion_scale,
            gate_threshold=upper_gate_threshold,
            gate_scale=upper_gate_scale,
        )

        lower_corrected = enhanced[:, joint_index[lower_name], :]
        lower_reference = reference_positions[:, joint_index[lower_name], :]
        lower_enhanced = apply_gyr_dynamic_segment(
            reference_point=lower_reference,
            corrected_point=lower_corrected,
            parent=upper_enhanced,
            desired_length=lower_length,
            angular_velocity=stern_relative_angular_velocity[lower_name],
            blend_gain_x=lower_blend_x * gyr_motion_scale,
            blend_gain_z=lower_blend_z * gyr_motion_scale,
            pitch_gain_x=lower_pitch_x * gyr_motion_scale,
            lift_gain_z=lower_lift_z * gyr_motion_scale,
            gate_threshold=lower_gate_threshold,
            gate_scale=lower_gate_scale,
        )

        enhanced[:, joint_index[upper_name]] = upper_enhanced
        enhanced[:, joint_index[lower_name]] = lower_enhanced

    return enhanced


def regularize_relative_positions(
    raw_positions: np.ndarray,
    sensor_data_by_joint: dict[str, dict[str, np.ndarray]],
    gyr_motion_scale: float,
) -> np.ndarray:
    positions = raw_positions.copy()
    stern_relative_angular_velocity = compute_stern_relative_angular_velocity(sensor_data_by_joint)
    for joint_index in range(1, positions.shape[1]):
        if JOINT_ORDER[joint_index] in {"left_hand", "right_hand", "left_foot", "right_foot"}:
            window_size = 5
        elif JOINT_ORDER[joint_index] in {"upper_arm_left", "upper_arm_right", "upper_leg_left", "upper_leg_right"}:
            window_size = 5
        else:
            window_size = 5
        positions[:, joint_index, :] = moving_average(positions[:, joint_index, :], window_size=window_size)

    joint_index = {name: idx for idx, name in enumerate(JOINT_ORDER)}
    corrected = np.zeros_like(positions)
    corrected[:, joint_index["stern"]] = 0.0

    head_length = float(np.linalg.norm(REST_VECTORS["head"]))
    head_direction = normalize_vector_rows(
        positions[:, joint_index["head"]] - ANCHORS["neck"],
        np.tile(normalize_vector(REST_VECTORS["head"], np.array([1.0, 0.0, 0.0], dtype=np.float64)), (positions.shape[0], 1)),
    )
    corrected[:, joint_index["head"]] = ANCHORS["neck"] + head_length * head_direction

    upper_arm_length = float(np.linalg.norm(REST_VECTORS["upper_arm"]))
    forearm_length = float(np.linalg.norm(REST_VECTORS["forearm"]))
    upper_leg_length = float(np.linalg.norm(REST_VECTORS["upper_leg"]))
    lower_leg_length = float(np.linalg.norm(REST_VECTORS["lower_leg"]))

    limb_specs = [
        (
            "upper_arm_left",
            "left_hand",
            ANCHORS["left_shoulder"],
            +1.0,
            upper_arm_length,
            forearm_length,
            (-0.04, 0.14),
            (-0.05, 0.14),
            (0.06, 0.06),
            (0.08, 0.08),
            (0.25, 0.85),
            (0.35, 1.00),
        ),
        (
            "upper_arm_right",
            "right_hand",
            ANCHORS["right_shoulder"],
            -1.0,
            upper_arm_length,
            forearm_length,
            (-0.04, 0.14),
            (-0.05, 0.14),
            (0.06, 0.06),
            (0.08, 0.08),
            (0.25, 0.85),
            (0.35, 1.00),
        ),
        (
            "upper_leg_left",
            "left_foot",
            ANCHORS["left_hip"],
            +1.0,
            upper_leg_length,
            lower_leg_length,
            (-0.14, 0.06),
            (-0.06, 0.14),
            (0.08, 0.10),
            (0.10, 0.12),
            (0.30, 0.95),
            (0.45, 1.10),
        ),
        (
            "upper_leg_right",
            "right_foot",
            ANCHORS["right_hip"],
            -1.0,
            upper_leg_length,
            lower_leg_length,
            (-0.14, 0.06),
            (-0.06, 0.14),
            (0.08, 0.10),
            (0.10, 0.12),
            (0.30, 0.95),
            (0.45, 1.10),
        ),
    ]

    for (
        upper_name,
        lower_name,
        anchor,
        side_sign,
        upper_length,
        lower_length,
        upper_x_bounds,
        lower_x_bounds,
        upper_x_relax,
        lower_x_relax,
        upper_motion_gate_params,
        lower_motion_gate_params,
    ) in limb_specs:
        upper_raw = positions[:, joint_index[upper_name], :]
        lower_raw = positions[:, joint_index[lower_name], :]
        upper_motion_gate = build_motion_gate(
            angular_velocity=stern_relative_angular_velocity[upper_name],
            threshold=upper_motion_gate_params[0],
            scale=upper_motion_gate_params[1],
        )
        lower_motion_gate = build_motion_gate(
            angular_velocity=stern_relative_angular_velocity[lower_name],
            threshold=lower_motion_gate_params[0],
            scale=lower_motion_gate_params[1],
        )

        if "arm" in upper_name:
            upper_abs_y = (0.02, 0.10)
            lower_abs_y = (0.015, 0.12)
            compression = 0.50
            upper_drop = (0.08, 0.18)
            lower_drop = (0.10, 0.19)
        else:
            upper_abs_y = (0.02, 0.11)
            lower_abs_y = (0.015, 0.13)
            compression = 0.42
            upper_drop = (0.08, 0.20)
            lower_drop = (0.09, 0.20)

        upper_corrected = enforce_side_constraint(
            point=upper_raw,
            parent=np.broadcast_to(anchor, upper_raw.shape),
            side_sign=side_sign,
            desired_length=upper_length,
            min_abs_y=upper_abs_y[0],
            max_abs_y=upper_abs_y[1],
            compression=compression,
            min_drop=upper_drop[0],
            max_drop=upper_drop[1],
            min_x=upper_x_bounds[0],
            max_x=upper_x_bounds[1],
            motion_gate=upper_motion_gate,
            relax_min_x=upper_x_relax[0],
            relax_max_x=upper_x_relax[1],
        )
        lower_corrected = enforce_side_constraint(
            point=lower_raw,
            parent=upper_corrected,
            side_sign=side_sign,
            desired_length=lower_length,
            min_abs_y=lower_abs_y[0],
            max_abs_y=lower_abs_y[1],
            compression=compression,
            min_drop=lower_drop[0],
            max_drop=lower_drop[1],
            min_x=lower_x_bounds[0],
            max_x=lower_x_bounds[1],
            motion_gate=lower_motion_gate,
            relax_min_x=lower_x_relax[0],
            relax_max_x=lower_x_relax[1],
        )

        upper_corrected = smooth_segment_with_fixed_length(
            point=upper_corrected,
            parent=np.broadcast_to(anchor, upper_corrected.shape),
            desired_length=upper_length,
            window_size=5,
        )
        lower_corrected = smooth_segment_with_fixed_length(
            point=lower_corrected,
            parent=upper_corrected,
            desired_length=lower_length,
            window_size=7,
        )

        corrected[:, joint_index[upper_name]] = upper_corrected
        corrected[:, joint_index[lower_name]] = lower_corrected

    return enhance_relative_positions_with_gyr(
        reference_positions=positions,
        corrected_positions=corrected,
        sensor_data_by_joint=sensor_data_by_joint,
        gyr_motion_scale=gyr_motion_scale,
    )


def build_root_delta_quaternions(stern_quat: np.ndarray) -> np.ndarray:
    first_inverse = quaternion_inverse(stern_quat[0:1])
    return normalize_quaternions(
        quaternion_multiply(stern_quat, np.broadcast_to(first_inverse, stern_quat.shape))
    )


def limit_heading_step(
    *,
    previous_forward: np.ndarray,
    target_forward: np.ndarray,
    max_step_radians: float | None,
) -> np.ndarray:
    if max_step_radians is None or max_step_radians <= 0.0:
        return target_forward

    dot = float(np.clip(np.dot(previous_forward, target_forward), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle <= max_step_radians:
        return target_forward

    cross_z = previous_forward[0] * target_forward[1] - previous_forward[1] * target_forward[0]
    sign = 1.0 if cross_z >= 0.0 else -1.0
    step = sign * max_step_radians
    cosine = float(np.cos(step))
    sine = float(np.sin(step))
    clipped_forward = np.array(
        [
            cosine * previous_forward[0] - sine * previous_forward[1],
            sine * previous_forward[0] + cosine * previous_forward[1],
            0.0,
        ],
        dtype=np.float64,
    )
    return normalize_vector(clipped_forward, previous_forward)


def build_heading_rotation_matrices(
    stern_quat: np.ndarray,
    max_heading_step_deg: float | None = DEFAULT_MAX_HEADING_STEP_DEG,
) -> np.ndarray:
    full_rotation_matrices = quaternion_to_rotation_matrices(build_root_delta_quaternions(stern_quat))
    forward_vectors = np.einsum(
        "tij,j->ti",
        full_rotation_matrices,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    forward_vectors[:, 2] = 0.0

    heading_rotation_matrices = np.zeros_like(full_rotation_matrices)
    previous_forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    max_step_radians = (
        None
        if max_heading_step_deg is None or max_heading_step_deg <= 0.0
        else float(np.deg2rad(max_heading_step_deg))
    )

    for frame_index in range(forward_vectors.shape[0]):
        forward = normalize_vector(forward_vectors[frame_index], previous_forward)
        forward = limit_heading_step(
            previous_forward=previous_forward,
            target_forward=forward,
            max_step_radians=max_step_radians,
        )
        previous_forward = forward
        lateral = normalize_vector(
            np.cross(GLOBAL_UP, forward),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
        vertical = normalize_vector(np.cross(forward, lateral), GLOBAL_UP)
        heading_rotation_matrices[frame_index] = np.column_stack([forward, lateral, vertical])

    return heading_rotation_matrices


def build_z_rotation_matrix(angle_radians: float) -> np.ndarray:
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def align_heading_to_motion(
    rotation_matrices: np.ndarray,
    root_translation: np.ndarray,
) -> tuple[np.ndarray, float]:
    velocity = np.diff(root_translation[:, :2], axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    if speed.size == 0:
        return rotation_matrices, 0.0

    valid = speed > max(float(np.percentile(speed, 65)), 1e-5)
    if valid.sum() < 10:
        return rotation_matrices, 0.0

    heading = rotation_matrices[:-1, :2, 0]
    heading = heading / np.maximum(np.linalg.norm(heading, axis=1, keepdims=True), 1e-8)
    velocity_direction = velocity / np.maximum(speed[:, None], 1e-8)

    dots = np.clip((heading[valid] * velocity_direction[valid]).sum(axis=1), -1.0, 1.0)
    crosses = heading[valid, 0] * velocity_direction[valid, 1] - heading[valid, 1] * velocity_direction[valid, 0]
    angle_offset = float(np.median(np.arctan2(crosses, dots)))

    rotation_offset = build_z_rotation_matrix(angle_offset)
    aligned_rotation_matrices = np.einsum("ij,tjk->tik", rotation_offset, rotation_matrices)
    return aligned_rotation_matrices, angle_offset


def estimate_root_translation(
    stern_freeacc: np.ndarray,
    rotation_matrices: np.ndarray,
    translation_scale: float,
    vertical_motion_scale: float,
    sample_rate_hz: float,
) -> np.ndarray:
    dt = 1.0 / sample_rate_hz
    acceleration = moving_average(stern_freeacc, window_size=5)
    acceleration = np.clip(acceleration, -8.0, 8.0)
    forward_vectors = rotation_matrices[:, :, 0].copy()
    forward_vectors[:, 2] = 0.0
    forward_vectors = normalize_vector_rows(
        forward_vectors,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )

    forward_acceleration = np.sum(
        acceleration[:, :2] * forward_vectors[:, :2],
        axis=1,
    )
    vertical_acceleration = acceleration[:, 2] * vertical_motion_scale

    forward_velocity = np.zeros(acceleration.shape[0], dtype=np.float64)
    vertical_velocity = np.zeros(acceleration.shape[0], dtype=np.float64)
    translation = np.zeros_like(acceleration)

    forward_velocity_decay = 0.94
    vertical_velocity_decay = 0.90
    for frame_index in range(1, acceleration.shape[0]):
        forward_velocity[frame_index] = (
            forward_velocity[frame_index - 1] * forward_velocity_decay
            + forward_acceleration[frame_index] * dt
        )
        vertical_velocity[frame_index] = (
            vertical_velocity[frame_index - 1] * vertical_velocity_decay
            + vertical_acceleration[frame_index] * dt
        )
        forward_speed = max(forward_velocity[frame_index], 0.0)
        translation[frame_index, :2] = (
            translation[frame_index - 1, :2]
            + forward_speed * forward_vectors[frame_index, :2] * dt
        )
        translation[frame_index, 2] = (
            translation[frame_index - 1, 2]
            + vertical_velocity[frame_index] * dt
        )

    translation *= translation_scale
    translation -= translation[0]
    translation = moving_average(translation, window_size=5)
    return translation


def build_scene_motion(
    relative_positions: np.ndarray,
    sensor_data_by_joint: dict[str, dict[str, np.ndarray]],
    root_motion_mode: str,
    translation_scale: float,
    vertical_motion_scale: float,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    stern_quat = sensor_data_by_joint["stern"]["quat"]
    rotation_matrices = build_heading_rotation_matrices(stern_quat)

    if root_motion_mode == "approximate":
        root_translation = estimate_root_translation(
            stern_freeacc=sensor_data_by_joint["stern"]["freeacc"],
            rotation_matrices=rotation_matrices,
            translation_scale=translation_scale,
            vertical_motion_scale=vertical_motion_scale,
            sample_rate_hz=sample_rate_hz,
        )
    else:
        root_translation = np.zeros((relative_positions.shape[0], 3), dtype=np.float64)

    _, heading_offset = align_heading_to_motion(
        rotation_matrices=rotation_matrices,
        root_translation=root_translation,
    )

    scene_positions = transform_dynamic_points(
        rotation_matrices=rotation_matrices,
        translation=root_translation,
        local_points=relative_positions,
    )

    local_body_point_names = list(LOCAL_BODY_POINTS.keys())
    transformed_body_points = transform_static_points(
        rotation_matrices=rotation_matrices,
        translation=root_translation,
        local_points=np.asarray([LOCAL_BODY_POINTS[name] for name in local_body_point_names], dtype=np.float64),
    )
    body_points_by_name = {
        name: transformed_body_points[:, index]
        for index, name in enumerate(local_body_point_names)
    }

    transformed_back_line = transform_static_points(
        rotation_matrices=rotation_matrices,
        translation=root_translation,
        local_points=LOCAL_BACK_LINE,
    )
    transformed_belly_line = transform_static_points(
        rotation_matrices=rotation_matrices,
        translation=root_translation,
        local_points=LOCAL_BELLY_LINE,
    )

    return (
        scene_positions,
        root_translation,
        rotation_matrices,
        body_points_by_name,
        transformed_back_line,
        transformed_belly_line,
        heading_offset,
    )


def select_frames(
    packet_counter: np.ndarray,
    positions: np.ndarray,
    start_frame: int,
    num_frames: int | None,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if start_frame >= len(packet_counter):
        raise ValueError("--start-frame is outside the sequence")

    if num_frames is None:
        stop_frame = len(packet_counter)
    else:
        if num_frames <= 0:
            raise ValueError("--num-frames must be positive when provided")
        stop_frame = min(len(packet_counter), start_frame + num_frames)

    indices = np.arange(start_frame, stop_frame, stride, dtype=np.int64)
    return indices, packet_counter[indices], positions[indices]


def find_most_active_window(
    positions: np.ndarray,
    window_frames: int,
    search_step: int,
) -> tuple[int, int, float]:
    if window_frames <= 1:
        return 0, min(window_frames, positions.shape[0]), 0.0
    if search_step <= 0:
        raise ValueError("--search-step must be positive")

    frame_motion = np.linalg.norm(np.diff(positions, axis=0), axis=2).mean(axis=1)
    total_frames = positions.shape[0]
    if total_frames <= window_frames:
        score = float(frame_motion.mean()) if frame_motion.size else 0.0
        return 0, total_frames, score

    best_start = 0
    best_score = -np.inf
    last_possible_start = total_frames - window_frames

    for start in range(0, last_possible_start + 1, search_step):
        stop = start + window_frames
        score = float(frame_motion[start : stop - 1].mean())
        if score > best_score:
            best_start = start
            best_score = score

    if last_possible_start % search_step != 0:
        start = last_possible_start
        stop = start + window_frames
        score = float(frame_motion[start : stop - 1].mean())
        if score > best_score:
            best_start = start
            best_score = score

    return best_start, best_start + window_frames, best_score


def save_coordinates(
    output_path: Path,
    frame_indices: np.ndarray,
    packet_counter: np.ndarray,
    positions: np.ndarray,
    relative_positions: np.ndarray | None = None,
    root_translation: np.ndarray | None = None,
    root_heading_6d: np.ndarray | None = None,
    is_interpolated: np.ndarray | None = None,
    frame_action_labels: np.ndarray | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".npz":
        payload = {
            "frame_index": frame_indices,
            "packet_counter": packet_counter,
            "joint_names": np.asarray(JOINT_ORDER),
            "positions": positions.astype(np.float32),
        }
        if relative_positions is not None:
            payload["relative_positions"] = relative_positions.astype(np.float32)
        if root_translation is not None:
            payload["root_translation"] = root_translation.astype(np.float32)
        if root_heading_6d is not None:
            payload["root_heading_6d"] = root_heading_6d.astype(np.float32)
        if is_interpolated is not None:
            payload["is_interpolated"] = is_interpolated.astype(bool)
        if frame_action_labels is not None:
            payload["frame_action_labels"] = frame_action_labels.astype(object)
        np.savez_compressed(output_path, **payload)
        return

    if output_path.suffix.lower() == ".csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "packet_counter", "joint", "x", "y", "z"])
            for local_frame, frame_index in enumerate(frame_indices):
                for joint_index, joint_name in enumerate(JOINT_ORDER):
                    x, y, z = positions[local_frame, joint_index]
                    writer.writerow(
                        [int(frame_index), int(packet_counter[local_frame]), joint_name, x, y, z]
                    )
        return

    raise ValueError("Coordinate output must end with .npz or .csv")


def rotation_matrices_to_rot6d(rotation_matrices: np.ndarray) -> np.ndarray:
    return np.concatenate([rotation_matrices[:, :, 0], rotation_matrices[:, :, 1]], axis=1)


def stack_joint_is_interpolated(sensor_data_by_joint: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    return np.stack(
        [sensor_data_by_joint[joint_name]["is_interpolated"] for joint_name in JOINT_ORDER],
        axis=1,
    )


def compute_pose_audit(
    relative_positions: np.ndarray,
    is_interpolated: np.ndarray,
) -> dict[str, float | str]:
    joint_index = {name: idx for idx, name in enumerate(JOINT_ORDER)}
    finite_ratio = float(np.isfinite(relative_positions).mean())
    interp_ratio = float(is_interpolated.mean())
    joint_interp_ratio = is_interpolated.mean(axis=0) if is_interpolated.size else np.zeros((len(JOINT_ORDER),))
    worst_interp_joint_index = int(np.argmax(joint_interp_ratio)) if joint_interp_ratio.size else 0
    max_joint_interp_ratio = float(joint_interp_ratio[worst_interp_joint_index]) if joint_interp_ratio.size else 0.0
    worst_interp_joint = JOINT_ORDER[worst_interp_joint_index]

    left_right_checks = [
        relative_positions[:, joint_index["upper_arm_left"], 1] > 0.0,
        relative_positions[:, joint_index["left_hand"], 1] > 0.0,
        relative_positions[:, joint_index["upper_leg_left"], 1] > 0.0,
        relative_positions[:, joint_index["left_foot"], 1] > 0.0,
        relative_positions[:, joint_index["upper_arm_right"], 1] < 0.0,
        relative_positions[:, joint_index["right_hand"], 1] < 0.0,
        relative_positions[:, joint_index["upper_leg_right"], 1] < 0.0,
        relative_positions[:, joint_index["right_foot"], 1] < 0.0,
    ]
    left_right_sign_consistency = float(np.mean(np.concatenate(left_right_checks, axis=0)))

    distal_below_parent_checks = [
        relative_positions[:, joint_index["left_hand"], 2] <= relative_positions[:, joint_index["upper_arm_left"], 2],
        relative_positions[:, joint_index["right_hand"], 2] <= relative_positions[:, joint_index["upper_arm_right"], 2],
        relative_positions[:, joint_index["left_foot"], 2] <= relative_positions[:, joint_index["upper_leg_left"], 2],
        relative_positions[:, joint_index["right_foot"], 2] <= relative_positions[:, joint_index["upper_leg_right"], 2],
    ]
    distal_below_parent_consistency = float(np.mean(np.concatenate(distal_below_parent_checks, axis=0)))

    if finite_ratio < 1.0 or interp_ratio > 0.20 or max_joint_interp_ratio > 0.20:
        preview_status = "fail"
    elif left_right_sign_consistency < 0.95 or distal_below_parent_consistency < 0.90:
        preview_status = "review"
    else:
        preview_status = "pass"

    return {
        "finite_ratio": finite_ratio,
        "interp_ratio": interp_ratio,
        "max_joint_interp_ratio": max_joint_interp_ratio,
        "worst_interp_joint": worst_interp_joint,
        "left_right_sign_consistency": left_right_sign_consistency,
        "distal_below_parent_consistency": distal_below_parent_consistency,
        "preview_status": preview_status,
    }


def infer_segment_metadata(data_dir: Path) -> dict[str, str]:
    participant = ""
    segment_id = ""
    if data_dir.parent.name and data_dir.parent.name != data_dir.anchor:
        participant = data_dir.parent.name
    if data_dir.name.isdigit():
        segment_id = data_dir.name
    return {
        "source_dir": str(data_dir),
        "participant": participant,
        "segment_id": segment_id,
    }


def save_preview_audit(
    output_path: Path,
    metadata: dict[str, str],
    frame_indices: np.ndarray,
    packet_counter: np.ndarray,
    sample_rate_hz: float,
    audit_metrics: dict[str, float | str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_dir",
        "participant",
        "segment_id",
        "sample_rate_hz",
        "frame_start",
        "frame_end",
        "packet_start",
        "packet_end",
        "num_frames",
        "finite_ratio",
        "interp_ratio",
        "max_joint_interp_ratio",
        "worst_interp_joint",
        "left_right_sign_consistency",
        "distal_below_parent_consistency",
        "preview_status",
    ]
    row = {
        **metadata,
        "sample_rate_hz": float(sample_rate_hz),
        "frame_start": int(frame_indices[0]),
        "frame_end": int(frame_indices[-1]),
        "packet_start": int(packet_counter[0]),
        "packet_end": int(packet_counter[-1]),
        "num_frames": int(frame_indices.shape[0]),
        **audit_metrics,
    }

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def set_equal_axes(ax: plt.Axes, coords: np.ndarray) -> None:
    mins = coords.min(axis=(0, 1))
    maxs = coords.max(axis=(0, 1))
    center = (mins + maxs) / 2.0
    span = max(float((maxs - mins).max()), 0.6)
    half = span / 2.0 + 0.14

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def transform_faces(
    local_faces: list[list[np.ndarray]],
    rotation_matrix: np.ndarray,
    translation: np.ndarray,
) -> list[list[np.ndarray]]:
    transformed_faces: list[list[np.ndarray]] = []
    for face in local_faces:
        local_face = np.asarray(face, dtype=np.float64)
        transformed_face = (rotation_matrix @ local_face.T).T + translation
        transformed_faces.append([vertex for vertex in transformed_face])
    return transformed_faces


def normalize_label(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


def coerce_label_sequence(values: np.ndarray | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    flattened = np.asarray(values).reshape(-1)
    return tuple(str(value) for value in flattened.tolist())


def to_dense_float_array(value: object) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    elif hasattr(value, "A"):
        value = value.A
    elif hasattr(value, "r"):
        value = value.r
    return np.asarray(value, dtype=np.float64)


def control_points_from_regressor(vertices: np.ndarray, regressor: object) -> np.ndarray:
    regressor_dense = to_dense_float_array(regressor)
    return regressor_dense @ vertices


def load_template_mesh_from_npz(model_path: Path) -> TemplateMeshModel:
    payload = np.load(model_path, allow_pickle=True)

    vertex_key = "vertices" if "vertices" in payload else "v_template"
    face_key = "faces" if "faces" in payload else "f"

    if vertex_key not in payload or face_key not in payload:
        raise ValueError(
            f"Template NPZ must contain vertices/v_template and faces/f arrays: {model_path}"
        )

    if "control_points" in payload:
        control_points = np.asarray(payload["control_points"], dtype=np.float64)
    elif "joint_positions" in payload:
        control_points = np.asarray(payload["joint_positions"], dtype=np.float64)
    elif "joints" in payload:
        control_points = np.asarray(payload["joints"], dtype=np.float64)
    elif "J" in payload:
        control_points = np.asarray(payload["J"], dtype=np.float64)
    elif "J_regressor" in payload:
        control_points = control_points_from_regressor(
            vertices=np.asarray(payload[vertex_key], dtype=np.float64),
            regressor=payload["J_regressor"],
        )
    else:
        raise ValueError(
            f"Template NPZ must contain control_points/joint_positions/joints/J/J_regressor: {model_path}"
        )

    if "control_labels" in payload:
        control_labels = coerce_label_sequence(payload["control_labels"])
    elif "joint_names" in payload:
        control_labels = coerce_label_sequence(payload["joint_names"])
    elif "control_names" in payload:
        control_labels = coerce_label_sequence(payload["control_names"])
    else:
        control_labels = tuple(str(index) for index in range(control_points.shape[0]))

    if len(control_labels) != control_points.shape[0]:
        raise ValueError(
            f"Control label count ({len(control_labels)}) does not match control point count "
            f"({control_points.shape[0]}) in {model_path}"
        )

    return TemplateMeshModel(
        vertices=np.asarray(payload[vertex_key], dtype=np.float64),
        faces=np.asarray(payload[face_key], dtype=np.int32),
        control_points=control_points,
        control_labels=control_labels,
        source_path=model_path,
    )


def load_template_mesh_from_pkl(model_path: Path) -> TemplateMeshModel:
    with model_path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported template PKL structure in {model_path}")

    if "v_template" not in payload or "f" not in payload:
        raise ValueError(f"Template PKL must contain v_template and f: {model_path}")

    vertices = to_dense_float_array(payload["v_template"])
    faces = np.asarray(payload["f"], dtype=np.int32)

    if "joint_positions" in payload:
        control_points = to_dense_float_array(payload["joint_positions"])
    elif "joints" in payload:
        control_points = to_dense_float_array(payload["joints"])
    elif "J" in payload:
        control_points = to_dense_float_array(payload["J"])
    elif "J_regressor" in payload:
        control_points = control_points_from_regressor(vertices=vertices, regressor=payload["J_regressor"])
    else:
        raise ValueError(
            f"Template PKL must contain joint_positions/joints/J/J_regressor in {model_path}"
        )

    if "joint_names" in payload:
        control_labels = coerce_label_sequence(payload["joint_names"])
    elif "control_labels" in payload:
        control_labels = coerce_label_sequence(payload["control_labels"])
    else:
        control_labels = tuple(str(index) for index in range(control_points.shape[0]))

    if len(control_labels) != control_points.shape[0]:
        raise ValueError(
            f"Control label count ({len(control_labels)}) does not match control point count "
            f"({control_points.shape[0]}) in {model_path}"
        )

    return TemplateMeshModel(
        vertices=vertices,
        faces=faces,
        control_points=control_points,
        control_labels=control_labels,
        source_path=model_path,
    )


def load_template_mesh_model(model_path: Path) -> TemplateMeshModel:
    if not model_path.exists():
        raise FileNotFoundError(f"Template mesh file not found: {model_path}")

    suffix = model_path.suffix.lower()
    if suffix == ".npz":
        return load_template_mesh_from_npz(model_path)
    if suffix == ".pkl":
        return load_template_mesh_from_pkl(model_path)
    raise ValueError("Template mesh must end with .npz or .pkl")


def load_smal_mapping(mapping_path: Path | None) -> dict[str, str]:
    if mapping_path is None:
        return {}

    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and all(isinstance(value, str) for value in payload.values()):
        return {str(key): str(value) for key, value in payload.items()}

    for key in ("mapping", "joint_map", "control_map"):
        value = payload.get(key)
        if isinstance(value, dict) and all(isinstance(item, str) for item in value.values()):
            return {str(map_key): str(map_value) for map_key, map_value in value.items()}

    raise ValueError(
        f"Mapping JSON must be a flat object or contain mapping/joint_map/control_map: {mapping_path}"
    )


def source_control_points_for_frame(
    current: np.ndarray,
    current_body_points: dict[str, np.ndarray],
    joint_index: dict[str, int],
) -> dict[str, np.ndarray]:
    source_points = {
        joint_name: current[index]
        for joint_name, index in joint_index.items()
    }
    source_points.update(current_body_points)
    return source_points


def auto_match_source_label(model_label: str) -> str | None:
    normalized = normalize_label(model_label)
    for source_label, aliases in DEFAULT_SMAL_SOURCE_ALIASES.items():
        normalized_aliases = {normalize_label(alias) for alias in aliases}
        if normalized in normalized_aliases:
            return source_label
    return None


def bind_template_mesh_model(
    template_mesh: TemplateMeshModel,
    mapping: dict[str, str],
) -> BoundTemplateMeshModel:
    bound_control_points: list[np.ndarray] = []
    bound_control_labels: list[str] = []
    source_labels: list[str] = []

    for index, (control_label, control_point) in enumerate(
        zip(template_mesh.control_labels, template_mesh.control_points)
    ):
        explicit_source = mapping.get(control_label, mapping.get(str(index)))
        source_label = explicit_source if explicit_source is not None else auto_match_source_label(control_label)
        if source_label is None:
            continue

        bound_control_points.append(control_point)
        bound_control_labels.append(control_label)
        source_labels.append(source_label)

    if len(bound_control_points) < 4:
        raise ValueError(
            "Unable to bind enough control points for SMAL deformation. "
            "Provide --smal-mapping with at least 4 matched controls."
        )

    return BoundTemplateMeshModel(
        vertices=template_mesh.vertices,
        faces=template_mesh.faces,
        control_points=np.asarray(bound_control_points, dtype=np.float64),
        control_labels=tuple(bound_control_labels),
        source_labels=tuple(source_labels),
        source_path=template_mesh.source_path,
    )


def fit_similarity_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    centered_source = source_points - source_center
    centered_target = target_points - target_center

    source_variance = float(np.sum(centered_source**2))
    if source_variance < 1e-10:
        return 1.0, np.eye(3, dtype=np.float64), target_center - source_center

    covariance = centered_source.T @ centered_target
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T

    scale = float(singular_values.sum() / source_variance)
    translation = target_center - scale * (source_center @ rotation.T)
    return scale, rotation, translation


def apply_similarity_transform(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    dense_points = np.asarray(points, dtype=np.float64)
    dense_rotation = np.asarray(rotation, dtype=np.float64)
    dense_translation = np.asarray(translation, dtype=np.float64)
    rotated = np.einsum("...i,ji->...j", dense_points, dense_rotation, optimize=True)
    return scale * rotated + dense_translation


def apply_inverse_distance_residuals(
    vertices: np.ndarray,
    control_points: np.ndarray,
    residuals: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    if control_points.shape[0] == 0:
        return vertices

    distances = np.linalg.norm(vertices[:, None, :] - control_points[None, :, :], axis=2)
    exact_mask = distances < 1e-8
    safe_distances = np.maximum(distances, 1e-8)
    weights = 1.0 / np.power(safe_distances, power)
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    deformed = vertices + np.einsum("vc,cj->vj", weights, residuals, optimize=True)

    has_exact = exact_mask.any(axis=1)
    if np.any(has_exact):
        exact_indices = np.argmax(exact_mask[has_exact], axis=1)
        deformed[has_exact] = vertices[has_exact] + residuals[exact_indices]
    return deformed


def build_template_mesh_frame(
    template_mesh: BoundTemplateMeshModel,
    current: np.ndarray,
    current_body_points: dict[str, np.ndarray],
    joint_index: dict[str, int],
) -> np.ndarray:
    current_source_points = source_control_points_for_frame(
        current=current,
        current_body_points=current_body_points,
        joint_index=joint_index,
    )
    target_controls = np.asarray(
        [current_source_points[source_label] for source_label in template_mesh.source_labels],
        dtype=np.float64,
    )

    scale, rotation, translation = fit_similarity_transform(
        source_points=template_mesh.control_points,
        target_points=target_controls,
    )
    transformed_vertices = apply_similarity_transform(
        points=template_mesh.vertices,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    transformed_controls = apply_similarity_transform(
        points=template_mesh.control_points,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    residuals = target_controls - transformed_controls
    return apply_inverse_distance_residuals(
        vertices=transformed_vertices,
        control_points=transformed_controls,
        residuals=residuals,
    )


def build_template_mesh_sequence(
    template_mesh: BoundTemplateMeshModel,
    positions: np.ndarray,
    body_points_by_name: dict[str, np.ndarray],
) -> np.ndarray:
    joint_index = {name: idx for idx, name in enumerate(JOINT_ORDER)}
    deformed_vertices = np.zeros(
        (positions.shape[0], template_mesh.vertices.shape[0], 3),
        dtype=np.float64,
    )
    progress = ProgressPrinter("Deforming template mesh", positions.shape[0])

    for frame_id in range(positions.shape[0]):
        current = positions[frame_id]
        current_body_points = {
            name: point_series[frame_id]
            for name, point_series in body_points_by_name.items()
        }
        deformed_vertices[frame_id] = build_template_mesh_frame(
            template_mesh=template_mesh,
            current=current,
            current_body_points=current_body_points,
            joint_index=joint_index,
        )
        progress.update(frame_id + 1)

    return deformed_vertices


class _FakeChumpyCh(object):
    def __setstate__(self, state: object) -> None:
        self.__dict__["_state"] = state

    @property
    def r(self) -> object:
        state = self.__dict__.get("_state")
        if isinstance(state, dict) and "x" in state:
            return state["x"]
        return state


def load_pickle_with_fake_chumpy(pickle_path: Path) -> object:
    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")
    chumpy_ch_module.Ch = _FakeChumpyCh
    chumpy_module.ch = chumpy_ch_module

    previous_chumpy = sys.modules.get("chumpy")
    previous_chumpy_ch = sys.modules.get("chumpy.ch")
    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Please import `csc_matrix` from the `scipy.sparse` namespace.*",
                category=DeprecationWarning,
            )
            with pickle_path.open("rb") as handle:
                return pickle.load(handle, encoding="latin1")
    finally:
        if previous_chumpy is None:
            sys.modules.pop("chumpy", None)
        else:
            sys.modules["chumpy"] = previous_chumpy
        if previous_chumpy_ch is None:
            sys.modules.pop("chumpy.ch", None)
        else:
            sys.modules["chumpy.ch"] = previous_chumpy_ch


def resolve_smal_data_path(model_path: Path, smal_data_path: Path | None) -> Path:
    if smal_data_path is not None:
        return smal_data_path
    sibling = model_path.with_name("smal_CVPR2017_data.pkl")
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        "SMAL data pkl was not provided and sibling smal_CVPR2017_data.pkl was not found"
    )


def is_smal_preset_payload(payload: object) -> bool:
    return isinstance(payload, dict) and {"beta", "pose", "trans"}.issubset(payload.keys())


def load_smal_preset(preset_path: Path) -> SMALPreset:
    payload = load_pickle_with_fake_chumpy(preset_path)
    if not is_smal_preset_payload(payload):
        raise ValueError(f"Unsupported SMAL preset payload in {preset_path}")

    return SMALPreset(
        beta=np.asarray(payload["beta"].r, dtype=np.float64),
        pose=np.asarray(payload["pose"].r, dtype=np.float64),
        trans=np.asarray(payload["trans"].r, dtype=np.float64),
        source_path=preset_path,
    )


def resolve_official_smal_model_path(model_or_preset_path: Path) -> Path:
    payload = load_pickle_with_fake_chumpy(model_or_preset_path)
    if isinstance(payload, dict) and "v_template" in payload:
        return model_or_preset_path
    if is_smal_preset_payload(payload):
        sibling = model_or_preset_path.with_name("smal_CVPR2017.pkl")
        if sibling.exists():
            return sibling
        raise FileNotFoundError(
            f"{model_or_preset_path} is a SMAL preset, but sibling smal_CVPR2017.pkl was not found"
        )
    raise ValueError(f"Unsupported SMAL file: {model_or_preset_path}")


def load_official_smal_model(
    model_path: Path,
    data_path: Path,
    family_index: int,
    betas_override: np.ndarray | None = None,
) -> OfficialSMALModel:
    model_payload = load_pickle_with_fake_chumpy(model_path)
    if not isinstance(model_payload, dict):
        raise ValueError(f"Unsupported official SMAL payload in {model_path}")

    data_payload = load_pickle_with_fake_chumpy(data_path)
    if not isinstance(data_payload, dict) or "cluster_means" not in data_payload:
        raise ValueError(f"SMAL data pkl must contain cluster_means: {data_path}")

    cluster_means = np.asarray(data_payload["cluster_means"], dtype=np.float64)
    if family_index < 0 or family_index >= cluster_means.shape[0]:
        raise ValueError(
            f"--smal-family-index {family_index} is out of range for cluster_means with "
            f"{cluster_means.shape[0]} entries"
        )

    vertices_template = np.asarray(model_payload["v_template"], dtype=np.float64)
    faces = np.asarray(model_payload["f"], dtype=np.int32)
    weights = np.asarray(model_payload["weights"], dtype=np.float64)
    posedirs = np.asarray(model_payload["posedirs"], dtype=np.float64)
    shapedirs = np.asarray(model_payload["shapedirs"].r, dtype=np.float64)
    betas = np.asarray(betas_override, dtype=np.float64) if betas_override is not None else cluster_means[family_index]
    rest_vertices = vertices_template + np.tensordot(
        shapedirs[:, :, : betas.shape[0]],
        betas,
        axes=([2], [0]),
    )
    j_regressor = model_payload["J_regressor"]
    rest_joints = np.asarray(j_regressor.dot(rest_vertices), dtype=np.float64)
    kintree_table = np.asarray(model_payload["kintree_table"], dtype=np.int64)
    parents = kintree_table[0].astype(np.int64)
    parents[0] = -1

    return OfficialSMALModel(
        vertices_template=vertices_template,
        faces=faces,
        weights=weights,
        posedirs=posedirs,
        shapedirs=shapedirs,
        j_regressor=j_regressor,
        kintree_table=kintree_table,
        parents=parents,
        betas=betas,
        rest_vertices=rest_vertices,
        rest_joints=rest_joints,
        joint_names=SMAL_JOINT_NAMES,
        source_path=model_path,
        data_path=data_path,
    )


def list_official_smal_controls(smal_model: OfficialSMALModel) -> None:
    for index, joint_name in enumerate(smal_model.joint_names):
        print(f"{index}\t{joint_name}")


def rest_bone_split_ratio(rest_joints: np.ndarray, start_index: int, mid_index: int, end_index: int) -> float:
    first = float(np.linalg.norm(rest_joints[mid_index] - rest_joints[start_index]))
    second = float(np.linalg.norm(rest_joints[end_index] - rest_joints[mid_index]))
    total = first + second
    if total < 1e-8:
        return 0.5
    return first / total


def cumulative_ratios(points: np.ndarray) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = cumulative[-1]
    if total < 1e-8:
        return np.linspace(0.0, 1.0, points.shape[0], dtype=np.float64)
    return cumulative / total


def interpolate_along_segment(
    start: np.ndarray,
    end: np.ndarray,
    ratio: float,
) -> np.ndarray:
    return (1.0 - ratio) * start + ratio * end


def build_smal_target_joints_from_frame(
    current: np.ndarray,
    current_body_points: dict[str, np.ndarray],
    rest_joints: np.ndarray,
    joint_index: dict[str, int],
    head_rotation_scene: np.ndarray | None = None,
    head_scale_override: float | None = None,
) -> np.ndarray:
    # We only observe 10 project joints directly from IMU. The denser SMAL joint set is
    # reconstructed by anchoring torso joints to guide points, splitting fore/hind limbs
    # with rest-pose bone ratios, and interpolating the tail along the rest skeleton.
    target = np.zeros((len(SMAL_JOINT_NAMES), 3), dtype=np.float64)

    pelvis_top = current_body_points["pelvis_top"]
    rib_top = current_body_points["rib_top"]
    belly_mid = current_body_points["belly_mid"]
    chest_lower = current_body_points["chest_lower"]
    stern = current[joint_index["stern"]]
    withers = current_body_points["withers"]
    neck = current_body_points["neck"]
    raw_head = current[joint_index["head"]]
    rest_head_offset = rest_joints[16] - rest_joints[15]
    rest_snout_offset = rest_joints[32] - rest_joints[15]
    rest_head_length = max(float(np.linalg.norm(rest_head_offset)), 1e-8)
    current_head_length = float(np.linalg.norm(raw_head - neck))
    head_scale = (
        current_head_length / rest_head_length
        if head_scale_override is None
        else float(head_scale_override)
    )

    if head_rotation_scene is None:
        snout_direction = normalize_vector(raw_head - neck, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        snout_length = head_scale * float(np.linalg.norm(rest_joints[32] - rest_joints[16]))
        head = raw_head
        snout = head + snout_direction * snout_length
    else:
        head = neck + head_scale * (head_rotation_scene @ rest_head_offset)
        snout = neck + head_scale * (head_rotation_scene @ rest_snout_offset)

    target[0] = pelvis_top
    target[1] = pelvis_top
    target[2] = interpolate_along_segment(pelvis_top, rib_top, 0.55)
    target[3] = interpolate_along_segment(pelvis_top, belly_mid, 0.85)
    target[4] = interpolate_along_segment(belly_mid, chest_lower, 0.55)
    target[5] = stern
    target[6] = withers
    target[15] = neck
    target[16] = head
    target[32] = snout

    left_shoulder = current_body_points["left_shoulder"]
    right_shoulder = current_body_points["right_shoulder"]
    left_elbow = current[joint_index["upper_arm_left"]]
    right_elbow = current[joint_index["upper_arm_right"]]
    left_front_paw = current[joint_index["left_hand"]]
    right_front_paw = current[joint_index["right_hand"]]

    left_front_ratio = rest_bone_split_ratio(rest_joints, 8, 9, 10)
    right_front_ratio = rest_bone_split_ratio(rest_joints, 12, 13, 14)

    target[7] = left_shoulder
    target[8] = left_elbow
    target[9] = interpolate_along_segment(left_elbow, left_front_paw, left_front_ratio)
    target[10] = left_front_paw
    target[11] = right_shoulder
    target[12] = right_elbow
    target[13] = interpolate_along_segment(right_elbow, right_front_paw, right_front_ratio)
    target[14] = right_front_paw

    left_hip = current_body_points["left_hip"]
    right_hip = current_body_points["right_hip"]
    left_knee = current[joint_index["upper_leg_left"]]
    right_knee = current[joint_index["upper_leg_right"]]
    left_back_paw = current[joint_index["left_foot"]]
    right_back_paw = current[joint_index["right_foot"]]

    left_hind_ratio = rest_bone_split_ratio(rest_joints, 18, 19, 20)
    right_hind_ratio = rest_bone_split_ratio(rest_joints, 22, 23, 24)

    target[17] = left_hip
    target[18] = left_knee
    target[19] = interpolate_along_segment(left_knee, left_back_paw, left_hind_ratio)
    target[20] = left_back_paw
    target[21] = right_hip
    target[22] = right_knee
    target[23] = interpolate_along_segment(right_knee, right_back_paw, right_hind_ratio)
    target[24] = right_back_paw

    tail_base = current_body_points["tail_base"]
    tail_tip = current_body_points["tail_tip"]
    tail_ratios = cumulative_ratios(rest_joints[25:32])
    for tail_offset, ratio in enumerate(tail_ratios):
        target[25 + tail_offset] = interpolate_along_segment(tail_base, tail_tip, float(ratio))

    return target


def fit_similarity_transform_inverse(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    dense_points = np.asarray(points, dtype=np.float64)
    dense_rotation = np.asarray(rotation, dtype=np.float64)
    dense_translation = np.asarray(translation, dtype=np.float64)
    centered = dense_points - dense_translation[None, :]
    return np.einsum("...i,ij->...j", centered, dense_rotation, optimize=True) / max(scale, 1e-8)


def align_vectors_rotation(
    source_vector: np.ndarray,
    target_vector: np.ndarray,
) -> np.ndarray:
    source = normalize_vector(source_vector, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    target = normalize_vector(target_vector, source)
    cross = np.cross(source, target)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = normalize_vector(
            np.cross(source, np.array([1.0, 0.0, 0.0], dtype=np.float64)),
            normalize_vector(
                np.cross(source, np.array([0.0, 1.0, 0.0], dtype=np.float64)),
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
            ),
        )
    else:
        axis = cross / cross_norm

    angle = float(np.arctan2(cross_norm, dot))
    return rotation_matrix_from_axis_angle(axis * angle)


def rotation_matrix_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = axis_angle / angle
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def estimate_rotation_from_vectors(
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
) -> np.ndarray:
    valid = (np.linalg.norm(source_vectors, axis=1) > 1e-8) & (np.linalg.norm(target_vectors, axis=1) > 1e-8)
    source = source_vectors[valid]
    target = target_vectors[valid]
    if source.shape[0] == 0:
        return np.eye(3, dtype=np.float64)
    if source.shape[0] == 1:
        return align_vectors_rotation(source[0], target[0])

    source = source / np.linalg.norm(source, axis=1, keepdims=True)
    target = target / np.linalg.norm(target, axis=1, keepdims=True)
    covariance = source.T @ target
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return rotation


def build_smal_children(parents: np.ndarray) -> list[list[int]]:
    children = [[] for _ in range(parents.shape[0])]
    for child_index in range(1, parents.shape[0]):
        parent_index = int(parents[child_index])
        if parent_index >= 0:
            children[parent_index].append(child_index)
    return children


def compute_smal_local_rotations(
    rest_joints: np.ndarray,
    target_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    children = build_smal_children(parents)
    global_rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], rest_joints.shape[0], axis=0)

    root_reference_indices = [6, 17, 21, 25]
    root_source = np.asarray([rest_joints[index] - rest_joints[0] for index in root_reference_indices], dtype=np.float64)
    root_target = np.asarray([target_joints[index] - target_joints[0] for index in root_reference_indices], dtype=np.float64)
    global_rotations[0] = estimate_rotation_from_vectors(root_source, root_target)

    for joint_index_smal in range(1, rest_joints.shape[0]):
        parent_index = int(parents[joint_index_smal])
        child_indices = children[joint_index_smal]

        if child_indices:
            source_vectors = np.asarray(
                [rest_joints[child] - rest_joints[joint_index_smal] for child in child_indices],
                dtype=np.float64,
            )
            target_vectors = np.asarray(
                [target_joints[child] - target_joints[joint_index_smal] for child in child_indices],
                dtype=np.float64,
            )
            global_rotations[joint_index_smal] = estimate_rotation_from_vectors(
                source_vectors=source_vectors,
                target_vectors=target_vectors,
            )
        else:
            global_rotations[joint_index_smal] = global_rotations[parent_index]

    local_rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], rest_joints.shape[0], axis=0)
    local_rotations[0] = global_rotations[0]
    for joint_index_smal in range(1, rest_joints.shape[0]):
        parent_index = int(parents[joint_index_smal])
        local_rotations[joint_index_smal] = global_rotations[parent_index].T @ global_rotations[joint_index_smal]

    return local_rotations


def with_zeros(matrix_3x4: np.ndarray) -> np.ndarray:
    return np.vstack([matrix_3x4, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)])


def pack_column(vector_4: np.ndarray) -> np.ndarray:
    packed = np.zeros((4, 4), dtype=np.float64)
    packed[:, 3] = vector_4
    return packed


def smal_lbs_vertices(
    rest_vertices: np.ndarray,
    rest_joints: np.ndarray,
    weights: np.ndarray,
    posedirs: np.ndarray,
    local_rotations: np.ndarray,
    parents: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    identity = np.eye(3, dtype=np.float64)
    pose_feature = (local_rotations[1:] - identity).reshape(-1)
    pose_offsets = np.tensordot(posedirs, pose_feature, axes=([2], [0]))
    posed_vertices = rest_vertices + pose_offsets

    joint_transforms = np.zeros((rest_joints.shape[0], 4, 4), dtype=np.float64)
    joint_transforms[0] = with_zeros(
        np.column_stack([local_rotations[0], rest_joints[0].reshape(3, 1)])
    )

    for joint_index_smal in range(1, rest_joints.shape[0]):
        parent_index = int(parents[joint_index_smal])
        joint_offset = (rest_joints[joint_index_smal] - rest_joints[parent_index]).reshape(3, 1)
        joint_transforms[joint_index_smal] = joint_transforms[parent_index] @ with_zeros(
            np.column_stack([local_rotations[joint_index_smal], joint_offset])
        )

    adjusted_transforms = joint_transforms.copy()
    for joint_index_smal in range(rest_joints.shape[0]):
        adjusted_transforms[joint_index_smal] = (
            joint_transforms[joint_index_smal]
            - pack_column(joint_transforms[joint_index_smal] @ np.append(rest_joints[joint_index_smal], 1.0))
        )

    blended = np.tensordot(weights, adjusted_transforms, axes=([1], [0]))
    homogeneous_vertices = np.column_stack([posed_vertices, np.ones((posed_vertices.shape[0], 1), dtype=np.float64)])
    deformed_vertices = np.einsum("vij,vj->vi", blended, homogeneous_vertices)[:, :3]
    return deformed_vertices + translation[None, :]


def build_official_smal_mesh_sequence(
    smal_model: OfficialSMALModel,
    positions: np.ndarray,
    body_points_by_name: dict[str, np.ndarray],
    head_rotation_matrices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    joint_index = {name: idx for idx, name in enumerate(JOINT_ORDER)}
    frame_count = positions.shape[0]
    target_joints_scene = np.zeros((frame_count, len(SMAL_JOINT_NAMES), 3), dtype=np.float64)
    neck_series = body_points_by_name["neck"]
    head_series = positions[:, joint_index["head"]]
    rest_head_length = max(float(np.linalg.norm(smal_model.rest_joints[16] - smal_model.rest_joints[15])), 1e-8)
    head_scale = float(np.median(np.linalg.norm(head_series - neck_series, axis=1))) / rest_head_length
    target_progress = ProgressPrinter("Preparing SMAL targets", frame_count)

    for frame_id in range(frame_count):
        current = positions[frame_id]
        current_body_points = {
            name: point_series[frame_id]
            for name, point_series in body_points_by_name.items()
        }
        target_joints_scene[frame_id] = build_smal_target_joints_from_frame(
            current=current,
            current_body_points=current_body_points,
            rest_joints=smal_model.rest_joints,
            joint_index=joint_index,
            head_rotation_scene=None if head_rotation_matrices is None else head_rotation_matrices[frame_id],
            head_scale_override=head_scale,
        )
        target_progress.update(frame_id + 1)

    deformed_vertices_scene = np.zeros(
        (frame_count, smal_model.rest_vertices.shape[0], 3),
        dtype=np.float64,
    )
    transformed_joints_scene = np.zeros_like(target_joints_scene)

    rest_reference = smal_model.rest_joints[list(SMAL_REFERENCE_JOINT_INDICES)]
    deform_progress = ProgressPrinter("Deforming SMAL mesh", frame_count)
    for frame_id in range(frame_count):
        target_reference = target_joints_scene[frame_id, list(SMAL_REFERENCE_JOINT_INDICES)]
        scene_scale, scene_rotation, scene_translation = fit_similarity_transform(
            source_points=rest_reference,
            target_points=target_reference,
        )
        transformed_vertices = apply_similarity_transform(
            points=smal_model.rest_vertices,
            scale=scene_scale,
            rotation=scene_rotation,
            translation=scene_translation,
        )
        transformed_joints = apply_similarity_transform(
            points=smal_model.rest_joints,
            scale=scene_scale,
            rotation=scene_rotation,
            translation=scene_translation,
        )
        residuals = target_joints_scene[frame_id] - transformed_joints
        effective_residuals = residuals.copy()
        effective_residuals[list(SMAL_DISABLED_RESIDUAL_JOINT_INDICES)] = 0.0
        deformed_vertices_scene[frame_id] = apply_inverse_distance_residuals(
            vertices=transformed_vertices,
            control_points=transformed_joints,
            residuals=effective_residuals,
            power=1.8,
        )
        transformed_joints_scene[frame_id] = transformed_joints + effective_residuals
        deform_progress.update(frame_id + 1)

    return deformed_vertices_scene, transformed_joints_scene


def render_animation(
    output_path: Path,
    frame_indices: np.ndarray,
    packet_counter: np.ndarray,
    positions: np.ndarray,
    body_points_by_name: dict[str, np.ndarray],
    back_line: np.ndarray,
    belly_line: np.ndarray,
    root_translation: np.ndarray,
    rotation_matrices: np.ndarray,
    fps: int,
    point_size: float,
    body_model: str,
    show_skeleton_overlay: bool,
    frame_action_labels: np.ndarray | None = None,
    smal_vertices: np.ndarray | None = None,
    smal_faces: np.ndarray | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    joint_index = {name: idx for idx, name in enumerate(JOINT_ORDER)}
    local_torso_faces = build_local_torso_faces() if body_model == "shell" else []
    overlay_enabled = body_model == "shell" or show_skeleton_overlay
    overlay_alpha = 0.95 if body_model == "shell" else 0.18
    scatter = None
    if overlay_enabled:
        scatter = ax.scatter(
            [],
            [],
            [],
            s=point_size * (0.45 if body_model == "shell" else 0.28),
            c="#c2410c",
            depthshade=True,
            alpha=overlay_alpha,
        )
    torso_collection = None
    limb_collection = None
    head_collection = None
    ear_collection = None
    smal_collection = None

    if body_model == "shell":
        torso_collection = Poly3DCollection(
            [],
            facecolors="#8b6b4a",
            edgecolors="none",
            alpha=0.36,
        )
        limb_collection = Poly3DCollection(
            [],
            facecolors="#a67c52",
            edgecolors="none",
            alpha=0.28,
        )
        head_collection = Poly3DCollection(
            [],
            facecolors="#6f4e37",
            edgecolors="none",
            alpha=0.42,
        )
        ear_collection = Poly3DCollection(
            [],
            facecolors="#5b3a29",
            edgecolors="none",
            alpha=0.60,
        )
        ax.add_collection3d(torso_collection)
        ax.add_collection3d(limb_collection)
        ax.add_collection3d(head_collection)
        ax.add_collection3d(ear_collection)
        set_equal_axes(ax, positions)
    else:
        if smal_vertices is None or smal_faces is None:
            raise ValueError("SMAL rendering requires both smal_vertices and smal_faces")
        smal_collection = Poly3DCollection(
            [],
            facecolors="#9c7a56",
            edgecolors="none",
            alpha=0.62,
        )
        ax.add_collection3d(smal_collection)
        set_equal_axes(ax, smal_vertices)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Relative Dog Skeleton from IMU ({body_model})")
    action_text_artist = None
    if frame_action_labels is not None:
        action_text_artist = fig.text(
            0.5,
            0.97,
            "",
            ha="center",
            va="top",
            fontsize=11,
            linespacing=1.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 4},
        )
    ax.view_init(elev=18, azim=-60)
    ax.grid(True, alpha=0.35)

    def update(frame_id: int):
        current = positions[frame_id]
        current_body_points = {
            name: point_series[frame_id]
            for name, point_series in body_points_by_name.items()
        }
        if body_model == "shell":
            assert torso_collection is not None
            assert limb_collection is not None
            assert head_collection is not None
            assert ear_collection is not None
            torso_faces = transform_faces(
                local_faces=local_torso_faces,
                rotation_matrix=rotation_matrices[frame_id],
                translation=root_translation[frame_id],
            )
            limb_faces, head_faces, ear_faces = build_body_meshes(
                current=current,
                current_body_points=current_body_points,
                joint_index=joint_index,
            )
            torso_collection.set_verts(torso_faces)
            limb_collection.set_verts(limb_faces)
            head_collection.set_verts(head_faces)
            ear_collection.set_verts(ear_faces)
        else:
            assert smal_collection is not None
            assert smal_vertices is not None
            assert smal_faces is not None
            smal_collection.set_verts(smal_vertices[frame_id][smal_faces])

        if overlay_enabled:
            assert scatter is not None
            scatter._offsets3d = (
                current[:, 0],
                current[:, 1],
                current[:, 2],
            )
        ax.set_title(
            f"Relative Dog Skeleton from IMU ({body_model}) | frame={int(frame_indices[frame_id])} "
            f"| packet={int(packet_counter[frame_id])}"
        )
        if action_text_artist is not None:
            action_text_artist.set_text(str(frame_action_labels[frame_id]))
        artists = []
        if overlay_enabled:
            assert scatter is not None
            artists.append(scatter)
        if body_model == "shell":
            artists.extend(
                [
                    torso_collection,
                    limb_collection,
                    head_collection,
                    ear_collection,
                ]
            )
        else:
            artists.append(smal_collection)
        if action_text_artist is not None:
            artists.append(action_text_artist)
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / max(fps, 1),
        blit=False,
    )
    render_progress = ProgressPrinter("Rendering animation", len(frame_indices))
    progress_callback = lambda current_frame, total_frames: render_progress.update(current_frame + 1)

    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        animation.save(output_path, writer=PillowWriter(fps=fps), progress_callback=progress_callback)
    elif suffix == ".mp4":
        animation.save(output_path, writer=FFMpegWriter(fps=fps), progress_callback=progress_callback)
    else:
        raise ValueError("Video output must end with .mp4 or .gif")

    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.body_model == "smal" and args.smal_model is None:
        raise ValueError("--smal-model is required when --body-model=smal")
    if args.list_smal_controls:
        if args.smal_model is None:
            raise ValueError("--smal-model is required when --list-smal-controls is used")
        if args.smal_model.suffix.lower() == ".pkl":
            resolved_model_path = resolve_official_smal_model_path(args.smal_model)
            smal_model = load_official_smal_model(
                model_path=resolved_model_path,
                data_path=resolve_smal_data_path(resolved_model_path, args.smal_data),
                family_index=args.smal_family_index,
            )
            list_official_smal_controls(smal_model)
        else:
            template_mesh = load_template_mesh_model(args.smal_model)
            for index, control_label in enumerate(template_mesh.control_labels):
                print(f"{index}\t{control_label}")
        return

    packet_counter, sensor_data_by_joint, sample_rate_hz, resolved_input_format, frame_action_labels = load_motion_data(
        data_dir=args.data_dir,
        input_format=args.input_format,
        target_rate_hz=args.target_rate_hz,
    )
    relative_positions = build_relative_positions(
        sensor_data_by_joint=sensor_data_by_joint,
        neutral_pose_mode=args.neutral_pose_mode,
        gyr_motion_scale=args.gyr_motion_scale,
    )
    (
        positions,
        root_translation,
        rotation_matrices,
        body_points_by_name,
        back_line,
        belly_line,
        heading_offset,
    ) = build_scene_motion(
        relative_positions=relative_positions,
        sensor_data_by_joint=sensor_data_by_joint,
        root_motion_mode=args.root_motion_mode,
        translation_scale=args.translation_scale,
        vertical_motion_scale=args.vertical_motion_scale,
        sample_rate_hz=sample_rate_hz,
    )
    joint_is_interpolated = stack_joint_is_interpolated(sensor_data_by_joint)
    root_heading_6d = rotation_matrices_to_rot6d(rotation_matrices)

    requested_start_frame = args.start_frame
    requested_num_frames = args.num_frames
    active_window_score = None

    if requested_start_frame is None:
        if requested_num_frames is not None and args.window_selection == "most-active":
            requested_start_frame, _, active_window_score = find_most_active_window(
                positions=positions,
                window_frames=requested_num_frames,
                search_step=args.search_step,
            )
        else:
            requested_start_frame = 0

    frame_indices, selected_packet_counter, selected_positions = select_frames(
        packet_counter=packet_counter,
        positions=positions,
        start_frame=requested_start_frame,
        num_frames=requested_num_frames,
        stride=args.stride,
    )
    _, _, selected_relative_positions = select_frames(
        packet_counter=packet_counter,
        positions=relative_positions,
        start_frame=requested_start_frame,
        num_frames=requested_num_frames,
        stride=args.stride,
    )
    _, _, selected_root_translation = select_frames(
        packet_counter=packet_counter,
        positions=root_translation[:, None, :],
        start_frame=requested_start_frame,
        num_frames=requested_num_frames,
        stride=args.stride,
    )
    selected_root_translation = selected_root_translation[:, 0, :]
    _, _, selected_root_heading_6d = select_frames(
        packet_counter=packet_counter,
        positions=root_heading_6d[:, None, :],
        start_frame=requested_start_frame,
        num_frames=requested_num_frames,
        stride=args.stride,
    )
    selected_root_heading_6d = selected_root_heading_6d[:, 0, :]
    _, _, selected_is_interpolated = select_frames(
        packet_counter=packet_counter,
        positions=joint_is_interpolated.astype(np.float64)[:, :, None],
        start_frame=requested_start_frame,
        num_frames=requested_num_frames,
        stride=args.stride,
    )
    selected_is_interpolated = selected_is_interpolated[:, :, 0] > 0.5
    selected_frame_action_labels = (
        frame_action_labels[frame_indices] if frame_action_labels is not None else None
    )
    clip_translation_offset = selected_root_translation[0].copy()
    selected_root_translation = selected_root_translation - clip_translation_offset
    selected_positions = selected_positions - clip_translation_offset[None, None, :]
    selected_rotation_matrices = rotation_matrices[frame_indices]
    head_local_delta = relative_segment_delta(
        root_quat=sensor_data_by_joint["stern"]["quat"],
        segment_quat=sensor_data_by_joint["head"]["quat"],
        neutral_pose_mode=args.neutral_pose_mode,
    )
    selected_head_local_rotation_matrices = quaternion_to_rotation_matrices(head_local_delta[frame_indices])
    selected_head_scene_rotation_matrices = np.einsum(
        "tij,tjk->tik",
        selected_rotation_matrices,
        selected_head_local_rotation_matrices,
        optimize=True,
    )
    selected_body_points_by_name = {
        name: point_series[frame_indices] - clip_translation_offset[None, :]
        for name, point_series in body_points_by_name.items()
    }
    selected_back_line = back_line[frame_indices] - clip_translation_offset[None, None, :]
    selected_belly_line = belly_line[frame_indices] - clip_translation_offset[None, None, :]
    selected_smal_vertices = None
    smal_faces = None
    bound_template_mesh = None
    official_smal_model = None
    smal_preset = None

    if args.body_model == "smal":
        if args.smal_model.suffix.lower() == ".pkl":
            resolved_model_path = resolve_official_smal_model_path(args.smal_model)
            preset_payload = load_pickle_with_fake_chumpy(args.smal_model)
            if is_smal_preset_payload(preset_payload):
                smal_preset = load_smal_preset(args.smal_model)
            official_smal_model = load_official_smal_model(
                model_path=resolved_model_path,
                data_path=resolve_smal_data_path(resolved_model_path, args.smal_data),
                family_index=args.smal_family_index,
                betas_override=None if smal_preset is None else smal_preset.beta,
            )
            selected_smal_vertices, _ = build_official_smal_mesh_sequence(
                smal_model=official_smal_model,
                positions=selected_positions,
                body_points_by_name=selected_body_points_by_name,
                head_rotation_matrices=selected_head_scene_rotation_matrices,
            )
            smal_faces = official_smal_model.faces
        else:
            template_mesh = load_template_mesh_model(args.smal_model)
            mapping = load_smal_mapping(args.smal_mapping)
            bound_template_mesh = bind_template_mesh_model(
                template_mesh=template_mesh,
                mapping=mapping,
            )
            selected_smal_vertices = build_template_mesh_sequence(
                template_mesh=bound_template_mesh,
                positions=selected_positions,
                body_points_by_name=selected_body_points_by_name,
            )
            smal_faces = bound_template_mesh.faces

    if args.coords_output is not None:
        save_coordinates(
            output_path=args.coords_output,
            frame_indices=frame_indices,
            packet_counter=selected_packet_counter,
            positions=selected_positions,
            relative_positions=selected_relative_positions,
            root_translation=selected_root_translation,
            root_heading_6d=selected_root_heading_6d,
            is_interpolated=selected_is_interpolated,
            frame_action_labels=selected_frame_action_labels,
        )

    audit_metrics = compute_pose_audit(
        relative_positions=selected_relative_positions,
        is_interpolated=selected_is_interpolated,
    )
    if args.audit_output is not None:
        save_preview_audit(
            output_path=args.audit_output,
            metadata=infer_segment_metadata(args.data_dir),
            frame_indices=frame_indices,
            packet_counter=selected_packet_counter,
            sample_rate_hz=sample_rate_hz,
            audit_metrics=audit_metrics,
        )

    if args.video_output is not None:
        render_animation(
            output_path=args.video_output,
            frame_indices=frame_indices,
            packet_counter=selected_packet_counter,
            positions=selected_positions,
            body_points_by_name=selected_body_points_by_name,
            back_line=selected_back_line,
            belly_line=selected_belly_line,
            root_translation=selected_root_translation,
            rotation_matrices=selected_rotation_matrices,
            fps=args.fps,
            point_size=args.point_size,
            body_model=args.body_model,
            show_skeleton_overlay=args.show_skeleton_overlay,
            frame_action_labels=selected_frame_action_labels,
            smal_vertices=selected_smal_vertices,
            smal_faces=smal_faces,
        )

    print(
        "Loaded {0} frames from {1} project joints. Selected {2} frames for export.".format(
            len(packet_counter),
            len(sensor_data_by_joint),
            len(frame_indices),
        )
    )
    print(f"Input format: {resolved_input_format}")
    print(f"Sample rate: {sample_rate_hz:.2f} Hz")
    print(
        "Frame window: start={0}, end={1}, stride={2}".format(
            int(frame_indices[0]),
            int(frame_indices[-1]),
            args.stride,
        )
    )
    if active_window_score is not None:
        print(
            "Window selection: most-active (mean joint displacement per frame = {0:.5f})".format(
                active_window_score
            )
        )
    print(f"Root motion mode: {args.root_motion_mode}")
    print(f"Body model: {args.body_model}")
    print(f"Neutral pose mode: {args.neutral_pose_mode}")
    print(f"Gyro motion scale: {args.gyr_motion_scale:.2f}")
    if selected_frame_action_labels is not None:
        unique_preview_labels = sorted({str(label) for label in selected_frame_action_labels.tolist()})
        print(f"Frame action labels: {len(unique_preview_labels)} unique labels in selected window")
    print(
        "Preview audit: status={0}, finite_ratio={1:.4f}, interp_ratio={2:.4f}, "
        "max_joint_interp_ratio={3:.4f}, worst_interp_joint={4}, "
        "left_right_sign_consistency={5:.4f}, distal_below_parent_consistency={6:.4f}".format(
            audit_metrics["preview_status"],
            audit_metrics["finite_ratio"],
            audit_metrics["interp_ratio"],
            audit_metrics["max_joint_interp_ratio"],
            audit_metrics["worst_interp_joint"],
            audit_metrics["left_right_sign_consistency"],
            audit_metrics["distal_below_parent_consistency"],
        )
    )
    if official_smal_model is not None:
        print(
            "Official SMAL: {0} | data={1} | family_index={2}".format(
                official_smal_model.source_path,
                official_smal_model.data_path,
                args.smal_family_index,
            )
        )
    if smal_preset is not None:
        print(
            "SMAL preset: {0} | beta_dim={1} | preset_pose_ignored=True".format(
                smal_preset.source_path,
                smal_preset.beta.shape[0],
            )
        )
    if bound_template_mesh is not None:
        print(
            "SMAL template: {0} | matched controls: {1}".format(
                bound_template_mesh.source_path,
                len(bound_template_mesh.source_labels),
            )
        )
    print(f"Heading-motion residual offset: {np.degrees(heading_offset):.2f} deg")
    if args.coords_output is not None:
        print(f"Saved coordinates to: {args.coords_output}")
    if args.video_output is not None:
        print(f"Saved animation to: {args.video_output}")
    if args.audit_output is not None:
        print(f"Saved preview audit to: {args.audit_output}")


if __name__ == "__main__":
    main()
