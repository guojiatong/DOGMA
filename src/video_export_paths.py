#!/usr/bin/env python3
"""
Helpers for consistent video export paths.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence


VIDEO_SUBDIR_NAME = "videos"
_UNSAFE_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def slugify_export_token(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "na"
    text = text.replace("/", "-").replace("\\", "-")
    text = _UNSAFE_TOKEN_PATTERN.sub("-", text)
    text = text.strip("-._")
    return text or "na"


def build_export_video_path(
    *,
    output_dir: Path,
    source_tag: str,
    participant: str | None,
    segment_id: str | int | None,
    start_frame_20hz: int | None,
    rank: int | None = None,
    extra_tags: Sequence[object] | None = None,
    suffix: str = ".mp4",
) -> Path:
    if suffix not in {".mp4", ".gif"}:
        raise ValueError(f"Unsupported video suffix: {suffix}")

    videos_dir = Path(output_dir) / VIDEO_SUBDIR_NAME
    videos_dir.mkdir(parents=True, exist_ok=True)

    tokens: list[str] = []
    if rank is not None:
        tokens.append(f"{int(rank):02d}")
    tokens.append(slugify_export_token(source_tag))
    for tag in extra_tags or ():
        if tag is None:
            continue
        token = slugify_export_token(tag)
        if token != "na":
            tokens.append(token)
    if participant not in (None, ""):
        tokens.append(slugify_export_token(participant))
    if segment_id not in (None, ""):
        tokens.append(f"segment_{slugify_export_token(segment_id)}")
    if start_frame_20hz is not None:
        tokens.append(f"start_{int(start_frame_20hz)}")
    return videos_dir / ("__".join(tokens) + suffix)
