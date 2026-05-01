#!/usr/bin/env python3
"""
Fill missing internal packet gaps for IMU_New2 sensor pickles.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from imu_new2_common import fill_sensor_packet_gaps, read_sensor_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill internal loss packets in Data/IMU_New2 and export canonical pkl files.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("Data/IMU_New2"),
        help="Input IMU_New2 root. Default: Data/IMU_New2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data/IMU_New2_Filled"),
        help="Output root for filled IMU pickles. Default: Data/IMU_New2_Filled",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=None,
        help="Optional audit csv path. Default: <output-root>/fill_packets_audit.csv",
    )
    return parser.parse_args()


def process_dataset(input_root: Path, output_root: Path, audit_csv_path: Path) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    participant_dirs = sorted(path for path in input_root.glob("*") if path.is_dir())
    for participant_dir in participant_dirs:
        segment_dirs = sorted(
            (path for path in participant_dir.glob("*") if path.is_dir()),
            key=lambda path: int(path.name),
        )
        for segment_dir in segment_dirs:
            output_segment_dir = output_root / participant_dir.name / segment_dir.name
            output_segment_dir.mkdir(parents=True, exist_ok=True)

            for mtb_path in sorted(segment_dir.glob("*.mtb")):
                shutil.copy2(mtb_path, output_segment_dir / mtb_path.name)

            sensor_paths = sorted(segment_dir.glob("*.pkl"))
            for sensor_path in sensor_paths:
                observed_frame, schema_type = read_sensor_pickle(sensor_path)
                filled_frame = fill_sensor_packet_gaps(observed_frame)
                output_path = output_segment_dir / sensor_path.name
                filled_frame.to_pickle(output_path)

                packet_start = int(filled_frame["packet_counter"].iloc[0]) if not filled_frame.empty else -1
                packet_end = int(filled_frame["packet_counter"].iloc[-1]) if not filled_frame.empty else -1
                rows.append(
                    {
                        "imu_participant": participant_dir.name,
                        "segment_id": segment_dir.name,
                        "sensor_id": sensor_path.stem.split("_")[-1],
                        "schema_type": schema_type,
                        "raw_rows": int(observed_frame.shape[0]),
                        "filled_rows": int(filled_frame.shape[0]),
                        "packet_start": packet_start,
                        "packet_end": packet_end,
                        "num_missing": int(filled_frame["is_interpolated"].sum()),
                    }
                )

    with open(audit_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "imu_participant",
                "segment_id",
                "sensor_id",
                "schema_type",
                "raw_rows",
                "filled_rows",
                "packet_start",
                "packet_end",
                "num_missing",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root
    audit_csv_path = args.audit_csv or (output_root / "fill_packets_audit.csv")

    process_dataset(input_root=input_root, output_root=output_root, audit_csv_path=audit_csv_path)
    print(f"Wrote filled IMU data to {output_root}")
    print(f"Wrote audit csv to {audit_csv_path}")


if __name__ == "__main__":
    main()
